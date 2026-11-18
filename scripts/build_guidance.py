#!/usr/bin/env python3
"""Build a compact WFO MOB NBM/NDFD MaxT/MinT guidance dataset."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
from shapely.geometry import shape, mapping, box
from scipy.spatial import cKDTree
from botocore import UNSIGNED
from botocore.client import Config
from eccodes import (
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
)

SITE = Path("site")
DATA = SITE / "data"
OUT = DATA / "guidance.json"

NBM_BUCKET = "noaa-nbm-grib2-pds"
NBM_CYCLES = (1, 7, 13, 19)
PERCENTILES = (10, 25, 50, 75, 90)
WFO = "MOB"
SAMPLE_SPACING = 0.025
MAX_VALID_PERIODS = 5

PUBLIC_ZONES_URL = (
    "https://mapservices.weather.noaa.gov/static/rest/services/"
    "nws_reference_maps/nws_reference_map/FeatureServer/8/query"
)
MARINE_ZONES_URL = (
    "https://mapservices.weather.noaa.gov/static/rest/services/"
    "nws_reference_maps/nws_reference_map/FeatureServer/5/query"
)
NDFD_BASE = (
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/"
    "AR.conus/VP.001-003"
)
USER_AGENT = "NBM-NDFD-Verification/1.0 (+https://github.com/RED112482/NBM-NDFD-Verification)"

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def fetch_json(url: str, timeout: int = 90):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def safe_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def point_in_ring(x, y, ring):
    inside = False
    if not ring:
        return False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_geom(x, y, geom):
    if not geom:
        return False
    typ = geom.get("type")
    coords = geom.get("coordinates") or []
    polys = [coords] if typ == "Polygon" else coords if typ == "MultiPolygon" else []
    for poly in polys:
        if not poly or not point_in_ring(x, y, poly[0]):
            continue
        if any(point_in_ring(x, y, hole) for hole in poly[1:]):
            continue
        return True
    return False


def feature_bounds(features):
    xs, ys = [], []

    def walk(v):
        if (
            isinstance(v, (list, tuple))
            and len(v) >= 2
            and isinstance(v[0], (int, float))
            and isinstance(v[1], (int, float))
        ):
            xs.append(float(v[0]))
            ys.append(float(v[1]))
        elif isinstance(v, (list, tuple)):
            for q in v:
                walk(q)

    for f in features:
        walk((f.get("geometry") or {}).get("coordinates") or [])
    return min(xs), min(ys), max(xs), max(ys)


def fetch_boundaries():
    land_params = {
        "where": "cwa='MOB'",
        "outFields": "state,cwa,zone,name,state_zone,shortname",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    marine_params = {
        "where": "wfo='MOB'",
        "outFields": "wfo,name,id",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    land = fetch_json(PUBLIC_ZONES_URL + "?" + urllib.parse.urlencode(land_params))
    marine = fetch_json(MARINE_ZONES_URL + "?" + urllib.parse.urlencode(marine_params))
    for coll in (land, marine):
        for feat in coll.get("features", []):
            feat["properties"] = {
                str(k).lower(): v for k, v in (feat.get("properties") or {}).items()
            }
    # NWS operational boundary layers: county=2, states=3, CWA=1.
    reference = "https://mapservices.weather.noaa.gov/static/rest/services/nws_reference_maps/nws_reference_map/FeatureServer"
    outlines = {}
    for name, layer_id, where in (
        ("counties", 2, "cwa='MOB'"),
        ("states", 3, "state IN ('AL','FL','MS')"),
        ("cwa", 1, "cwa='MOB'"),
    ):
        try:
            params = {"where": where, "outFields": "*", "returnGeometry": "true",
                      "outSR": "4326", "f": "geojson"}
            outlines[name] = fetch_json(
                f"{reference}/{layer_id}/query?" + urllib.parse.urlencode(params)
            )
        except Exception as exc:
            print(f"WARNING: {name} outlines unavailable: {exc}")
            outlines[name] = {"type": "FeatureCollection", "features": []}
    # Keep the original high-resolution official zones for analysis/sampling.
    # Rendering simplification must NEVER change which grid nodes are land/marine.
    full_land = json.loads(json.dumps(land))
    full_marine = json.loads(json.dumps(marine))
    # Clip and simplify outlines for the MOB map before sending them to browsers.
    # Keeping entire AL/FL/MS polygon coastlines can exceed 10 MB and stall maps.
    wb,sb,eb,nb=feature_bounds((land.get("features") or [])+(marine.get("features") or []))
    viewport=box(wb-.18,sb-.18,eb+.18,nb+.18)
    for collection in (land,marine,*outlines.values()):
        compact=[]
        for feat in collection.get("features",[]):
            try:
                geom=shape(feat.get("geometry") or {})
                cut=geom.intersection(viewport).simplify(.006,preserve_topology=True)
                if cut.is_empty:continue
                feat["geometry"]=mapping(cut)
                compact.append(feat)
            except Exception as exc:
                print("WARNING: couldn't simplify boundary:",str(exc)[:160])
        collection["features"]=compact
    return land, marine, outlines, full_land, full_marine


def build_sample_points(land, marine):
    land_geoms = [f.get("geometry") or {} for f in land.get("features", [])]
    marine_geoms = [f.get("geometry") or {} for f in marine.get("features", [])]
    bounds = feature_bounds((land.get("features") or []) + (marine.get("features") or []))
    xmin, ymin, xmax, ymax = bounds
    points = []
    lat = ymin + SAMPLE_SPACING / 2
    idx = 0
    while lat <= ymax:
        lon = xmin + SAMPLE_SPACING / 2
        while lon <= xmax:
            is_land = any(point_in_geom(lon, lat, g) for g in land_geoms)
            is_marine = any(point_in_geom(lon, lat, g) for g in marine_geoms)
            if is_land or is_marine:
                points.append(
                    {
                        "id": idx,
                        "lat": round(lat, 5),
                        "lon": round(lon, 5),
                        "land": is_land,
                        "marine": is_marine,
                    }
                )
                idx += 1
            lon += SAMPLE_SPACING
        lat += SAMPLE_SPACING
    return points, bounds


def latest_cycle_occurrences(now):
    # Give each NBM cycle some time to finish arriving in the public bucket.
    cutoff = now - timedelta(hours=2)
    out = []
    for hh in NBM_CYCLES:
        candidate = cutoff.replace(hour=hh, minute=0, second=0, microsecond=0)
        if candidate > cutoff:
            candidate -= timedelta(days=1)
        out.append(candidate)
    return sorted(out)


def valid_targets(now):
    start = now.replace(minute=0, second=0, microsecond=0)
    result = {"mint": [], "maxt": []}
    for product, hour in (("mint", 12), ("maxt", 0)):
        day = start.date()
        dt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        if dt <= start:
            dt += timedelta(days=1)
        for _ in range(MAX_VALID_PERIODS):
            result[product].append(dt)
            dt += timedelta(days=1)
    return result


def object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def nbm_key(init_dt, lead, family):
    ds = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")
    fh = f"{lead:03d}"
    return f"blend.{ds}/{hh}/{family}/blend.t{hh}z.{family}.f{fh}.co.grib2"


def idx_lines(bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key + ".idx")["Body"].read()
    return body.decode("utf-8", "replace").splitlines()


def byte_range(lines, i, object_size):
    start = int(lines[i].split(":", 2)[1])
    end = (
        int(lines[i + 1].split(":", 2)[1]) - 1
        if i + 1 < len(lines)
        else object_size - 1
    )
    return start, end


def decode_message_bytes(body: bytes):
    fd, path = tempfile.mkstemp(suffix=".grib2")
    os.close(fd)
    try:
        Path(path).write_bytes(body)
        with open(path, "rb") as fh:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                raise RuntimeError("Unable to decode GRIB message")
            try:
                vals = np.asarray(codes_get_array(gid, "values"), dtype=float)
                lats = np.asarray(codes_get_array(gid, "latitudes"), dtype=float)
                lons = np.asarray(codes_get_array(gid, "longitudes"), dtype=float)
                meta = {}
                for name in (
                    "shortName",
                    "name",
                    "units",
                    "stepType",
                    "percentileValue",
                    "validityDate",
                    "validityTime",
                    "dataDate",
                    "dataTime",
                    "forecastTime",
                    "startStep",
                    "endStep",
                    "stepRange",
                ):
                    try:
                        meta[name] = codes_get(gid, name)
                    except Exception:
                        pass
            finally:
                codes_release(gid)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    units = str(meta.get("units") or "")
    if units == "K" or "kelvin" in units.lower():
        vals = (vals - 273.15) * 9 / 5 + 32
    vals = np.where(np.isfinite(vals) & (vals > -120) & (vals < 160), vals, np.nan)
    lons = np.where(lons > 180, lons - 360, lons)
    return vals, lats, lons, meta


def fetch_idx_message(bucket, key, lines, i):
    size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    a, b = byte_range(lines, i, size)
    body = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={a}-{b}")["Body"].read()
    return decode_message_bytes(body)


def meta_valid(meta):
    try:
        ds = str(int(meta["validityDate"])).zfill(8)
        tm = str(int(meta.get("validityTime", 0))).zfill(4)
        return datetime.strptime(ds + tm[-4:], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def field_candidates(lines, product):
    token = ":TMIN:" if product == "mint" else ":TMAX:"
    return [i for i, line in enumerate(lines) if token in line.upper()]


def read_nbm_field(init_dt, valid_dt, product):
    lead = int((valid_dt - init_dt).total_seconds() // 3600)
    if lead <= 0 or lead > 264:
        return None

    output = {"det": None, "percentiles": {}, "leadHours": lead}
    core = nbm_key(init_dt, lead, "core")
    # NBM QMD cycles are 00/06/12/18Z, typically disseminated 6-8h
    # after initialization. Only consider cycles available BEFORE the
    # corresponding NBM text/product comparison cycle. This prevents
    # using guidance that didn't yet exist at 00/12Z NDFD issuance.
    qmd_cutoff = init_dt - timedelta(hours=8)
    qmd_init = qmd_cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
    for hour in (0, 6, 12, 18):
        candidate = qmd_cutoff.replace(hour=hour)
        if candidate <= qmd_cutoff: qmd_init = candidate
    # CONUS QMD MinT/MaxT are 18h extrema ending roughly 6h
    # after NDFD's conventional 12Z/00Z display slot.
    qmd_expected_valid = valid_dt + timedelta(hours=6)
    qmd_lead = int((qmd_expected_valid - qmd_init).total_seconds() // 3600)
    qmd = nbm_key(qmd_init, qmd_lead, "qmd")
    output["qmdCycleUtc"] = qmd_init.isoformat().replace("+00:00", "Z")
    output["qmdValidUtc"] = qmd_expected_valid.isoformat().replace("+00:00", "Z")

    if object_exists(NBM_BUCKET, core) and object_exists(NBM_BUCKET, core + ".idx"):
        lines = idx_lines(NBM_BUCKET, core)
        for i in field_candidates(lines, product):
            try:
                msg = fetch_idx_message(NBM_BUCKET, core, lines, i)
                if meta_valid(msg[3]) != valid_dt:
                    continue
                # Deterministic message should not carry a percentile.
                if msg[3].get("percentileValue") not in (None, "missing"):
                    continue
                output["det"] = msg
                break
            except Exception:
                continue

    if object_exists(NBM_BUCKET, qmd) and object_exists(NBM_BUCKET, qmd + ".idx"):
        lines = idx_lines(NBM_BUCKET, qmd)
        # CONUS QMD probabilistic extrema are encoded as 'TMP' with an
        # 18-hour min/max statistical process, *not* as TMIN/TMAX GRIB names.
        # Select only actual percentile-level messages (exclude probability
        # exceedance thresholds and instantaneous TMP percentiles).
        extreme = "MIN" if product == "mint" else "MAX"
        qmd_candidates = [
            i for i, line in enumerate(lines)
            if ":TMP:" in line.upper()
            and f" HOUR {extreme} FCST:" in line.upper()
            and re.search(r":(10|25|50|75|90)% level", line, re.I)
        ]
        if not qmd_candidates and os.environ.get("DEBUG_QMD", "0") == "1":
            print("No QMD 18h temperature percentiles for", qmd, product)
        for i in qmd_candidates:
            try:
                msg = fetch_idx_message(NBM_BUCKET, qmd, lines, i)
                if meta_valid(msg[3]) != qmd_expected_valid:
                    continue
                pct = safe_float(msg[3].get("percentileValue"))
                if pct not in PERCENTILES:
                    # GRIB2 index is the authoritative percentile inventory.
                    found = re.search(r":(10|25|50|75|90)% level", lines[i], re.I)
                    pct = float(found.group(1)) if found else None
                if pct is not None and int(round(pct)) in PERCENTILES:
                    output["percentiles"][int(round(pct))] = msg
            except Exception:
                continue
    if output["det"] is None and not output["percentiles"]:
        return None
    return output


def iter_grib_messages(raw: bytes):
    pos = 0
    while True:
        a = raw.find(b"GRIB", pos)
        if a < 0:
            return
        z = raw.find(b"7777", a)
        if z < 0:
            return
        yield raw[a : z + 4]
        pos = z + 4


def read_ndfd_product(product):
    url = f"{NDFD_BASE}/ds.{product}.bin"
    raw = fetch_bytes(url)
    out = {}
    for chunk in iter_grib_messages(raw):
        try:
            msg = decode_message_bytes(chunk)
            valid = meta_valid(msg[3])
            if valid:
                out[valid.isoformat().replace("+00:00", "Z")] = msg
        except Exception:
            continue
    return out, url


def sample_grib(grib, sample_points):
    vals, lats, lons, _ = grib
    plat = np.asarray([p["lat"] for p in sample_points], dtype=float)
    plon = np.asarray([p["lon"] for p in sample_points], dtype=float)
    mask = (
        np.isfinite(vals)
        & (lats >= plat.min() - 0.3)
        & (lats <= plat.max() + 0.3)
        & (lons >= plon.min() - 0.3)
        & (lons <= plon.max() + 0.3)
    )
    sv, sy, sx = vals[mask], lats[mask], lons[mask]
    if not len(sv):
        return [None] * len(sample_points)
    # One spatial index per GRIB field instead of 1,100 repeated full-grid
    # distance scans. Essential for 5 NBM percentile maps per valid period.
    scale = math.cos(math.radians(float(np.mean(plat))))
    tree = cKDTree(np.column_stack((sy, sx * scale)))
    query = np.column_stack((plat, plon * scale))
    # Inverse-distance interpolation from the four closest *native* GRIB
    # nodes avoids nearest-neighbor stair-stepping / artificial banding.
    distances, indices = tree.query(query, k=4)
    weights = 1.0 / np.maximum(distances, 1e-9) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    sampled = np.sum(sv[indices] * weights, axis=1)
    return [round(float(v), 1) if np.isfinite(v) else None for v in sampled]


def stats(values, sample_points, mode):
    selected = []
    for v, p in zip(values, sample_points):
        if v is None:
            continue
        include = (
            p["land"] if mode == "land"
            else p["marine"] if mode == "marine"
            else (p["land"] or p["marine"])
        )
        if include:
            selected.append(float(v))
    if not selected:
        return None
    arr = np.asarray(selected)
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "min": round(float(np.min(arr)), 1),
        "max": round(float(np.max(arr)), 1),
        "p10": round(float(np.percentile(arr, 10)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
    }


def layer_payload(grib, sample_points, source=None):
    values = sample_grib(grib, sample_points)
    meta = grib[3]
    return {
        "values": values,
        "stats": {
            mode: stats(values, sample_points, mode)
            for mode in ("land", "marine", "both")
        },
        "meta": {
            "validUtc": (
                meta_valid(meta).isoformat().replace("+00:00", "Z")
                if meta_valid(meta)
                else None
            ),
            "stepType": str(meta.get("stepType") or ""),
            "stepRange": str(meta.get("stepRange") or ""),
            "source": source,
        },
    }


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    land, marine, outlines, raw_land, raw_marine = fetch_boundaries()
    sample_points, bounds = build_sample_points(raw_land, raw_marine)
    cycles = latest_cycle_occurrences(now)
    targets = valid_targets(now)

    result = {
        "schemaVersion": 1,
        "generatedUtc": now.isoformat().replace("+00:00", "Z"),
        "office": WFO,
        "units": "F",
        "sampleSpacingDegrees": SAMPLE_SPACING,
        "bounds": [round(x, 4) for x in bounds],
        "boundaries": {"land": land, "marine": marine, **outlines},
        "points": sample_points,
        "products": {"mint": {}, "maxt": {}},
        "cycles": [c.isoformat().replace("+00:00", "Z") for c in cycles],
        "errors": [],
        "sourceNotes": {
            "nbm": "NOAA Open Data NBM CONUS Core/QMD GRIB2",
            "ndfd": "Operational NWS NDFD CONUS GRIB2",
        },
    }

    ndfd = {}
    for product in ("mint", "maxt"):
        try:
            ndfd[product] = read_ndfd_product(product)
        except Exception as exc:
            result["errors"].append(f"NDFD {product}: {type(exc).__name__}: {exc}")
            ndfd[product] = ({}, None)

    for product in ("mint", "maxt"):
        ndfd_messages, ndfd_url = ndfd[product]
        for valid_dt in targets[product]:
            valid_key = valid_dt.isoformat().replace("+00:00", "Z")
            block = {
                "validUtc": valid_key,
                "ndfd": None,
                "nbm": {},
            }
            nmsg = ndfd_messages.get(valid_key)
            if nmsg is not None:
                block["ndfd"] = layer_payload(nmsg, sample_points, ndfd_url)

            for init_dt in cycles:
                cycle_key = init_dt.isoformat().replace("+00:00", "Z")
                try:
                    guidance = read_nbm_field(init_dt, valid_dt, product)
                    if guidance is None:
                        block["nbm"][cycle_key] = {"available": False}
                        continue
                    entry = {
                        "available": True,
                        "leadHours": guidance["leadHours"],
                        "det": None,
                        "percentiles": {},
                        "qmdCycleUtc": guidance.get("qmdCycleUtc"),
                        "qmdValidUtc": guidance.get("qmdValidUtc"),
                    }
                    if guidance["det"] is not None:
                        entry["det"] = layer_payload(
                            guidance["det"], sample_points, "NBM core"
                        )
                    for pct, msg in guidance["percentiles"].items():
                        entry["percentiles"][str(pct)] = layer_payload(
                            msg, sample_points, "NBM qmd"
                        )
                    block["nbm"][cycle_key] = entry
                except Exception as exc:
                    result["errors"].append(
                        f"NBM {product} {cycle_key} -> {valid_key}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    block["nbm"][cycle_key] = {
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            result["products"][product][valid_key] = block

    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(
        f"Wrote {OUT} with {len(sample_points)} sample points, "
        f"{len(cycles)} NBM cycles, and "
        f"{sum(len(v) for v in result['products'].values())} valid periods."
    )
    if result["errors"]:
        print(f"Completed with {len(result['errors'])} non-fatal data gaps.")


if __name__ == "__main__":
    main()
