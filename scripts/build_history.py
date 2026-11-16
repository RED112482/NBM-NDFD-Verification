#!/usr/bin/env python3
"""Build provenance-aware WFO MOB historical MinT/MaxT guidance snapshots.

A date is a *valid date*, not a model initialization. Each nominal lead L
is matched to the operational NBM 01/13Z cycle at L-1 exact hours
before the 12Z MinT/00Z MaxT valid-time endpoint. This preserves the
operational 00/12Z NDFD comparison clock without pretending the hour offsets
are exact. NDFD is the most recent archived WMO issuance no later than the
target 00/12Z reference time. URMA is optional; missing verifying analyses
are never replaced with invented data.

All grid layers and analyses are sampled to the *same* MOB sample nodes.
"""
from __future__ import annotations
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release
from build_guidance import (DATA, NBM_BUCKET, WFO, PERCENTILES, safe_float,
                            s3, idx_lines, nbm_key, fetch_idx_message, meta_valid,
                            object_exists, read_nbm_field, stats, layer_payload)

OUT = DATA / "history.json"
GEOMETRY = DATA / "guidance.json"
NDFD_BUCKET = "noaa-ndfd-pds"
URMA_BUCKET = "noaa-urma-pds"
LEADS = (12, 24, 36, 48, 60)
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", "3"))
MAX_ARCHIVE_OBJECTS = 8
_ndfd_listing = {}
_grid_index_cache = {}


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def grid_values(grib, points):
    vals, lats, lons, meta = grib
    yy = np.array([p["lat"] for p in points], dtype=float)
    xx = np.array([p["lon"] for p in points], dtype=float)
    mask = (np.isfinite(vals) & (lats >= yy.min()-.3) & (lats <= yy.max()+.3)
            & (lons >= xx.min()-.3) & (lons <= xx.max()+.3))
    if not mask.any():
        return [None] * len(points)
    local_lat = lats[mask]
    local_lon = lons[mask]
    # At MOB latitude, longitude distances must be latitude corrected.
    scl = math.cos(math.radians(float(np.mean(yy))))
    coords = np.column_stack((local_lat, local_lon*scl))
    query = np.column_stack((yy, xx*scl))
    index = cKDTree(coords).query(query, k=1)[1]
    chosen = np.asarray(vals[mask][index], dtype=float)
    return [round(float(v), 1) if np.isfinite(v) else None for v in chosen]


def payload(grib, points, source, issued=None):
    values = grid_values(grib, points)
    meta = grib[3]
    valid = meta_valid(meta)
    return {
        "values": values,
        "stats": {mode: stats(values, points, mode) for mode in ("land","marine","both")},
        "meta": {
            "source": source,
            "issuedUtc": iso(issued) if issued else None,
            "validUtc": iso(valid) if valid else None,
            "stepType": str(meta.get("stepType") or ""),
            "stepRange": str(meta.get("stepRange") or ""),
            "name": str(meta.get("name") or ""),
            "percentile": safe_float(meta.get("percentileValue")),
        },
    }


def match_errors(forecast, analysis, points):
    if not forecast or not analysis:
        return None
    f = np.asarray([np.nan if v is None else v for v in forecast["values"]], dtype=float)
    a = np.asarray([np.nan if v is None else v for v in analysis["values"]], dtype=float)
    delta = np.where(np.isfinite(f) & np.isfinite(a), f-a, np.nan)
    output = {"values": [round(float(v),1) if np.isfinite(v) else None for v in delta]}
    by_mode = {}
    for mode in ("land", "marine", "both"):
        keep = np.array([p["land"] if mode=="land" else p["marine"] if mode=="marine"
                         else p["land"] or p["marine"] for p in points])
        local = delta[keep & np.isfinite(delta)]
        by_mode[mode] = ({"mae": round(float(np.mean(np.abs(local))), 2),
                          "bias": round(float(np.mean(local)), 2),
                          "n": int(len(local))} if len(local) else None)
    output["stats"] = by_mode
    return output


def list_ndfd(day, product):
    label = (day.isoformat(), product)
    if label in _ndfd_listing:
        return _ndfd_listing[label]
    prefix = f"wmo/{product}/{day:%Y/%m/%d}/"
    rows = []
    token = None
    while True:
        kw = {"Bucket":NDFD_BUCKET, "Prefix":prefix, "MaxKeys":1000}
        if token: kw["ContinuationToken"] = token
        result = s3.list_objects_v2(**kw)
        for obj in result.get("Contents", []):
            key = obj.get("Key","")
            name = key.split("/")[-1]
            # CONUS NWS WMO products; exclude indexes and unrelated sectors.
            if "KWBN" not in name or not name.endswith(".bin") and ".bin" in name:
                continue
            stamp = re.search(r"(20\d{10})", name)
            if not stamp: continue
            try:
                dt = datetime.strptime(stamp.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                rows.append((dt, key))
            except ValueError:
                continue
        if not result.get("IsTruncated"): break
        token=result["NextContinuationToken"]
    _ndfd_listing[label] = sorted(rows)
    return _ndfd_listing[label]


def ndfd_keys_before(reference, product):
    rows = []
    for day_delta in (0, -1):
        rows += list_ndfd((reference+timedelta(days=day_delta)).date(), product)
    prior = [r for r in rows if r[0] <= reference]
    if not prior:
        return []
    latest = max(r[0] for r in prior)
    group = [key for issued,key in prior if issued == latest]
    return [(latest,key) for key in group[:MAX_ARCHIVE_OBJECTS]]


def decode_grib_file(raw, wanted_dt, namehint):
    """Use ecCodes message framing, never split on a potentially interior 7777."""
    fd,path=tempfile.mkstemp(suffix=".grib2")
    os.close(fd)
    Path(path).write_bytes(raw)
    grids=[]
    try:
        with open(path,"rb") as fh:
            while True:
                gid=codes_grib_new_from_file(fh)
                if gid is None:break
                try:
                    meta={}
                    for key in ("shortName","name","units","stepType","stepRange",
                                "percentileValue","validityDate","validityTime",
                                "dataDate","dataTime","forecastTime"):
                        try: meta[key]=codes_get(gid,key)
                        except Exception: pass
                    if meta_valid(meta) != wanted_dt: continue
                    short=str(meta.get("shortName") or "").lower()
                    name=str(meta.get("name") or "").lower()
                    if not (namehint in short or
                            ("minimum" in name if namehint=="tmin" else "maximum" in name)):
                        continue
                    vals=np.asarray(codes_get_array(gid,"values"),dtype=float)
                    lats=np.asarray(codes_get_array(gid,"latitudes"),dtype=float)
                    lons=np.asarray(codes_get_array(gid,"longitudes"),dtype=float)
                    units=str(meta.get("units") or "")
                    if units=="K" or "kelvin" in units.lower():
                        vals=(vals-273.15)*9/5+32
                    vals=np.where(np.isfinite(vals)&(vals>-120)&(vals<160),vals,np.nan)
                    lons=np.where(lons>180,lons-360,lons)
                    grids.append((vals,lats,lons,meta))
                finally:
                    codes_release(gid)
    finally:
        Path(path).unlink(missing_ok=True)
    return grids


def ndfd_grib_before(reference, valid, product, points):
    # Never use an issuance newer than the requested reference time.
    for issued,key in ndfd_keys_before(reference,product):
        try:
            raw=s3.get_object(Bucket=NDFD_BUCKET,Key=key)["Body"].read()
            matches=decode_grib_file(raw,valid,"tmin" if product=="mint" else "tmax")
            if not matches: continue
            # Multiple sectors may share validity; choose the one with MOB coverage.
            choices=[]
            for grib in matches:
                content=payload(grib,points,f"NDFD WMO archive: {key}",issued)
                n=content["stats"]["land"]["n"] if content["stats"]["land"] else 0
                choices.append((n,content))
            if choices:
                best=max(choices,key=lambda pair:pair[0])
                if best[0]>0:return best[1]
        except Exception as exc:
            print("NDFD warning",product,valid,repr(exc),flush=True)
    return None


def read_urma(valid, product):
    """A conservative URMA lookup: require exact field, validity and minimum/maximum step."""
    ds=valid.strftime("%Y%m%d")
    key=f"urma2p5.{ds}/urma2p5.t20z.2dvaranl_ndfd.grb2_wexp"
    if not object_exists(URMA_BUCKET,key):return None
    lines=idx_lines(URMA_BUCKET,key)
    token=":TMIN:" if product=="mint" else ":TMAX:"
    candidates=[i for i,line in enumerate(lines) if token in line.upper()]
    for i in candidates:
        try:
            g=fetch_idx_message(URMA_BUCKET,key,lines,i)
            if meta_valid(g[3])!=valid:continue
            if str(g[3].get("stepType","")).lower() not in ("min","max","minimum","maximum"):
                continue
            return g
        except Exception as exc:
            print("URMA candidate warning",repr(exc),flush=True)
    return None


def run():
    base=json.loads(GEOMETRY.read_text())
    points=base["points"]
    now=datetime.now(timezone.utc)
    # Most recently completed 12Z minimum / 00Z maximum, respectively.
    latest_mint=now.date() if now.hour>=15 else now.date()-timedelta(days=1)
    latest_maxt=now.date() if now.hour>=3 else now.date()-timedelta(days=1)
    history={
      "schemaVersion":1,"generatedUtc":iso(now),"office":WFO,
      "points":points,"bounds":base["bounds"],"boundaries":{},
      "leads":list(LEADS),"days":{"mint":{},"maxt":{}},
      "errors":[],"method":{
          "nbm":"01Z/13Z operational comparison: nominal lead 12h = actual 11h; 24h = actual 23h.",
          "ndfd":"Historical NWS WMO archive, last issuance at/before reference time.",
          "verification":"URMA gridded TMIN/TMAX used only when exact 00Z/12Z valid metadata found.",
          "percentiles":"QMD 00/06/12/18Z source cycles must be available before reference. 18h QMD is not verified using 12h URMA."
       }
    }
    for product,latest in (("mint",latest_mint),("maxt",latest_maxt)):
        for offset in range(HISTORY_DAYS):
            day=latest-timedelta(days=offset)
            valid=datetime(day.year,day.month,day.day,
                           12 if product=="mint" else 0,tzinfo=timezone.utc)
            key=iso(valid)
            print("HISTORY",product,key,flush=True)
            block={"validUtc":key,"analysis":None,"leads":{}}
            try:
                truth=read_urma(valid,product)
                if truth is not None:
                    block["analysis"]=payload(truth,points,"NOAA URMA")
            except Exception as exc:
                history["errors"].append(f"{product} {key} URMA: {type(exc).__name__} {exc}")
            for lead in LEADS:
                # 11/23/35/47/59 actual elapsed hours. Preserve true hours.
                init=valid-timedelta(hours=lead-1)
                reference=valid-timedelta(hours=lead)
                slot={"nominalLead":lead,"actualLeadHours":lead-1,
                      "nbmCycleUtc":iso(init),"ndfdReferenceUtc":iso(reference),
                      "nbm":{"det":None,"percentiles":{}},
                      "ndfd":None,"errors":{}}
                try:
                    raw=read_nbm_field(init,valid,product)
                    if raw is None:
                        raw={"det":None,"percentiles":{}}
                    # Some 01/13Z core 12-hour extrema are supplied at F012
                    # (one hour after the 00/12Z display convention), not F011.
                    # Show the actual 1-hour-offset guidance, but do not call
                    # its errors exact verification against the 00/12Z analysis.
                    if lead==12 and raw.get("det") is None:
                        shifted=read_nbm_field(init,valid+timedelta(hours=1),product)
                        if shifted and shifted.get("det") is not None:
                            raw["det"]=shifted["det"]
                            slot["nbm"]["validOffsetHours"]=1
                    if raw:
                        if raw.get("det") is not None:
                            slot["nbm"]["det"]=payload(raw["det"],points,"NBM core",init)
                        slot["nbm"]["qmdCycleUtc"]=raw.get("qmdCycleUtc")
                        for pct,g in raw.get("percentiles",{}).items():
                            slot["nbm"]["percentiles"][str(pct)]=payload(
                                g,points,"NBM QMD",
                            datetime.fromisoformat(raw["qmdCycleUtc"].replace("Z","+00:00")))
                except Exception as exc:
                    slot["errors"]["nbm"]=str(exc)
                    print("NBM error",str(exc),flush=True)
                try:
                    slot["ndfd"]=ndfd_grib_before(reference,valid,product,points)
                except Exception as exc:
                    slot["errors"]["ndfd"]=str(exc)
                    print("NDFD error",str(exc),flush=True)
                if block["analysis"]:
                    if slot["nbm"]["det"]:
                        if slot["nbm"].get("validOffsetHours"):
                            slot["nbm"]["det"]["error"]=None
                            slot["nbm"]["det"]["meta"]["verificationNote"]=(
                                "Actual NBM validity is 1 hour after NDFD 00/12Z endpoint; not exactly verifiable")
                        else:
                            slot["nbm"]["det"]["error"]=match_errors(
                                slot["nbm"]["det"],block["analysis"],points)
                    for p in slot["nbm"]["percentiles"].values():
                        # 18-hour QMD extrema must not be scored against the
                        # 12-hour URMA analysis merely because day labels match.
                        p["error"]=None
                        p["meta"]["verificationNote"]="18h QMD/12h URMA mismatch; map and raw statistics only"
                    if slot["ndfd"]:
                        slot["ndfd"]["error"]=match_errors(
                            slot["ndfd"],block["analysis"],points)
                block["leads"][str(lead)]=slot
                # Checkpoint each lead so failed runs can be inspected.
                history["days"][product][key]=block
                OUT.write_text(json.dumps(history,separators=(",",":")))
            print("HISTORY completed",product,key,flush=True)
    print("History output",OUT,OUT.stat().st_size,"bytes",flush=True)


if __name__=="__main__":
    run()
