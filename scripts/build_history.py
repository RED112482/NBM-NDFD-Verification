#!/usr/bin/env python3
"""Build provenance-aware WFO MOB historical MinT/MaxT guidance snapshots.

A date is a *valid date*, not a model initialization. Each nominal lead L
is matched to the operational NBM 01/13Z cycle 11 hours BEFORE
the archived NDFD 00/12Z reference snapshot (not after it).
Thus a nominal NDFD -12h lead uses NBM initialized -23h from validity,
and a -24h NDFD lead uses NBM initialized -35h. Exact cycle/lead stamps
are preserved and any 1h grid valid-time offset is explicitly flagged. NDFD is the most recent archived WMO issuance no later than the
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release
from build_guidance import (DATA, NBM_BUCKET, WFO, PERCENTILES, safe_float,
                            s3, idx_lines, nbm_key, fetch_idx_message, meta_valid,
                            object_exists, read_nbm_field, stats, layer_payload,
                            extrema_metadata_valid, grib_extrema_window)

OUT = DATA / "history.json"
GEOMETRY = DATA / "guidance.json"
NDFD_BUCKET = "noaa-ndfd-pds"
URMA_BUCKET = "noaa-urma-pds"
RTMA_BUCKET = "noaa-rtma-pds"
LEADS = (12, 24, 36, 48, 60)
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", "3"))
ANALYSIS_HOURS = (12, 18)  # validate each forecast against its native window; all hourly samples required
MAX_ARCHIVE_OBJECTS = 24
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
    distances, indices = cKDTree(coords).query(query, k=4)
    weights = 1.0 / np.maximum(distances, 1e-9) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    chosen = np.sum(np.asarray(vals[mask][indices], dtype=float) * weights, axis=1)
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
            "periodHours": (safe_float(meta.get("endStep"))-safe_float(meta.get("startStep")))
                if safe_float(meta.get("endStep")) is not None and safe_float(meta.get("startStep")) is not None else None,
            "shortName": str(meta.get("shortName") or ""),
            "startStep": safe_float(meta.get("startStep")),
            "endStep": safe_float(meta.get("endStep")),
            "typeOfLevel": str(meta.get("typeOfLevel") or ""),
            "level": safe_float(meta.get("level")),
            "gribReferenceDate": str(meta.get("dataDate") or ""),
            "gribReferenceTime": str(meta.get("dataTime") or ""),
            "actualWindowStartUtc": iso(grib_extrema_window(meta)[0]) if grib_extrema_window(meta) else None,
            "actualWindowEndUtc": iso(grib_extrema_window(meta)[1]) if grib_extrema_window(meta) else None,
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
    # The newest WMO issuance can omit an older valid period. Try prior
    # issuances, newest first; never borrow one issued after reference.
    ordered = sorted(prior, reverse=True)
    return ordered[:MAX_ARCHIVE_OBJECTS]


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
                                "dataDate","dataTime","forecastTime","startStep","endStep",
                                "typeOfLevel","level","productDefinitionTemplateNumber"):
                        try: meta[key]=codes_get(gid,key)
                        except Exception: pass
                    if meta_valid(meta) != wanted_dt: continue
                    short=str(meta.get("shortName") or "").lower()
                    name=str(meta.get("name") or "").lower()
                    if not (namehint in short or
                            ("minimum" in name if namehint=="tmin" else "maximum" in name)):
                        continue
                    product = "mint" if namehint == "tmin" else "maxt"
                    valid, reason = extrema_metadata_valid(meta, product, wanted_dt)
                    exact_window=(wanted_dt-timedelta(hours=12),wanted_dt)
                    window=grib_extrema_window(meta)
                    if valid and window!=exact_window:
                        valid=False
                        reason="actual GRIB extrema window "+str(window)+" is not "+str(exact_window)
                    if not valid:
                        print("Rejected NDFD archived field:", reason, flush=True)
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
    """Require a genuinely covering MOB sector; a small neighboring sector
    must never be treated as the full official NDFD forecast grid."""
    land_count=sum(bool(p["land"]) for p in points)
    min_coverage=max(1,math.ceil(.80*land_count))
    for issued,key in ndfd_keys_before(reference,product):
        try:
            raw=s3.get_object(Bucket=NDFD_BUCKET,Key=key)["Body"].read()
            matches=decode_grib_file(raw,valid,"tmin" if product=="mint" else "tmax")
            for grib in matches:
                content=payload(grib,points,f"NDFD WMO archive: {key}",issued)
                local=content["stats"].get("land")
                n=local["n"] if local else 0
                if n < min_coverage:
                    continue
                content["meta"]["landCoverageFraction"]=round(n/land_count,3)
                return content
        except Exception as exc:
            print("NDFD warning",product,valid,repr(exc),flush=True)
    print("NDFD rejected: no 12h field covering 80% MOB land",
          product,iso(valid),iso(reference),flush=True)
    return None


def read_analysis_hour(source, stamp, points):
    """Read ONLY the true 2 m hourly TMP analysis from NOAA public GRIB2.
    The index byte range avoids downloading ~80 MB hourly analysis bundles.
    """
    bucket = URMA_BUCKET if source == "urma" else RTMA_BUCKET
    ds = stamp.strftime("%Y%m%d")
    prefix = "urma2p5" if source == "urma" else "rtma2p5"
    key = f"{prefix}.{ds}/{prefix}.t{stamp:%H}z.2dvaranl_ndfd.grb2_wexp"
    if not object_exists(bucket, key + ".idx"):
        return stamp, None
    try:
        lines = idx_lines(bucket, key)
        candidates = [i for i, row in enumerate(lines)
                      if ":TMP:" in row.upper() and
                      "2 M ABOVE GROUND" in row.upper() and
                      (":ANL:" in row.upper() or ":0 HOUR FCST:" in row.upper())]
        for i in candidates:
            grib = fetch_idx_message(bucket, key, lines, i)
            if meta_valid(grib[3]) != stamp:
                continue
            return stamp, grid_values(grib, points)
    except Exception as exc:
        print(f"{source.upper()} hourly {iso(stamp)}: {type(exc).__name__} {exc}", flush=True)
    return stamp, None


def analysis_extrema(source, valid, product, points, hourly_cache, hours=12):
    """Derive extrema from complete actual hourly 2-m analyses.

    NDFD/core NBM: [valid - 12h, valid] inclusive (13 samples).
    CONUS QMD MinT/MaxT: [valid - 12h, valid + 6h] inclusive
    (19 samples). NBM labels these 18-hour extrema at the
    conventional 12Z MinT / 00Z MaxT slot, but GRIB ends 6h later.
    Never compare mismatched 12h and 18h extrema.
    """
    if hours not in (12,18):
        raise ValueError("Unsupported extrema window")
    start = valid - timedelta(hours=12)
    end = valid + timedelta(hours=6 if hours == 18 else 0)
    times = [start + timedelta(hours=n) for n in range(hours + 1)]
    rows = [hourly_cache.get((source,t)) for t in times]
    if any(row is None for row in rows):
        return None
    arr = np.asarray([[np.nan if v is None else v for v in row]
                      for row in rows], dtype=float)
    good = np.all(np.isfinite(arr), axis=0)
    extreme = np.min(arr, axis=0) if product == "mint" else np.max(arr, axis=0)
    final = [round(float(v),1) if keep else None
             for v,keep in zip(extreme,good)]
    if not any(v is not None for v in final):
        return None
    return {
       "values":final,
       "stats":{scope:stats(final,points,scope)
                for scope in ("land","marine","both")},
       "meta":{
          "source":f"NOAA {source.upper()} hourly 2-m temperature analyses",
          "validUtc":iso(end),"startUtc":iso(start),"endUtc":iso(end),
          "periodHours":hours,"stepRange":f"{hours}h",
          "analysisMethod":"hourly_extrema",
          "hourlySamples":len(times),
          "verificationNote":f"DERIVED {hours}-hour extreme from {len(times)} actual hourly 2-m analyses; not native URMA TMIN/TMAX",
       },
    }


def load_hourly_cache(targets, points):
    stamps = sorted({valid + timedelta(hours=h) for valid in targets
                     for h in range(-12,7)})
    cache={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(read_analysis_hour,src,stamp,points):(src,stamp)
                 for src in ("urma","rtma") for stamp in stamps}
        for f in as_completed(futures):
            src,stamp=futures[f]
            try:
                _,vals=f.result()
                cache[(src,stamp)]=vals
            except Exception as exc:
                print(f"{src} {stamp} fetch failed: {exc}",flush=True)
                cache[(src,stamp)]=None
    for src in ("urma","rtma"):
        n=sum(cache.get((src,stamp)) is not None for stamp in stamps)
        print(src.upper(),"hourly analyses",n,"/",len(stamps),flush=True)
    return cache


def run():
    base=json.loads(GEOMETRY.read_text())
    points=base["points"]
    now=datetime.now(timezone.utc)
    # Most recently completed 12Z minimum / 00Z maximum, respectively.
    latest_mint=now.date() if now.hour>=15 else now.date()-timedelta(days=1)
    latest_maxt=now.date() if now.hour>=3 else now.date()-timedelta(days=1)
    history={
      "schemaVersion":1,"generatedUtc":iso(now),"office":WFO,
      "points":points,"bounds":base["bounds"],"sampleSpacingDegrees":base.get("sampleSpacingDegrees",.075),"boundaries":{},
      "leads":list(LEADS),"days":{"mint":{},"maxt":{}},
      "errors":[],"audit":[],"method":{
          "nbm":"Operational pairing: 00Z NDFD reference -> preceding 13Z NBM, 12Z NDFD reference -> preceding 01Z NBM. Actual NBM forecast age = NDFD lead + 11h.",
          "ndfd":"Historical NWS WMO archive, last issuance at/before reference time.",
          "verification":"NDFD/NBM core: 12h extrema ending at 12Z/00Z. QMD percentiles: 18h extrema ending at 18Z/06Z; scored against separate 18h URMA/RTMA derived truth. All analysis requires complete hourly 2m coverage.",
          "percentiles":"QMD 00/06/12/18Z source cycles must be available before reference. 18h QMD is not verified using 12h URMA."
       }
    }
    latest_dates = {"mint": latest_mint, "maxt": latest_maxt}
    targets = [datetime((latest_dates[p]-timedelta(days=d)).year,
                        (latest_dates[p]-timedelta(days=d)).month,
                        (latest_dates[p]-timedelta(days=d)).day,
                        12 if p=="mint" else 0,tzinfo=timezone.utc)
               for p in ("mint","maxt") for d in range(HISTORY_DAYS)]
    hourly_cache = load_hourly_cache(targets, points)
    for product,latest in (("mint",latest_mint),("maxt",latest_maxt)):
        for offset in range(HISTORY_DAYS):
            day=latest-timedelta(days=offset)
            valid=datetime(day.year,day.month,day.day,
                           12 if product=="mint" else 0,tzinfo=timezone.utc)
            key=iso(valid)
            print("HISTORY",product,key,flush=True)
            block={"validUtc":key,"analysis":None,"rtma":None,"analysis18":None,"rtma18":None,"leads":{}}
            for source, field12, field18 in (
                ("urma","analysis","analysis18"),
                ("rtma","rtma","rtma18")
            ):
                for field,hours in ((field12,12),(field18,18)):
                    try:
                        block[field]=analysis_extrema(
                            source,valid,product,points,hourly_cache,hours=hours)
                    except Exception as exc:
                        history["errors"].append(
                            f"{product} {key} {source.upper()} {hours}h: {type(exc).__name__}: {exc}")
            for lead in LEADS:
                # NDFD release/reference at -12/-24/-36/-48/-60h.
                # Pair to last operational 13Z/01Z NBM run before that
                # release, never a later run (no look-ahead contamination).
                reference=valid-timedelta(hours=lead)
                init=reference-timedelta(hours=11)
                slot={"nominalLead":lead,"actualLeadHours":lead+11,
                      "nbmCycleUtc":iso(init),"ndfdReferenceUtc":iso(reference),
                      "expectedWindowStartUtc":iso(valid-timedelta(hours=12)),
                      "expectedWindowEndUtc":iso(valid),
                      "nbm":{"det":None,"percentiles":{}},
                      "ndfd":None,"errors":{}}
                try:
                    raw=read_nbm_field(init,valid,product)
                    if raw is None:
                        raw={"det":None,"percentiles":{}}
                    # Do not substitute the +1h core field: the user is
                    # verifying actual published 00/12Z period extrema.
                    # An unavailable exactly matched source remains missing.
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
                # Use URMA primarily; RTMA is labeled fallback for verification.
                truth=block["analysis"] or block["rtma"]
                truth_source="URMA" if block["analysis"] else "RTMA" if block["rtma"] else None
                if truth:
                    if slot["nbm"]["det"]:
                        if slot["nbm"].get("validOffsetHours"):
                            slot["nbm"]["det"]["error"]=None
                            slot["nbm"]["det"]["meta"]["verificationNote"]=(
                                "Actual NBM validity is 1 hour after NDFD 00/12Z endpoint; not exactly verifiable")
                        else:
                            slot["nbm"]["det"]["error"]=match_errors(
                                slot["nbm"]["det"],truth,points)
                    for p in slot["nbm"]["percentiles"].values():
                        p["error"]=None
                        p["meta"]["verificationNote"]="NBM QMD native 18h extrema: use URMA/RTMA 18h verification, NOT the 12h analysis"
                        p["meta"]["periodHours"]=18
                        p["meta"]["windowStartUtc"]=iso(valid-timedelta(hours=12))
                        p["meta"]["windowEndUtc"]=iso(valid+timedelta(hours=6))
                        # This is a separate valid forecast-verification family.
                        truth18=block["analysis18"] or block["rtma18"]
                        if truth18:
                            p["error"]=match_errors(p,truth18,points)
                    if slot["ndfd"]:
                        slot["ndfd"]["error"]=match_errors(
                            slot["ndfd"],truth,points)
                slot["verificationSource"] = truth_source
                # Collect independently inspectable raw point samples.
                # These are original unsmoothed GRIB samples, NOT bias maps.
                cities=(("Mobile",30.6954,-88.0399),
                        ("Pensacola",30.4213,-87.2169),
                        ("Evergreen",31.4335,-86.9569),
                        ("Waynesboro",31.6749,-88.6462),
                        ("Bay Minette",30.8829,-87.7731),
                        ("Gulf Shores",30.2460,-87.7008))
                lat_scale=math.cos(math.radians(30.5))
                audit_points={}
                for name,lat,lon in cities:
                    ix=min(range(len(points)),
                           key=lambda j:(points[j]["lat"]-lat)**2+
                           ((points[j]["lon"]-lon)*lat_scale)**2)
                    p=points[ix]
                    pair={"lat":p["lat"],"lon":p["lon"]}
                    for field_name,layer in (
                        ("ndfd",slot["ndfd"]),("det",slot["nbm"]["det"]),
                        ("urma12",block["analysis"]),("rtma12",block["rtma"]),
                        ("urma18",block["analysis18"]),("rtma18",block["rtma18"])
                    ):
                        pair[field_name]=(layer["values"][ix] if layer else None)
                    audit_points[name]=pair
                slot["diagnostics"]={"sampledPoints":audit_points,
                  "status":"Raw NOAA field samples; no display smoothing",
                  "comparability":"NDFD/NBM 12h; QMD 18h evaluated separately"}
                # Persist a five-day scientific audit, including outlier counts.
                # NEVER clip or suppress extreme errors: expose their provenance.
                record = {"product":product,"validUtc":key,"nominalLead":lead,
                    "nbmCycleUtc":slot["nbmCycleUtc"],"referenceUtc":slot["ndfdReferenceUtc"],
                    "analysisSource":truth_source}
                analysis_fields = {"URMA":block["analysis"],"RTMA":block["rtma"]}
                for src_name, field in (("NDFD",slot["ndfd"]),
                                        ("NBM deterministic",slot["nbm"]["det"])):
                    label="ndfd" if src_name=="NDFD" else "nbmDet"
                    if field:
                        record[label]={"meta":field["meta"],
                                      "stats":field["stats"].get("land")}
                        if slot["nbm"].get("validOffsetHours") and src_name!="NDFD":
                            record[label]["status"]="VALID TIME OFFSET: NOT EXACTLY VERIFIED"
                        for ana_name, ana_field in analysis_fields.items():
                            if not ana_field:continue
                            forecast=np.asarray([np.nan if x is None else x for x in field["values"]],dtype=float)
                            observed=np.asarray([np.nan if x is None else x for x in ana_field["values"]],dtype=float)
                            land=np.asarray([p["land"] for p in points],dtype=bool)
                            mask=np.isfinite(forecast)&np.isfinite(observed)&land
                            diff=forecast[mask]-observed[mask]
                            if diff.size:
                                record[label][ana_name]={"n":int(diff.size),
                                    "mae":round(float(np.mean(abs(diff))),2),
                                    "bias":round(float(np.mean(diff)),2),
                                    "absErrorOver20F":int(np.sum(np.abs(diff)>20)),
                                    "maxAbsoluteErrorF":round(float(np.max(np.abs(diff))),1)}
                    else:
                        record[label]={"status":"MISSING OR REJECTED: check GRIB interval/validity"}
                history["audit"].append(record)
                print("FIELD AUDIT",json.dumps(record,separators=(",",":")),flush=True)
                block["leads"][str(lead)]=slot
                # Checkpoint each lead so failed runs can be inspected.
                history["days"][product][key]=block
                OUT.write_text(json.dumps(history,separators=(",",":")))
            print("HISTORY completed",product,key,flush=True)
    print("History output",OUT,OUT.stat().st_size,"bytes",flush=True)


if __name__=="__main__":
    run()
