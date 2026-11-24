#!/usr/bin/env python3
"""Historical PoP12 verification using original NDFD/NBM probabilities and URMA 6h QPE.

A 12h PoP is the probability of 0.01 inch (0.254 mm) or more.
Official NDFD and the NBM *core* probability >0.254 mm are verified
against the sum of two non-overlapping 6h URMA precipitation analyses.
There is NO imputation of unavailable GRIB fields, and forecast products
are never allowed to be issued after their historical reference cutoff.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import json, math, re
import numpy as np
from build_guidance import (DATA, NBM_BUCKET, PERCENTILES, s3, nbm_key, idx_lines,
                            byte_range, object_exists, meta_valid, grib_extrema_window,
                            fetch_idx_message, decode_message_bytes_grib2io,
                            iter_grib_messages, safe_float)
from build_history import (GEOMETRY, NDFD_BUCKET, URMA_BUCKET, grid_values,
                           stats, iso, HISTORY_DAYS, LEADS)

OUT = DATA / "pop_history.json"
PCP_MM_THRESHOLD = 0.254  # Measurable precipitation: >= 0.01 inch
REQUIRED_PERIOD_HOURS = 12

def valid_period(meta, valid, hours):
    window = grib_extrema_window(meta)
    return bool(window and
                abs((window[1]-window[0]).total_seconds()-hours*3600)<90 and
                abs((window[1]-valid).total_seconds())<90)

def sample(grib, points, source):
    values=grid_values(grib,points,max_nearest_distance=.065)
    return {
      "values":values,
      "stats":{area:stats(values,points,area) for area in ("land","marine","both")},
      "meta":{
        "source":source,"validUtc":iso(meta_valid(grib[3])) if meta_valid(grib[3]) else None,
        "actualWindowStartUtc":iso(grib_extrema_window(grib[3])[0]) if grib_extrema_window(grib[3]) else None,
        "actualWindowEndUtc":iso(grib_extrema_window(grib[3])[1]) if grib_extrema_window(grib[3]) else None,
        "periodHours":12,"decoder":"NOAA-MDL grib2io",
        "units":"percent",
      }
    }

@lru_cache(maxsize=50)
def ndfd_keys(day, family):
    date=day.strftime("%Y/%m/%d")
    prefix=f"wmo/pop12/{date}/YDU{family}_KWBN_"  # U=CONUS, A=Puerto Rico
    result=[]
    paginator=s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=NDFD_BUCKET,Prefix=prefix):
        for item in page.get("Contents",[]):
            key=item["Key"]
            stamp=key.rsplit("_",1)[-1]
            try:ts=datetime.strptime(stamp,"%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            except ValueError:continue
            result.append((ts,key))
    return tuple(sorted(result,reverse=True))

@lru_cache(maxsize=2)
def ndfd_decoded(key):
    raw=s3.get_object(Bucket=NDFD_BUCKET,Key=key)["Body"].read()
    result=[]
    for blob in iter_grib_messages(raw):
        try:
            msg=decode_message_bytes_grib2io(blob)
            result.append(msg)
        except Exception as exc:
            print("NDFD pop12 decode",key,type(exc).__name__,str(exc)[:150],flush=True)
    return result

def ndfd_pop(reference,valid,points):
    horizon=(valid-reference).total_seconds()/3600
    family="Z98" if horizon<=72 else "Z97"
    candidate=[]
    for offset in (0,-1):
        candidate.extend(ndfd_keys(reference.date()+timedelta(days=offset),family))
    for issued,key in sorted(candidate,reverse=True):
        if issued>reference or reference-issued>timedelta(hours=7):continue
        try:
            for grib in ndfd_decoded(key):
                meta=grib[3]
                if meta_valid(meta)!=valid or not valid_period(meta,valid,12):continue
                # PoP12 is expressed in percentage points, NOT QPF mm.
                vals=grib[0]
                validvalues=vals[np.isfinite(vals)]
                if validvalues.size and (validvalues.min() < -.01 or validvalues.max()>100.01):
                    continue
                layer=sample(grib,points,"Official NDFD PoP12 WMO: "+key)
                coverage=layer["stats"]["land"]
                count=sum(bool(p["land"]) for p in points)
                if not coverage or coverage["n"]<count*.8:continue
                layer["meta"].update({"issuedUtc":iso(issued),"family":family,
                    "coverageFraction":round(coverage["n"]/count,3)})
                return layer
        except Exception as exc:
            print("NDFD pop12 retrieval",key,type(exc).__name__,str(exc)[:140],flush=True)
    return None

def nbm_pop(init,valid,points):
    lead=int((valid-init).total_seconds()/3600)
    if lead<=0:return None
    key=nbm_key(init,lead,"core")
    if not object_exists(NBM_BUCKET,key+".idx"):return None
    lines=idx_lines(NBM_BUCKET,key)
    start=lead-12
    for i,row in enumerate(lines):
        line=row.upper()
        if ":APCP:" not in line or ":PROB >0.254:" not in line:continue
        if f":{start}-{lead} HOUR ACC FCST:" not in line:continue
        try:
            grib=fetch_idx_message(NBM_BUCKET,key,lines,i,use_grib2io=True)
            if meta_valid(grib[3])!=valid or not valid_period(grib[3],valid,12):continue
            vals=grib[0]
            ok=vals[np.isfinite(vals)]
            if not ok.size or ok.min()<-.01 or ok.max()>100.01:continue
            layer=sample(grib,points,"NBM core PoP12: "+key)
            layer["meta"].update({"issuedUtc":iso(init),"nbmCycleUtc":iso(init),
                "leadHours":lead,"index":row})
            return layer
        except Exception as exc:
            print("NBM pop12 decode",key,type(exc).__name__,str(exc)[:160],flush=True)
    return None

@lru_cache(maxsize=2)
def qpe_raw(valid):
    ds=valid.strftime("%Y%m%d")
    key=f"urma2p5.{ds}/urma2p5.{ds}{valid:%H}.pcp_06h.wexp.grb2"
    if not object_exists(URMA_BUCKET,key):
        print("MISSING 6h URMA QPE:",key,flush=True)
        return None
    try:
        body=s3.get_object(Bucket=URMA_BUCKET,Key=key)["Body"].read()
        for blob in iter_grib_messages(body):
            grib=decode_message_bytes_grib2io(blob)
            meta=grib[3]
            if meta_valid(meta)!=valid or not valid_period(meta,valid,6):
                continue
            # APCP is mass/area (kg m^-2) numerically equivalent to mm.
            units=str(meta.get("units") or "").strip().lower()
            if units not in ("kg m**-2","kg m-2","kg/m^2","kg m^-2","mm",
                             "kg m**-2 ","kg/(m^2)","kg/(m²)"):
                if "kg" not in units and "mm" not in units:
                    print("URMA QPE unsupported units",key,repr(units),flush=True)
                    continue
            return grib,key
    except Exception as exc:
        print("URMA QPE failure",key,type(exc).__name__,str(exc)[:200],flush=True)
    return None

def urma_pop_verification(valid,points):
    stamps=(valid-timedelta(hours=6),valid)
    parts=[]
    for ts in stamps:
        raw=qpe_raw(ts)
        if not raw:return None,{"missing6hEndUtc":iso(ts),"available":False}
        grib,key=raw
        values=grid_values(grib,points,max_nearest_distance=.065)
        parts.append((values,key))
    mm=[]
    events=[]
    for x,y in zip(parts[0][0],parts[1][0]):
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            mm.append(None);events.append(None)
        else:
            accum=max(0,float(x))+max(0,float(y))
            mm.append(round(accum/25.4,4))
            events.append(100 if accum>=PCP_MM_THRESHOLD else 0)
    layer={
      "values":events,"qpeInches":mm,
      "stats":{scope:stats(events,points,scope) for scope in ("land","marine","both")},
      "meta":{"source":"URMA 6h QPE + URMA 6h QPE","periodHours":12,
              "validUtc":iso(valid),"startUtc":iso(valid-timedelta(hours=12)),
              "endUtc":iso(valid),"units":"0 or 100 (measurable precipitation)",
              "thresholdInches":0.01,
              "qpeKeys":[parts[0][1],parts[1][1]],
              "analysisMethod":"sum 2 actual 6h URMA APCP fields; >=0.254mm = yes"}
    }
    return layer,{"available":True,"qpe6hKeys":[parts[0][1],parts[1][1]]}

def verify(forecast,analysis,points):
    if not forecast or not analysis:return None
    a=np.array([np.nan if v is None else v for v in forecast["values"]],dtype=float)
    b=np.array([np.nan if v is None else v for v in analysis["values"]],dtype=float)
    result={}
    for scope in ("land","marine","both"):
        mask=np.array([p["land"] if scope=="land" else p["marine"] if scope=="marine"
                       else p["land"] or p["marine"] for p in points],dtype=bool)
        good=mask&np.isfinite(a)&np.isfinite(b)
        if not np.any(good):result[scope]=None;continue
        prob=a[good]/100
        obs=b[good]/100
        binary=prob>=.5
        yes=obs>=.5
        h=int(np.sum(binary&yes));m=int(np.sum(~binary&yes))
        f=int(np.sum(binary&~yes));c=int(np.sum(~binary&~yes))
        mean_err=float(np.mean(prob-obs))
        result[scope]={
            "n":int(good.sum()),"mae":round(100*float(np.mean(np.abs(prob-obs))),3),
            "bias":round(100*mean_err,3),
            "brier":round(float(np.mean((prob-obs)**2)),5),
            "meanForecast":round(100*float(np.mean(prob)),2),
            "observedFrequency":round(100*float(np.mean(obs)),2),
            "hits":h,"misses":m,"falseAlarms":f,"correctNegatives":c,
            "pod":round(h/(h+m),4) if h+m else None,
            "far":round(f/(h+f),4) if h+f else None,
            "csi":round(h/(h+f+m),4) if h+f+m else None,
            "accuracy":round((h+c)/(h+m+f+c),4)
        }
    return result

def build():
    base=json.loads(GEOMETRY.read_text())
    points=base["points"];now=datetime.now(timezone.utc)
    last12=(now.date() if now.hour>=12 else now.date()-timedelta(days=1))
    last00=now.date()
    result={"schemaVersion":1,"generatedUtc":iso(now),"office":"MOB",
      "product":"pop","units":"percent","thresholdInches":0.01,
      "bounds":base["bounds"],"points":points,
      "sampleSpacingDegrees":base.get("sampleSpacingDegrees",.025),
      "days":{},"errors":[],"leads":list(LEADS),
      "method":{"forecast":"NDFD PoP12 WMO and NBM core APCP probability >0.254 mm",
                "analysis":"Two complete, adjacent 6h URMA precipitation analyses summed to 12h; at least 0.254mm means YES",
                "cutoffs":"12h steps at past 00Z/12Z NDFD reference and preceding 13Z/01Z NBM cycles",
                "verification":"Brier score and probability bias use fractional forecasts against binary outcomes; POD/FAR/CSI use configurable default 50% PoP decision threshold"}
    }
    targets=[]
    for hour,last in ((12,last12),(0,last00)):
        for i in range(HISTORY_DAYS):
            day=last-timedelta(days=i)
            targets.append(datetime(day.year,day.month,day.day,hour,tzinfo=timezone.utc))
    for valid in sorted(targets,reverse=True):
        key=iso(valid)
        truth,status=urma_pop_verification(valid,points)
        day={"validUtc":key,"analysis":truth,"analysisStatus":status,"leads":{}}
        result["days"][key]=day
        print("POP12",key,"URMA",status,flush=True)
        for lead in LEADS:
            reference=valid-timedelta(hours=lead)
            init=reference-timedelta(hours=11)
            slot={"nominalLead":lead,"ndfdReferenceUtc":iso(reference),
                  "nbmCycleUtc":iso(init),"ndfd":None,
                  "nbm":{"det":None,"percentiles":{}},"errors":{}}
            try:slot["ndfd"]=ndfd_pop(reference,valid,points)
            except Exception as exc:
                slot["errors"]["ndfd"]=str(exc);print("POP NDFD ERROR",key,lead,repr(exc),flush=True)
            try:slot["nbm"]["det"]=nbm_pop(init,valid,points)
            except Exception as exc:
                slot["errors"]["nbm"]=str(exc);print("POP NBM ERROR",key,lead,repr(exc),flush=True)
            for name,field in (("ndfd",slot["ndfd"]),("det",slot["nbm"]["det"])):
                if field:field["verification"]=verify(field,truth,points)
            day["leads"][str(lead)]=slot
            print("POP12 LEAD",key,lead,
                  "NDFD",bool(slot["ndfd"]),"NBM",bool(slot["nbm"]["det"]),
                  "TRUTH",bool(truth),flush=True)
        OUT.write_text(json.dumps(result,separators=(",",":")))
    print("POP12 COMPLETE",OUT,OUT.stat().st_size,flush=True)

if __name__=="__main__":
    build()
