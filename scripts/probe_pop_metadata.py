#!/usr/bin/env python3
"""Single-file read-only audit: actual NOAA NDFD PoP12 GRIB metadata."""
import sys
sys.path.insert(0,"scripts")
from datetime import datetime,timezone,timedelta
from build_pop_history import ndfd_keys, ndfd_decoded, valid_period
from build_guidance import meta_valid, grib_extrema_window
from build_history import iso, grid_values
from build_guidance import DATA
import json
import numpy as np
ref=datetime(2026,11,24,0,tzinfo=timezone.utc)
valid=ref+timedelta(hours=12)
points=json.loads((DATA/"guidance.json").read_text())["points"]
land_total=sum(bool(p["land"]) for p in points)
for fam in ("Z98","Z97"):
 for d in (ref.date()-timedelta(days=1),):
  keys=ndfd_keys(d,fam)
  print("NDFD PROBE KEYS",d,fam,len(keys),[(iso(ts),key) for ts,key in keys[:4]],flush=True)
  recent=[(issued,key) for issued,key in keys if issued<=ref and ref-issued<=timedelta(hours=7)]
  for issued,key in recent[:2]:
   print("NDFD PROBE",key,"issued",iso(issued),flush=True)
   for i,(vals,lats,lons,meta) in enumerate(ndfd_decoded(key)):
    subset={k:meta.get(k) for k in ("shortName","name","units","stepType","stepRange","dataDate","dataTime","startStep","endStep","validityDate","validityTime","typeOfLevel","productDefinitionTemplateNumber")}
    finite=np.asarray(vals)[np.isfinite(vals)]
    if meta_valid(meta)==valid:
     land=grid_values((vals,lats,lons,meta),points,max_nearest_distance=.065)
     count=sum(v is not None and p["land"] for v,p in zip(land,points))
     print("NDFD MOB COVERAGE",key,"land",count,"/",land_total,
       "grid_bbox",(round(float(np.nanmin(lats)),3),round(float(np.nanmin(lons)),3),
                    round(float(np.nanmax(lats)),3),round(float(np.nanmax(lons)),3)),
       "sample_vals",land[:18],flush=True)
    print("MSG",i,subset,"META_VALID",meta_valid(meta),"WINDOW",grib_extrema_window(meta),
      "WANTED",valid,"MATCH",valid_period(meta,valid,12),
      "VALUES",((float(finite.min()),float(finite.max())) if finite.size else None),flush=True)

# Latest Monday MaxT percentile inventory. TMP, not TMAX, is the correct field!
from build_guidance import s3,NBM_BUCKET,idx_lines
for cycle,lead in (("12",42),("06",48),("00",54)):
 key=f"blend.20261122/{cycle}/qmd/blend.t{cycle}z.qmd.f{lead:03d}.co.grib2"
 try:
  rows=idx_lines(NBM_BUCKET,key)
  matches=[row for row in rows if ":TMP:" in row.upper() and
           " HOUR MAX FCST:" in row.upper() and
           any(f"{p}% level" in row for p in (10,25,50,75,90))]
  print("LATEST_MAXT_QMD",key,"matches",len(matches),matches[:8],flush=True)
 except Exception as exc:print("LATEST_MAXT_QMD",key,"error",repr(exc),flush=True)

# Run the exact existing production QMD selector for Monday Nov23 MaxT.
from build_guidance import read_nbm_field
init=datetime(2026,11,23,1,tzinfo=timezone.utc)
v=datetime(2026,11,24,0,tzinfo=timezone.utc)
try:
 layer=read_nbm_field(init,v,"maxt")
 print("ACTUAL_MAXT_SELECTOR",{"valid":iso(v),"init":iso(init),
  "qmdCycle":layer.get("qmdCycleUtc") if layer else None,
  "percentiles":sorted(layer["percentiles"]) if layer else [],
  "hasDet":bool(layer.get("det")) if layer else False},flush=True)
except Exception as exc:print("ACTUAL_MAXT_SELECTOR ERROR",repr(exc),flush=True)
