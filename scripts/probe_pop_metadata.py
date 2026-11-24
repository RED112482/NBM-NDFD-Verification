#!/usr/bin/env python3
"""Single-file read-only audit: actual NOAA NDFD PoP12 GRIB metadata."""
import sys
sys.path.insert(0,"scripts")
from datetime import datetime,timezone,timedelta
from build_pop_history import ndfd_keys, ndfd_decoded, valid_period
from build_guidance import meta_valid, grib_extrema_window
from build_history import iso
import numpy as np
ref=datetime(2026,11,24,0,tzinfo=timezone.utc)
valid=ref+timedelta(hours=12)
for fam in ("Z98","Z97"):
 for d in (ref.date(),ref.date()-timedelta(days=1)):
  keys=ndfd_keys(d,fam)
  print("NDFD PROBE KEYS",d,fam,len(keys),[(iso(ts),key) for ts,key in keys[:4]],flush=True)
  recent=[(issued,key) for issued,key in keys if issued<=ref and ref-issued<=timedelta(hours=7)]
  for issued,key in recent[:2]:
   print("NDFD PROBE",key,"issued",iso(issued),flush=True)
   for i,(vals,lats,lons,meta) in enumerate(ndfd_decoded(key)):
    subset={k:meta.get(k) for k in ("shortName","name","units","stepType","stepRange","dataDate","dataTime","startStep","endStep","validityDate","validityTime","typeOfLevel","productDefinitionTemplateNumber")}
    finite=np.asarray(vals)[np.isfinite(vals)]
    print("MSG",i,subset,"META_VALID",meta_valid(meta),"WINDOW",grib_extrema_window(meta),
      "WANTED",valid,"MATCH",valid_period(meta,valid,12),
      "VALUES",((float(finite.min()),float(finite.max())) if finite.size else None),flush=True)
