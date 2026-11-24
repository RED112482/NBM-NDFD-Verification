#!/usr/bin/env python3
"""Read-only source inventory for historical PoP12 and precipitation URMA."""
import boto3,datetime,json
from botocore import UNSIGNED
from botocore.config import Config
s3=boto3.client('s3',region_name='us-east-1',config=Config(signature_version=UNSIGNED,connect_timeout=8,read_timeout=15,retries={'max_attempts':2}))
def listing(bucket,prefix,maxkeys=20):
 try:
  d=s3.list_objects_v2(Bucket=bucket,Prefix=prefix,MaxKeys=maxkeys)
  return {'count':d.get('KeyCount',0),'truncated':d.get('IsTruncated',False),'items':[x['Key'] for x in d.get('Contents',[])]}
 except Exception as e:return {'error':str(e)}
def index(bucket,key,terms=('POP','APCP','QPF','PROB')):
 try:
  txt=s3.get_object(Bucket=bucket,Key=key+'.idx')['Body'].read().decode()
  lines=[line for line in txt.splitlines() if any(k in line.upper() for k in terms)]
  return {'found':len(lines),'examples':lines[:16]}
 except Exception as e:return {'error':str(e)}
for day in ('20261123','20261124'):
 for hh in ('00','06','12','18'):
  print('URMA_PRECIP',day,hh,json.dumps(listing('noaa-urma-pds',f'urma2p5.{day}/urma2p5.{day}{hh}.pcp_06h')))
  print('URMA_HOURLY',day,hh,json.dumps(listing('noaa-urma-pds',f'urma2p5.{day}/urma2p5.t{hh}z.2dvaranl_ndfd')))
for prefix in ('wmo/pop12/2026/11/23/','wmo/pop12/2026/11/24/','wmo/po12/2026/11/23/','wmo/2026/11/23/'):
 print('NDFD_LIST',prefix,json.dumps(listing('noaa-ndfd-pds',prefix,8)))
for family in ('core','nbp','qmd'):
 for cycle in ('13','01'):
  for lead in ('023','035','047','059'):
   date='20261123'
   key=f'blend.{date}/{cycle}/{family}/blend.t{cycle}z.{family}.f{lead}.co.grib2'
   if family=='qmd' and lead!='023':continue
   print('NBM_IDX',key,json.dumps(index('noaa-nbm-grib2-pds',key)))
