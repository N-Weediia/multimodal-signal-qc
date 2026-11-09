"""Relative MBLL sensitivity specification and data audit."""
from pathlib import Path
import json, csv, re, argparse
import numpy as np
from scipy import signal
from scipy.io import loadmat
import sys
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data_repro_20260903'; OUT=ROOT/'mbll_relative_sensitivity_20260908'
sys.path.insert(0,str(ROOT)); from fnirs_qc import read_fnirs, build_fnirs_sample_masks, is_infrared_eye_tracking_task
EXT={730.0:{'hbo':1094.0,'hbr':1800.0},850.0:{'hbo':2526.0,'hbr':1798.0}}
def dpf(w,a):
 a=float(np.clip(a,18,90)); l=float(w); return 223.3+.05624*a**.8493-5.723e-7*l**3+.001245*l**2-.9025*l
def age(p):
 ms=re.findall(r'(?:19|20)\d{2}[-_]?(?:\d{2})[-_]?(?:\d{2})',p.name)
 if not ms:return 60.
 s=ms[-1]; return float(max(18,min(90,2026-int(s[:4])-((9-int(s[5:7]))*30<int(s[8:10])))))
def od(x):
 x=np.asarray(x,float); v=np.isfinite(x)&(x>0); b=np.nanmedian(np.where(v,x,np.nan),axis=0,keepdims=True); z=-np.log(np.maximum(x,1e-12)/np.maximum(b,1e-12))
 for j in range(z.shape[1]):
  q=np.isfinite(z[:,j]); z[~q,j]=np.interp(np.flatnonzero(~q),np.flatnonzero(q),z[q,j]) if q.sum()>=2 else 0
 return z
def marks(p):
 try:
  d=loadmat(p,variable_names=['t','s'],simplify_cells=True); t=np.ravel(np.asarray(d.get('t',[]),float)); s=np.asarray(d.get('s',[])); ix=np.flatnonzero(s!=0) if s.ndim==1 else np.flatnonzero(np.any(s!=0,axis=1)); out=[]
  for v in t[ix]:
   if not out or v-out[-1]>1:out.append(float(v))
  return out
 except Exception:return []
def solve(p,ds,dm,lo,hi):
 r=read_fnirs(p,None); fs=r.sample_rate_hz; a=np.column_stack([c.intensity_a for c in r.channels]); b=np.column_stack([c.intensity_b for c in r.channels]); z=np.stack([od(a),od(b)],axis=1); aa=age(p); dd=[]
 for c in r.channels:dd.append(np.linalg.norm(r.source_positions[c.source-1,:2]-r.detector_positions[c.detector-1,:2]))
 dd=np.asarray(dd); L=dd if dm=='real' else np.full_like(dd,{'fixed5':5.,'fixed45':4.5,'fixed578':5.78}[dm]); E=np.array([[EXT[730.]['hbo']*dpf(730,aa)*ds,EXT[730.]['hbr']*dpf(730,aa)*ds],[EXT[850.]['hbo']*dpf(850,aa)*ds,EXT[850.]['hbr']*dpf(850,aa)*ds]])
 v=np.empty((len(a),2,len(r.channels))); inv=np.linalg.pinv(E)
 for j,l in enumerate(L):v[:,:,j]=(inv@z[:,:,j].T).T/l*1e6
 m=np.stack([build_fnirs_sample_masks(c,fs)[3] for c in r.channels],axis=1)
 for j in range(v.shape[2]): v[~m[:,j],:,j]=np.nan
 out=[]; sos=signal.butter(4,(lo,min(hi,fs*.45)),btype='bandpass',fs=fs,output='sos')
 for k in range(2):v[:,k,:]=signal.sosfiltfilt(sos,np.where(np.isfinite(v[:,k,:]),v[:,k,:],np.nanmedian(v[:,k,:],axis=0,keepdims=True)),axis=0)
 for i,on in enumerate(marks(p)):
  s=int(round(on*fs)); p0=s-int(5*fs); p1=s+int(2*fs); p2=s+int(12*fs)
  if p0>=0 and p2<=len(v):out.append((i+1,float(np.nanmean(v[p1:p2,0,:])-np.nanmean(v[p0:s,0,:])),float(np.nanmean(v[p1:p2,1,:])-np.nanmean(v[p0:s,1,:]))))
 return out,{'age_years':aa,'dpf_730':dpf(730,aa)*ds,'dpf_850':dpf(850,aa)*ds,'distance_median_cm':float(np.median(dd)),'distance_min_cm':float(np.min(dd)),'distance_max_cm':float(np.max(dd)),'n_channels':len(r.channels),'condition':float(np.linalg.cond(E)),'branch':'EYE_IR_DEGRADED_MASKED' if is_infrared_eye_tracking_task(p) else 'STANDARD_MASKED'}
def tname(i):return '摸头' if 3<=i<=7 else '摸膝' if 9<=i<=13 else '摸腰' if 15<=i<=19 else None
def run():
 OUT.mkdir(parents=True,exist_ok=True); ev=[]; au=[]
 for p in sorted(RAW.glob('*/*/FNIRS/*/*.nirs')):
  sub, ses, task=p.parts[-5],p.parts[-4],p.parent.name
  if task not in ('运动','眼动'):continue
  for ds in (.8,1.,1.2):
   for dm in ('real','fixed5','fixed45','fixed578'):
    for lo,hi in ((.005,.2),(.01,.2),(.01,.3)):
     var=f'dpf{ds:g}_{dm}_f{lo:g}-{hi:g}'
     try: rr,meta=solve(p,ds,dm,lo,hi); au.append({'file':str(p.relative_to(ROOT)),'subject':sub,'session':ses,'task':task,'variant':var,**meta,'status':'ok'})
     except Exception as e:au.append({'file':str(p.relative_to(ROOT)),'subject':sub,'session':ses,'task':task,'variant':var,'status':'error','reason':str(e)});continue
     for i,h,o in rr:
      tn=tname(i)
      if tn:ev.append({'subject':sub,'session':ses,'task':task,'branch':meta['branch'],'variant':var,'task_name':tn,'marker_index':i,'relative_hbo_uM':h,'relative_hbr_uM':o,'dpf_scale':ds,'distance_mode':dm,'filter_low':lo,'filter_high':hi})
 fields_ev=['subject','session','task','branch','variant','task_name','marker_index','relative_hbo_uM','relative_hbr_uM','dpf_scale','distance_mode','filter_low','filter_high']
 fields_au=['file','subject','session','task','variant','age_years','dpf_730','dpf_850','distance_median_cm','distance_min_cm','distance_max_cm','n_channels','condition','branch','status','reason']
 for n,rows,fields in [('mbll_event_responses.csv',ev,fields_ev),('mbll_run_audit.csv',au,fields_au)]:
  with (OUT/n).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 with (OUT/'metadata.json').open('w',encoding='utf-8') as f:json.dump({"wavelengths_nm":[730,850],"channels":44,"spatial_unit":"cm","distance_range_cm":[4.5,5.78],"extinction_source":"Prahl/OMLC https://omlc.org/spectra/hemoglobin/summary.html","extinction_units":"cm^-1 M^-1","dpf_source":"Scholkmann et al. NeuroImage 2014 doi:10.1016/j.neuroimage.2013.06.056","dpf_formula":"223.3 + 0.05624*age^0.8493 - 5.723e-7*lambda^3 + 0.001245*lambda^2 - 0.9025*lambda","interpretation":"relative HbO/HbR indices only; not absolute concentration; eye branch separate; EOG/ECG unconnected"},f,ensure_ascii=False,indent=2)
 (OUT/'README.md').write_text('Raw 730/850 nm intensity -> OD -> two-wavelength MBLL. Prahl/OMLC extinction coefficients and Scholkmann age DPF. Variants cover DPF 0.8/1/1.2, real and fixed SD distances, and fourth-order Butterworth 0.005-0.2, 0.01-0.2, 0.01-0.3 Hz. Results are relative HbO/HbR indices, not absolute concentrations. Eye recordings retain EYE_IR_DEGRADED_MASKED. EOG/ECG electrodes were unconnected, so no correction was applied.\n',encoding='utf-8')
 print('events',len(ev),'audits',len(au),'out',OUT)
if __name__=='__main__': run()
