"""R2-5: paired difference in success rate (GSS-only minus classifier-assisted), 71 test FOV, tolerance 1.0 deg.
Percentile bootstrap over FOV (10,000 resamples, fixed seed) and Newcombe-Wilson paired interval (method 10)."""
import sys,numpy as np,pandas as pd
D=sys.argv[1]; rng=np.random.default_rng(20261120)
t=pd.read_csv(f"{D}/test_per_fov.csv"); u=pd.read_csv(f"{D}/../output_unitfix/per_fov.csv")
g=u[(u.strategy=="gss_only")&(u.delta==1.0)][["dataset","fov_id","success"]].rename(columns={"success":"g"})
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newcombe(b,c,n,z=1.96):
    # paired: p1=(a+b)/n, p2=(a+c)/n, discordant b (g only), c (model only)
    # use phi correlation per Newcombe 1998 method 10 given the 2x2
    return None
rows=[]
for m in ("vit_tiny","focusnet_nano","focusnet_micro","yolo26n","swin_tiny"):
    x=t[t.model==m].merge(g,on=["dataset","fov_id"]); n=len(x)
    gs=x.g.values.astype(int); cs=x.success.values.astype(int)
    d=gs-cs; est=d.mean()
    bs=[d[rng.integers(0,n,n)].mean() for _ in range(10000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    b=int(((gs==1)&(cs==0)).sum()); c=int(((gs==0)&(cs==1)).sum())
    # Newcombe paired interval method 10
    p1=gs.mean(); p2=cs.mean(); l1,u1=wil(gs.sum(),n); l2,u2=wil(cs.sum(),n)
    a=int(((gs==1)&(cs==1)).sum()); dd=int(((gs==0)&(cs==0)).sum())
    den=(a+b)*(c+dd)*(a+c)*(b+dd)
    phi=0 if den==0 else (a*dd-b*c)/np.sqrt(den)
    A=(a+b)*(a+c)*(b+dd)*(c+dd); 
    phi=0 if A==0 else (a*dd-b*c)/np.sqrt(A)
    phi=max(phi,0)
    D_=p1-p2
    L=D_-np.sqrt((p1-l1)**2+(u2-p2)**2-2*phi*(p1-l1)*(u2-p2)); U=D_+np.sqrt((u1-p1)**2+(p2-l2)**2-2*phi*(u1-p1)*(p2-l2))
    rows.append((m,int(cs.sum()),n,b,c,round(100*est,1),round(100*lo,1),round(100*hi,1),round(100*L,1),round(100*U,1)))
df=pd.DataFrame(rows,columns=["model","success","n","gss_only_wins","model_wins","loss_pp","boot_lo","boot_hi","newcombe_lo","newcombe_hi"]); print(df.to_string(index=False)); df.to_csv("paired_ci_result.csv",index=False)
