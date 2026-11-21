"""Spearman correlation between standalone classifier properties and pipeline results (15 classifiers).
Permutation test: two-sided, N_PERM random permutations of one variable, fixed seed;
p = (1 + #{|rho_perm| >= |rho_obs| - 1e-12}) / (N_PERM + 1). Ties: average ranks (scipy.stats.spearmanr)."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
B=sys.argv[1]; N_PERM=int(sys.argv[2]) if len(sys.argv)>2 else 1_000_000; SEED=20261120
cmp_=pd.read_csv(f"{B}/02_backbone_training/output/comparison_table.csv")
cmp_=cmp_[cmp_.model!="focusnet_small"].copy(); cmp_["model"]=cmp_["model"].replace({"yolo26n_cls":"yolo26n"})
rep=pd.read_csv(f"{B}/03_pipeline_validation/output_selection/test_report.csv")
m=cmp_.merge(rep,on="model"); assert len(m)==15,len(m)
props={"Macro-F1":"test_f1_macro","Accuracy":"test_accuracy","Focused recall":"focused_recall","Focused precision":"focused_precision","Parameters":"params","Latency":"inference_ms"}
outs={"success":"test_success_pct","steps":"test_mean_steps"}
rng=np.random.default_rng(SEED)
def rank(x): return rankdata(x,method="average")
def perm_p(x,y):
    rx,ry=rank(x),rank(y); rx=(rx-rx.mean())/rx.std(); ry=(ry-ry.mean())/ry.std(); n=len(x)
    obs=(rx*ry).mean()
    P=np.array([rng.permutation(ry) for _ in range(0)]) if False else None
    cnt=0; done=0; B_=50_000
    while done<N_PERM:
        k=min(B_,N_PERM-done); idx=np.argsort(rng.random((k,n)),axis=1); r=(rx[None,:]*ry[idx]).mean(1)
        cnt+=int((np.abs(r)>=abs(obs)-1e-12).sum()); done+=k
    return obs,(cnt+1)/(N_PERM+1)
rows=[]
for pn,pc in props.items():
    row={"Property":pn}
    for on,oc in outs.items():
        rho,p=perm_p(m[pc].values,m[oc].values); r2=spearmanr(m[pc],m[oc])[0]; assert abs(rho-r2)<1e-9
        row[f"rho_{on}"]=round(rho,2); row[f"p_{on}"]=round(p,3)
    rows.append(row)
df=pd.DataFrame(rows); print(df.to_string(index=False))
print("ties in success:",m.test_success_pct.duplicated(keep=False).sum(),"values; ties in steps:",m.test_mean_steps.duplicated(keep=False).sum())
print("N_PERM",N_PERM,"SEED",SEED)
df.to_csv(f"{B}/../../INNAS/spearman_perm_result.csv" if False else "spearman_perm_result.csv",index=False)
