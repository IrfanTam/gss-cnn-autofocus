"""R1-6: label of the final (triggering) frame for classifier-terminated runs, 71 test FOV x 15 classifiers.
Label of a frame from its normalised Brenner score P: Focused P>0.55, Medium 0.30<P<=0.55, Blur P<=0.30."""
import sys,pandas as pd
D=sys.argv[1]
t=pd.read_csv(f"{D}/test_per_fov.csv"); r=pd.read_csv(f"{D}/retained_sharpness_per_fov.csv")
r=r[r.model!="gss_only"]
m=t.merge(r[["model","dataset","fov_id","p_final"]],on=["model","dataset","fov_id"],how="left"); assert m.p_final.notna().all() and len(m)==1065,(len(m),m.p_final.isna().sum())
lab=lambda p:"Focused" if p>0.55 else ("Medium" if p>0.30 else "Blur")
m["label"]=m.p_final.map(lab)
print("runs",len(m)); 
for trg,g in m.groupby("trigger"):
    print(trg,len(g),g.label.value_counts().to_dict(),"success",int(g.success.sum()),"mean P",round(g.p_final.mean(),3),"min P",round(g.p_final.min(),3))
c=m[m.trigger=="cnn"]
print("cnn Focused %",round(100*(c.label=="Focused").mean(),2),"; Focused & success",int(((c.label=="Focused")&c.success).sum()),"of",int((c.label=="Focused").sum()))
print("cnn non-success label counts",c[~c.success].label.value_counts().to_dict())
print("cnn blur:",int((c.label=="Blur").sum()),"min P cnn",c.p_final.min())
print("ALL Focused",int((m.label=="Focused").sum()))
print(m.groupby(["trigger","label"]).size())
# per classifier
pc=c.groupby("model").agg(n=("label","size"),focused=("label",lambda x:(x=="Focused").sum()),medium=("label",lambda x:(x=="Medium").sum()),blur=("label",lambda x:(x=="Blur").sum()))
print(pc.to_string())
m.to_csv("trigger_label_per_run.csv",index=False)
