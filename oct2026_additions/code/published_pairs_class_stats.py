"""Descriptive stats of the 30 published pairs by retrieval-structure class.
Rows copied from analysis/scatter_circularity_gain.py ROWS (same order as master table rows 1-14, 16-31)."""
import json, statistics as st
# (row#, benchmark, C_KG, delta, metric, flag, system, class)
R = [
 (1,"MuSiQue",0.00,10.7,"recall","", "HippoRAG",2),
 (2,"MuSiQue",0.00,8.6,"recall","", "HippoRAG",2),
 (3,"MuSiQue",0.00,31.2,"recall","", "HippoRAG 2",2),
 (4,"MuSiQue",0.00,28.4,"em_proxy","", "HopRAG",2),
 (5,"HotpotQA",0.00,5.5,"recall","", "HippoRAG",2),
 (6,"HotpotQA",0.00,5.1,"recall","", "HippoRAG",2),
 (7,"HotpotQA",0.00,21.5,"recall","", "HippoRAG 2",2),
 (8,"MultiHop-RAG",0.00,0.9,"map","", "KG-CQR",2),
 (9,"MultiHop-RAG",0.00,1.9,"recall","", "KG-CQR",2),
 (10,"MultiHop-RAG",0.00,4.6,"recall","", "KG-CQR",2),
 (11,"NaturalQuestions",0.00,-11.7,"recall","", "HippoRAG (repro)",2),
 (12,"NaturalQuestions",0.00,21.9,"recall","", "HippoRAG 2",2),
 (13,"NaturalQuestions",0.00,4.8,"em_proxy","", "GraphRetriever",4),
 (14,"TriviaQA",0.00,1.0,"em_proxy","", "GraphRetriever",4),
 (16,"STaRK-MAG",0.75,32.3,"hit1","", "MoR",1),
 (17,"STaRK-Amazon",0.75,7.3,"hit1","", "MoR",1),
 (18,"STaRK-Prime",0.75,23.7,"hit1","", "MoR",1),
 (19,"STaRK-MAG",0.75,-13.0,"hit1","", "QAGNN",1),
 (20,"GraphRAG-Bench",0.75,23.4,"recall","dense_baseline","HippoRAG",2),
 (21,"GraphRAG-Bench",0.75,21.1,"recall","dense_baseline","LightRAG",2),
 (22,"2WikiMultiHopQA",1.00,27.2,"recall","", "HippoRAG",2),
 (23,"2WikiMultiHopQA",1.00,18.9,"recall","", "HippoRAG",2),
 (24,"2WikiMultiHopQA",1.00,25.1,"recall","", "HippoRAG 2",2),
 (25,"MetaQA",1.00,18.9,"hit1","", "PullNet",1),
 (26,"MetaQA",1.00,13.2,"hit1","", "PullNet",1),
 (27,"MetaQA",1.00,58.6,"hit1","", "GraftNet",1),
 (28,"MetaQA",1.00,37.5,"hit1","", "GraftNet",1),
 (29,"WebQSP",1.00,43.3,"hit1","", "PullNet",1),
 (30,"WebQSP",1.00,41.5,"hit1","", "GraftNet",1),
 (31,"WebQSP",1.00,23.5,"hit1","", "KV-Mem",1),
]
assert len(R)==30
def med(xs):
    xs=sorted(xs); n=len(xs)
    return xs[n//2] if n%2 else round(0.5*(xs[n//2-1]+xs[n//2]),2)
def q(xs):
    xs=sorted(xs); 
    if len(xs)<4: return None
    qs=st.quantiles(xs,n=4,method="inclusive"); return [round(v,2) for v in (qs[0],qs[2])]
def summ(rows):
    d=[r[3] for r in rows]
    return {"n":len(d),"median":med(d),"IQR":q(d),"min":min(d),"max":max(d),"mean":round(sum(d)/len(d),2)} if d else {"n":0}
out={}
for c in (1,2,3,4):
    out[f"class{c}_all"]=summ([r for r in R if r[7]==c])
    out[f"class{c}_sparse"]=summ([r for r in R if r[7]==c and r[5]==""])
hi=[r for r in R if r[2]>=0.75]; lo=[r for r in R if r[2]==0]
out["hiC_class1"]=summ([r for r in hi if r[7]==1])
out["hiC_class2_all"]=summ([r for r in hi if r[7]==2])
out["hiC_class2_sparse(2Wiki)"]=summ([r for r in hi if r[7]==2 and r[5]==""])
out["loC_class2"]=summ([r for r in lo if r[7]==2])
out["loC_class4"]=summ([r for r in lo if r[7]==4])
# benchmark medians with class composition
bm={}
for r in R:
    bm.setdefault(r[1],[]).append(r)
bench=[]
for b,rs in bm.items():
    cls=sorted(set(r[7] for r in rs))
    bench.append({"benchmark":b,"C_KG":rs[0][2],"n":len(rs),"median":med([r[3] for r in rs]),"classes":cls,
                  "class_counts":{str(c):sum(1 for r in rs if r[7]==c) for c in cls},"dense":rs[0][5]=="dense_baseline"})
out["benchmarks"]=sorted(bench,key=lambda x:(x["C_KG"],x["median"]))
# benchmark-median summary by (modal) class, sparse only
def modal(b): 
    cc=b["class_counts"]; return int(max(cc,key=lambda k:cc[k]))
for c in (1,2,4):
    ms=[b["median"] for b in bench if not b["dense"] and modal(b)==c]
    out[f"bench_medians_modalclass{c}"]={"n":len(ms),"values":sorted(ms),"median_of_medians":med(ms) if ms else None}
json.dump(out,open("class_stats.json","w"),indent=1)
print(json.dumps(out,indent=1))
