import csv, json, sys
rows=list(csv.DictReader(open('judge_spotcheck_sheet.csv',encoding='utf-8-sig',newline='')))
key=json.load(open('sealed_judge_verdicts.json'))
hv=[r['your_verdict_CORRECT_or_INCORRECT'].strip().upper() for r in rows]
if any(v not in ('CORRECT','INCORRECT','UNSURE') for v in hv): sys.exit('ABORT: fill every row with CORRECT, INCORRECT, or UNSURE')
n_unsure=sum(v=='UNSURE' for v in hv)
keep=[i for i,v in enumerate(hv) if v!='UNSURE']
h=[hv[i] for i in keep]; j=[key[rows[i]['key']] for i in keep]
n=len(h); agree=sum(a==b for a,b in zip(h,j))/n
ph=sum(v=='CORRECT' for v in h)/n; pj=sum(v=='CORRECT' for v in j)/n; pe=ph*pj+(1-ph)*(1-pj)
kappa=(agree-pe)/(1-pe)
tp=sum(a=='CORRECT' and b=='CORRECT' for a,b in zip(h,j)); fp=sum(a=='INCORRECT' and b=='CORRECT' for a,b in zip(h,j)); fn=sum(a=='CORRECT' and b=='INCORRECT' for a,b in zip(h,j))
out={'n':n,'n_unsure':n_unsure,'agreement':agree,'kappa':kappa,'judge_precision_vs_human':tp/max(1,tp+fp),'judge_recall_vs_human':tp/max(1,tp+fn)}
json.dump(out,open('spotcheck_result.json','w'),indent=1); print(out)
