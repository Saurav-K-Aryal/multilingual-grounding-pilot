"""Verify all 234 recorded outputs and metric cells using Python only."""
from pathlib import Path
import collections,hashlib,json,math,re,string

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def normalize(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def f1(a,b):
 a=normalize(a).split();b=normalize(b).split()
 if not a or not b:return float(a==b)
 common=sum((collections.Counter(a)&collections.Counter(b)).values());return 2*common/(len(a)+len(b))
def main():
 metadata={r['id']:r for r in read(HERE/'data/cases_metadata.json')};checked=0
 for folder in sorted((HERE/'results/observed').iterdir()):
  scores=read(folder/'decisions.json');result=read(folder/'result.json');responses=[json.loads(s) for s in (folder/'responses.jsonl').read_text().splitlines()]
  assert len(scores)==len(responses)==117
  for row,response in zip(scores,responses):
   case=metadata[row['id']];assert response['case_id']==row['id'] and response['error'] is None
   obj=json.loads(response['raw_response']);assert set(obj)=={'answer'};answer=obj['answer'];assert answer is None or isinstance(answer,str)
   answer=answer.strip() if isinstance(answer,str) else answer
   assert row['answer']==answer and row['schema_valid']
   if case['dataset']=='AfriQA':
    if case['requires_abstention']:assert row['strict_null_correct']==(answer is None)
    else:
     assert row['strict_gold_span_em']==(answer in case['gold_spans'])
     assert row['normalized_em']==bool(isinstance(answer,str) and any(normalize(answer)==normalize(g) for g in case['gold_spans']))
     expected=max(f1(answer,g) for g in case['gold_spans']) if isinstance(answer,str) else 0.
     assert math.isclose(expected,row['token_f1'],abs_tol=1e-10)
   else:assert row['correct']==(answer==case['gold_letter']) and answer in ['A','B','C','D']
   checked+=1
  for dataset,groups in result['metrics'].items():
   for key,expected in groups.items():
    lang,condition=key.split('_',1);rows=[r for r in scores if r['split']=='locked' and r['dataset']==dataset and r['source_language']==lang and r['condition']==condition];assert len(rows)==expected['items']==12
    if dataset=='Afri-MCQA':assert sum(r['correct'] for r in rows)==expected['correct']
    else:
     for field,key in [('strict_gold_span_em','strict_gold_span_correct'),('normalized_em','normalized_em_correct'),('strict_null_correct','strict_null_correct'),('answered','answered'),('answer_is_supplied_contiguous_span','supplied_contiguous_span_answers')]:assert sum(bool(r[field]) for r in rows)==expected[key]
     assert math.isclose(sum(r['token_f1'] or 0. for r in rows),expected['sum_token_f1'],abs_tol=1e-10)
  print(result['model']+':117 responses and all locked metric cells verified.')
 assert checked==234
 # The manifest binds projected public files to their unchanged original hashes.
 provenance=read(HERE/'PUBLICATION_PROVENANCE.json')
 for row in provenance['files']+provenance.get('generated_files',[]):assert hashlib.sha256((HERE/row['published']).read_bytes()).hexdigest()==row['published_sha256'],row['published']
 print('PASS:234 outputs, public-file hashes, strict labels, token F1 and aggregates; no model calls.')
if __name__=='__main__':main()
