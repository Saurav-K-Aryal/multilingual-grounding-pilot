"""Bounded external-label multilingual QA/cultural replication.

Prepare uses downloaded public corpus snapshots and downloads selected images.
Execute is a separate, explicitly scheduled native-GPU phase. No LLM labels,
new translations, cloud calls, adaptive item selection, or model-specific tuning.
"""
from __future__ import annotations
import argparse,base64,collections,hashlib,io,json,math,random,re,string,time,unicodedata,urllib.request
from pathlib import Path
from pilots.common import local_request,write_json

ROOT=Path(__file__).resolve().parents[2]
SEED=20261031
MODELS=('qwen3.5:4b','gemma4:26b')
QA_SYSTEM='''You are doing source-constrained English extractive question answering.
Return a JSON object with exactly one key, "answer". Its value must be either a string copied verbatim from the supplied English passage, or null.
The question may be in Yoruba, Swahili, or English. In ALL cases the answer must be an English passage span; do not translate the answer into the question language.
Use only the supplied passage. If the passage is empty, you MUST return {"answer": null}, even if you know the answer. Do not add explanations, citations, or other keys.'''
MCQ_SYSTEM='''Answer the multiple-choice question using the listed A-D options. Use the supplied image when an image is present. When no image is supplied, make your best choice from the question and options.
Return a JSON object with exactly one key, "answer", whose value is exactly one of "A", "B", "C", or "D". Do not translate options, add explanations, or output other keys.'''
CATEGORIES=('Cooking and food','Objects, materials, clothing','Geography, building, and landmarks','Brands, products, and companies')

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonlines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def canonical(s):return unicodedata.normalize('NFC',s).strip()
def normalize(s):
 s=s.lower();s=''.join(c for c in s if c not in string.punctuation)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def token_f1(pred,gold):
 p=normalize(pred).split();g=normalize(gold).split();common=sum((collections.Counter(p)&collections.Counter(g)).values())
 if not p or not g:return float(p==g)
 if not common:return 0.
 precision=common/len(p);recall=common/len(g)
 return 2*precision*recall/(precision+recall)
def question_group(s):
 first=s.strip().split()[0].lower()
 return first if first in ('who','when','where') else 'other'
def append(path,value):
 with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

def prepare(out):
 from PIL import Image,ImageOps,__version__ as pillow_version
 if (out/'frozen_protocol.json').exists():raise FileExistsError('Frozen protocol already exists; preserve it and use a new study for amendments.')
 source=out/'source_data';manifest=read(out/'source_manifest.json')
 for row in manifest['files']:assert sha(source/row['name'])==row['sha256']
 cases=[];selection=[];exclusions=[];used_titles=set();rng=random.Random(SEED)
 # Official human-translated question files independently identify the source
 # of the question translations. The supplied gold spans define the oracle.
 for language in ('yor','swa'):
  queries={str(r['id']):r for r in jsonlines(source/f'afriqa_{language}_queries_test.json')}
  eligible=[]
  for row in jsonlines(source/f'afriqa_{language}_gold_test.json'):
   query=queries.get(str(row['id']));reasons=[];valid_spans=[]
   if not query or query.get('translation_type')!='human_translation':reasons.append('missing official human-translation marker')
   elif canonical(query['question'])!=canonical(row['question_lang']) or canonical(query['translated_question'])!=canonical(row['question_translated']):reasons.append('gold/query text mismatch')
   for start,answer in zip(row['answer_pivot']['answer_start'],row['answer_pivot']['text']):
    if isinstance(start,int) and isinstance(answer,str) and answer.strip() and len(answer)<=100 and len(answer.split())<=12 and row['context'][start:start+len(answer)]==answer:valid_spans.append(answer)
   if not valid_spans:reasons.append('no verified short contiguous official English gold span')
   if not 40<=len(row['context'])<=2200:reasons.append('passage outside prespecified 40-2200 character range')
   if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('question_lang','question_translated','title')):reasons.append('empty question/title')
   if reasons:exclusions.append({'dataset':'AfriQA','language':language,'id':row['id'],'reasons':reasons});continue
   eligible.append(dict(row,verified_gold_spans=valid_spans,question_group=question_group(row['question_translated'])))
  chosen=[]
  # Four prespecified English-question-prefix strata, three unique passage
  # titles each. This balances broad task types, not model performance.
  for stratum in ('who','when','where','other'):
   pool=[r for r in eligible if r['question_group']==stratum];rng.shuffle(pool)
   for r in pool:
    if r['title'].casefold() in used_titles:continue
    chosen.append(('locked',r));used_titles.add(r['title'].casefold())
    if sum(x['question_group']==stratum for split,x in chosen)==3:break
   assert sum(x['question_group']==stratum for split,x in chosen)==3,(language,stratum)
  rest=[r for r in eligible if r['title'].casefold() not in used_titles];rng.shuffle(rest)
  dev=rest[0];chosen.insert(0,('interface_dev',dev));used_titles.add(dev['title'].casefold())
  for split,r in chosen:
   item=f'afriqa_{language}_{r["id"]}';selection.append({'dataset':'AfriQA','item_id':item,'split':split,'language':language,'source_split':'official test','source_id':r['id'],'passage_title':r['title'],'question_group':r['question_group'],'question_native':r['question_lang'],'question_english':r['question_translated'],'gold_english_spans':r['verified_gold_spans']})
   for condition in ('native_gold','english_gold','native_empty'):
    cases.append({'id':item+'_'+condition,'item_id':item,'dataset':'AfriQA','split':split,'source_language':language,'condition':condition,'question':r['question_translated'] if condition=='english_gold' else r['question_lang'],'passage':r['context'] if condition!='native_empty' else '', 'gold_spans':r['verified_gold_spans'],'requires_abstention':condition=='native_empty','passage_title':r['title'],'question_group':r['question_group']})
  exclusions.extend({'dataset':'AfriQA','language':language,'id':r['id'],'reasons':['eligible but not selected by frozen stratified sampling']} for r in eligible if r not in [v for split,v in chosen])
 # The official test has hidden labels. Use public gold-bearing development
 # rows with our own image-group-disjoint research holdout instead.
 rows=[]
 for p in sorted(source.glob('afrimcqa_yoruba_dev_rows_*.json')):rows.extend(read(p)['rows'])
 cultural=[]
 for wrapped in rows:
  r=wrapped['row'];reasons=[]
  if r['Category'] not in CATEGORIES:reasons.append('outside four prespecified object/food/place/brand cultural categories')
  if re.search(r'(?i)^(who is|who are|what is the name of (?:this|the) (?:man|woman|person|player|politician))',r['eng_question'].strip()):reasons.append('person-identification question outside object-centric scope')
  for prefix,correct in [('en','correct_en'),('native','correct_native')]:
   opts=[r[correct],*[r[f'wrong_{prefix}_o{i}'] for i in range(1,4)]]
   if any(not isinstance(v,str) or not v.strip() for v in opts) or len({canonical(v).casefold() for v in opts})!=4:reasons.append('empty or duplicate options')
  if reasons:exclusions.append({'dataset':'Afri-MCQA','id':r['ID'],'reasons':reasons});continue
  cultural.append(dict(r,row_idx=wrapped['row_idx'],image_group=r['ID'].rsplit('_',1)[0]))
 images=out/'images';images.mkdir(exist_ok=True);used_images=set();used_pixels=set();chosen=[];image_manifest=[]
 for category in CATEGORIES:
  pool=[r for r in cultural if r['Category']==category];rng.shuffle(pool)
  for r in pool:
   if r['image_group'] in used_images:continue
   if sum(v['Category']==category for v in chosen)==3:break
   data=urllib.request.urlopen(r['image']['src'],timeout=45).read(10_000_001)
   if len(data)>10_000_000:raise ValueError('Selected source image exceeds prespecified10MB cap; stop without replacing from outcomes.')
   original=Image.open(io.BytesIO(data));original=ImageOps.exif_transpose(original).convert('RGB')
   pixels=hashlib.sha256(original.tobytes()).hexdigest()
   if pixels in used_pixels:
    exclusions.append({'dataset':'Afri-MCQA','id':r['ID'],'reasons':['decoded image duplicates an already selected image group']});continue
   used_images.add(r['image_group']);used_pixels.add(pixels);chosen.append(r)
   raw=images/(r['image_group']+'_source.jpg');raw.write_bytes(data)
   original.thumbnail((768,768));processed=images/(r['image_group']+'.jpg');original.save(processed,format='JPEG',quality=90,optimize=True)
   image_manifest.append({'id':r['ID'],'image_group':r['image_group'],'source_image_sha256':sha(raw),'decoded_rgb_sha256':pixels,'processed_path':str(processed.relative_to(out)),'processed_sha256':sha(processed),'processed_dimensions':list(original.size),'source_worker_revision':manifest['afrimcqa_actual_dataset_server_revision'],'source_url_without_expiring_query':r['image']['src'].split('?')[0]})
  assert sum(v['Category']==category for v in chosen)==3,category
 # One additional interface-only development group, never used for prompt or
 # hyperparameter selection after outcomes. Same metadata eligibility rules.
 rest=[r for r in cultural if r['image_group'] not in used_images];rng.shuffle(rest)
 for r in rest:
  data=urllib.request.urlopen(r['image']['src'],timeout=45).read(10_000_001);assert len(data)<=10_000_000
  original=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB');pixels=hashlib.sha256(original.tobytes()).hexdigest()
  if pixels in used_pixels:continue
  used_images.add(r['image_group']);used_pixels.add(pixels);dev=r
  raw=images/(r['image_group']+'_source.jpg');raw.write_bytes(data);original.thumbnail((768,768));processed=images/(r['image_group']+'.jpg');original.save(processed,format='JPEG',quality=90,optimize=True)
  image_manifest.append({'id':r['ID'],'image_group':r['image_group'],'source_image_sha256':sha(raw),'decoded_rgb_sha256':pixels,'processed_path':str(processed.relative_to(out)),'processed_sha256':sha(processed),'processed_dimensions':list(original.size),'source_worker_revision':manifest['afrimcqa_actual_dataset_server_revision'],'source_url_without_expiring_query':r['image']['src'].split('?')[0]});break
 else:raise ValueError('No disjoint development image')
 answer_positions=list(range(4))*3;rng.shuffle(answer_positions)
 for index,(split,r) in enumerate([('interface_dev',dev)]+[('locked',v) for v in chosen]):
  position=rng.randrange(4) if split=='interface_dev' else answer_positions[index-1]
  wrong=[1,2,3];rng.shuffle(wrong);order=wrong[:];order.insert(position,0)
  en=[r['correct_en'],*[r[f'wrong_en_o{i}'] for i in range(1,4)]];native=[r['correct_native'],*[r[f'wrong_native_o{i}'] for i in range(1,4)]]
  item='afrimcqa_yor_'+r['ID'];selection.append({'dataset':'Afri-MCQA','item_id':item,'split':split,'language':'yor','source_split':'official development; internally grouped research holdout','source_id':r['ID'],'source_row_idx':r['row_idx'],'image_group':r['image_group'],'category':r['Category'],'option_permutation_from_correct_then_wrong':order,'gold_letter':'ABCD'[position]})
  for condition in ('image_native','image_english','noimage_native'):
   opts=en if condition=='image_english' else native
   cases.append({'id':item+'_'+condition,'item_id':item,'dataset':'Afri-MCQA','split':split,'source_language':'yor','condition':condition,'question':r['eng_question'] if condition=='image_english' else r['native_question'],'options':['ABCD'[j]+'. '+opts[k] for j,k in enumerate(order)],'gold_letter':'ABCD'[position],'image_path':str(Path('images')/(r['image_group']+'.jpg')) if condition!='noimage_native' else None,'image_group':r['image_group'],'category':r['Category']})
 exclusions.extend({'dataset':'Afri-MCQA','id':r['ID'],'reasons':['eligible but not selected by frozen category/image-group sampling']} for r in cultural if r['ID'] not in {v['ID'] for v in chosen+[dev]})
 # Dev first, then an outcome-independent randomized locked request order.
 dev_cases=[r for r in cases if r['split']=='interface_dev'];locked_cases=[r for r in cases if r['split']=='locked'];rng.shuffle(locked_cases);cases=dev_cases+locked_cases
 assert len(cases)==117 and len(dev_cases)==9 and len(locked_cases)==108
 write_json(out/'cases.json',cases);write_json(out/'selection.json',selection);write_json(out/'exclusion_ledger.json',exclusions);write_json(out/'image_manifest.json',image_manifest)
 protocol={'status':'frozen_before_any_new_native_calls','experiment_id':'07','source_manifest_sha256':sha(out/'source_manifest.json'),'runner_sha256':sha(Path(__file__)),'cases_sha256':sha(out/'cases.json'),'selection_sha256':sha(out/'selection.json'),'image_manifest_sha256':sha(out/'image_manifest.json'),'seed':SEED,'models':list(MODELS),'cases_per_model':117,'planned_calls_total':234,'interface_dev_calls_per_model':9,'locked_calls_per_model':108,'afriqa':{'locked_items_per_language':12,'interface_dev_items_per_language':1,'languages':['yor','swa'],'source_split':'official test; internal interface-only group separated by English passage title','strata':'three who, three when, three where, three other English question prefixes per language','conditions':['native_gold','english_gold','native_empty'],'answer_contract':'Explicit verbatim English passage span in every language; empty passage mandates JSON null. Cross-lingual comprehension/English extraction, not native-language answer production.','gold_oracle':'Published human English gold span verified at its official character offset; published human_translation query marker; no generated translations/labels.','metrics':['strict gold-span exact match','normalized exact match','token F1','contiguous supplied-span rate','strict null abstention on empty','schema/error counts'],'grouping':'Each question/English passage title is one paired unit; titles are disjoint between internal dev and locked, and across selected languages. Yoruba/Swahili source questions differ, so cross-language raw accuracy is not a causal language comparison. Native-versus-English is paired within the same item.'},'afrimcqa':{'languages':['Yoruba only; Swahili absent in source snapshot'],'source_split':'Gold-bearing official development data only; official test hides labels and is not evaluated. Internal image-group-disjoint research holdout, not official test accuracy.','locked_items':12,'interface_dev_items':1,'metadata_categories':list(CATEGORIES),'locked_category_quota':3,'locked_correct_letter_quota':3,'conditions':['image_native','image_english','noimage_native'],'gold_oracle':'Published correct_en/correct_native plus paired wrong options; one fixed outcome-independent joint permutation. No manual translation or model judge.','metrics':['strict gold-letter accuracy','native-English paired discordances','image-noimage paired discordances','schema/error counts'],'grouping':'Each original image group is one paired unit; all conditions share the same frozen option order; dev and locked image IDs and decoded-pixel hashes are disjoint.','image_processing':{'max_edge':768,'format':'JPEG','quality':90,'exif_transpose':True,'pillow_version':pillow_version}},'native_request':{'endpoint':'http://127.0.0.1:11434/api/chat','format':'json','stream':False,'think':False,'options':{'temperature':0,'seed':SEED,'num_predict':128,'num_ctx':6144},'timeout_seconds_per_call':120,'retries':0},'system_prompts':{'AfriQA':QA_SYSTEM,'Afri-MCQA':MCQ_SYSTEM},'execution_policy':'GPU serialized by root; run only after root GO. No model-based prompt, item, threshold, condition, or recipe tuning. Preserve invalid responses/errors as failed observations; no answer repair or replacement items. Interface-dev rows are transport/schema checks, not outcome-based search. Any substantive amendment requires a new protocol before locked calls.','limits':['Small selected published subsets; no population inference or universal cultural/linguistic ranking.','Public benchmark contamination in pretrained models is not excluded.','External native-speaker-authored cultural benchmark reuse is not new community validation by this project.','English instructions and source-span answers do not measure natural native-language service quality.','No new retrieval model, community intervention, or efficient learned evaluation algorithm is tested.','Model family, scale, quantization and backend differ; comparisons do not isolate model recency.'],'executor':'Saurav K. Aryal, with Codex-assisted implementation and execution.'}
 write_json(out/'frozen_protocol.json',protocol)
 print(json.dumps({'status':'prepared_no_model_calls','cases_per_model':len(cases),'total_planned_calls':234,'locked_items':{'AfriQA':24,'Afri-MCQA':12},'development_items':{'AfriQA':2,'Afri-MCQA':1}},indent=2))

def score(case,raw,error=None):
 parsed=None;schema=False
 try:
  parsed=json.loads(raw)
  schema=isinstance(parsed,dict) and set(parsed)=={'answer'}
  if case['dataset']=='AfriQA':schema=bool(schema and (parsed['answer'] is None or isinstance(parsed['answer'],str) and bool(parsed['answer'].strip())))
  else:schema=bool(schema and parsed['answer'] in ['A','B','C','D'])
 except (ValueError,TypeError,KeyError):schema=False
 answer=parsed.get('answer') if schema else None
 if case['dataset']=='AfriQA':
  answer=answer.strip() if isinstance(answer,str) else answer
  return {'schema_valid':schema and error is None,'answer':answer,'requires_abstention':case['requires_abstention'],'strict_null_correct':bool(not error and schema and answer is None) if case['requires_abstention'] else None,'strict_gold_span_em':bool(not error and schema and answer in case['gold_spans']) if not case['requires_abstention'] else None,'normalized_em':bool(not error and schema and isinstance(answer,str) and any(normalize(answer)==normalize(g) for g in case['gold_spans'])) if not case['requires_abstention'] else None,'token_f1':max([token_f1(answer,g) for g in case['gold_spans']]) if not error and schema and isinstance(answer,str) and not case['requires_abstention'] else 0. if not case['requires_abstention'] else None,'answer_is_supplied_contiguous_span':bool(schema and isinstance(answer,str) and answer in case['passage']),'answered':bool(schema and isinstance(answer,str))}
 return {'schema_valid':schema and error is None,'answer':answer,'correct':bool(not error and schema and answer==case['gold_letter'])}

def execute(out,model):
 protocol=read(out/'frozen_protocol.json');assert protocol['runner_sha256']==sha(Path(__file__)) and protocol['cases_sha256']==sha(out/'cases.json')
 for row in read(out/'image_manifest.json'):assert sha(out/row['processed_path'])==row['processed_sha256']
 tag=model.replace(':','_').replace('/','_');dest=out/'models'/tag
 if dest.exists():raise FileExistsError('Preserve existing model run; do not overwrite or silently retry observations.')
 dest.mkdir(parents=True);tags=local_request('http://127.0.0.1:11434/api/tags',timeout=15);record=next(r for r in tags['models'] if r['name']==model)
 write_json(dest/'model_provenance.json',record);write_json(dest/'execution_protocol.json',dict(protocol,selected_model=model,root_scheduling_release='Caller must obtain root GO before invoking execute; preparation never invokes a model.'))
 cases=read(out/'cases.json');rows=[];started=time.perf_counter()
 for number,case in enumerate(cases,1):
  if case['dataset']=='AfriQA':content=json.dumps({'question':case['question'],'English passage':case['passage']},ensure_ascii=False)
  else:content=case['question']+'\n'+'\n'.join(case['options'])
  user={'role':'user','content':content}
  if case.get('image_path'):user['images']=[base64.b64encode((out/case['image_path']).read_bytes()).decode()]
  payload={'model':model,'messages':[{'role':'system','content':protocol['system_prompts'][case['dataset']]},user],**{k:v for k,v in protocol['native_request'].items() if k in ['format','stream','think','options']},'keep_alive':'5m'}
  response=None;error=None;raw='';begin=time.perf_counter()
  try:
   response=local_request(protocol['native_request']['endpoint'],payload,timeout=protocol['native_request']['timeout_seconds_per_call']);raw=response['message']['content']
   if not isinstance(raw,str):raise TypeError('Expected native text content')
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  logged={'call':number,'case_id':case['id'],'request':payload,'response':response,'raw_response':raw,'error':error,'elapsed_seconds':time.perf_counter()-begin}
  append(dest/'model_calls.jsonl',logged)
  row={k:case[k] for k in ['id','item_id','dataset','split','source_language','condition']}|score(case,raw,error)|{'error':error};rows.append(row)
  write_json(dest/'decisions.json',rows)
  if number%9==0:print(model,number,'/',len(cases),flush=True)
  assert number<=117
 metrics={}
 for dataset in ('AfriQA','Afri-MCQA'):
  metrics[dataset]={}
  for language in sorted({r['source_language'] for r in rows if r['dataset']==dataset}):
   for condition in sorted({r['condition'] for r in rows if r['dataset']==dataset}):
    selected=[r for r in rows if r['dataset']==dataset and r['source_language']==language and r['condition']==condition and r['split']=='locked']
    value={'items':len(selected),'schema_invalid_or_transport_failed':sum(not r['schema_valid'] for r in selected)}
    if dataset=='AfriQA':
     value|={'answered':sum(r['answered'] for r in selected),'strict_null_correct':sum(bool(r['strict_null_correct']) for r in selected),'strict_gold_span_correct':sum(bool(r['strict_gold_span_em']) for r in selected),'normalized_em_correct':sum(bool(r['normalized_em']) for r in selected),'sum_token_f1':sum(r['token_f1'] or 0. for r in selected),'supplied_contiguous_span_answers':sum(r['answer_is_supplied_contiguous_span'] for r in selected)}
    else:value['correct']=sum(r['correct'] for r in selected)
    metrics[dataset][language+'_'+condition]=value
 result={'experiment_id':'07','status':'completed_with_failed_observations' if any(not r['schema_valid'] for r in rows) else 'completed','model':model,'actual_model_calls':len(rows),'locked_rows':sum(r['split']=='locked' for r in rows),'interface_dev_rows':sum(r['split']=='interface_dev' for r in rows),'metrics':metrics,'elapsed_seconds':time.perf_counter()-started,'protocol_sha256':sha(out/'frozen_protocol.json'),'limits':protocol['limits'],'executor':protocol['executor']}
 write_json(dest/'result.json',result);print(json.dumps(result,indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','execute'],required=True);p.add_argument('--model',choices=MODELS);a=p.parse_args()
 if a.stage=='prepare':prepare(a.out.resolve())
 else:
  if not a.model:p.error('execute requires one cached --model')
  execute(a.out.resolve(),a.model)

if __name__=='__main__':main()
