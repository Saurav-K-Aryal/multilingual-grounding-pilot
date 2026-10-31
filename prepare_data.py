"""Fetch official inputs for a fresh, identical-case replay; never run models."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,subprocess,sys,urllib.request

HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.exists() and any(out.iterdir()):p.error('Use a fresh empty directory.')
 source=out/'source_data';source.mkdir(parents=True)
 original=json.loads((HERE/'protocol/source_manifest.json').read_text());manifest={k:v for k,v in original.items() if k!='files'};manifest['files']=[]
 for entry in original['files']:
  req=urllib.request.Request(entry['url'],headers={'User-Agent':'academic-multilingual-pilot-reproduction'})
  with urllib.request.urlopen(req,timeout=60) as f:data=f.read(10_000_001)
  if len(data)>10_000_000:raise ValueError('Unexpected oversized source response.')
  if entry['name'].startswith('afriqa_') and hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Pinned AfriQA source changed.')
  if entry['name'].startswith('afrimcqa_'):
   rows=json.loads(data)['rows'];revisions={re.search(r'/--/([a-f0-9]{40})/--/',r['row']['image']['src']).group(1) for r in rows}
   if revisions!={original['afrimcqa_actual_dataset_server_revision']}:raise ValueError('Dataset server now exposes a different source revision. Stop rather than resample or relabel. The required pinned Hub revision is recorded in protocol/source_manifest.json.')
  (source/entry['name']).write_bytes(data);manifest['files'].append(dict(entry,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 subprocess.run([sys.executable,'-m','pilots.strengthening.multilingual_external_qa','--stage','prepare','--out',str(out)],cwd=HERE,check=True)
 expected=json.loads((HERE/'protocol/frozen_protocol.json').read_text())
 actual=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest()
 if actual!=expected['cases_sha256']:raise ValueError('Re-created case identities/text differ from the observed study; no inference executed.')
 frozen_images=json.loads((HERE/'protocol/image_manifest.json').read_text());fresh_images=json.loads((out/'image_manifest.json').read_text());old={r['image_group']:r['processed_sha256'] for r in frozen_images};new={r['image_group']:r['processed_sha256'] for r in fresh_images}
 if old!=new:raise ValueError('Processed input image bytes differ; no inference executed.')
 print('Prepared all117 identical cases and13 identical image packets. No model calls. Inference requires a separate explicit execute command.')
if __name__=='__main__':main()
