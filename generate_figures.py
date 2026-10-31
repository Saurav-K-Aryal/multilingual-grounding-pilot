"""Plot the included recorded counts; no external data or model calls."""
from pathlib import Path
import argparse,hashlib,json
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
def font(size):
 for p in ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
  if Path(p).exists():return ImageFont.truetype(p,size)
 return ImageFont.load_default(size=size)
def plot(groups,series,title,subtitle,path):
 image=Image.new('RGB',(1800,850),'white');d=ImageDraw.Draw(image);d.text((60,24),title,font=font(42),fill='#172634');d.text((60,83),subtitle,font=font(27),fill='#52616b')
 left,right,top,bottom=130,1735,190,665
 for tick in [0,3,6,9,12]:
  y=bottom-(bottom-top)*tick/12;d.line((left,y,right,y),fill='#dce3e8',width=2);d.text((left-25,y),str(tick),font=font(28),anchor='rm',fill='#52616b')
 step=(right-left)/len(groups);width=65 if len(groups)==4 else 125;space=12
 for i,(label,values) in enumerate(groups):
  center=left+(i+.5)*step;total=len(series)*width+(len(series)-1)*space
  for j,(name,color) in enumerate(series):
   value=values[j];x=center-total/2+j*(width+space);y=bottom-(bottom-top)*value/12
   d.rectangle((x,y,x+width,bottom),fill=color);d.text((x+width/2,y-35),str(value),font=font(31),anchor='mt',fill='#172634')
  for line_index,line in enumerate(label.split('\n')):d.text((center,bottom+20+line_index*33),line,font=font(28),anchor='mt',fill='#172634')
 total_width=440*len(series);start=(1800-total_width)/2
 for j,(name,color) in enumerate(series):
  x=start+j*440;d.rectangle((x,780,x+30,808),fill=color);d.text((x+45,778),name,font=font(27),fill='#172634')
 image.save(path,dpi=(300,300))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-source-root',type=Path);a=p.parse_args();data=json.loads((HERE/'data/model_results.json').read_text())
 if a.verify_source_root:
  for r in data['provenance']:assert hashlib.sha256((a.verify_source_root/r['source']).read_bytes()).hexdigest()==r['sha256']
 by={r['model']:r['metrics'] for r in data['models']};groups=[]
 for tag,label in [('qwen3.5:4b','Qwen3.5'),('gemma4:26b','Gemma4')]:
  for lang,name in [('yor','Yoruba'),('swa','Swahili')]:
   groups.append((label+'\n'+name,[by[tag]['AfriQA'][lang+'_native_gold']['strict_gold_span_correct'],by[tag]['AfriQA'][lang+'_english_gold']['strict_gold_span_correct']]))
 out=HERE/'figures';out.mkdir(exist_ok=True)
 plot(groups,[('Native question','#276784'),('English question','#bc6b39')],'English source-span exact matches /12','Same passage and English answer contract within each pair; different items across languages',out/'source_extraction.png')
 groups=[]
 for tag,label in [('qwen3.5:4b','Qwen3.5'),('gemma4:26b','Gemma4')]:
  groups.append((label,[by[tag]['Afri-MCQA']['yor_'+condition]['correct'] for condition in ['image_native','image_english','noimage_native']]))
 plot(groups,[('Image + native','#276784'),('Image + English','#bc6b39'),('No image + native','#81908a')],'Yoruba cultural question accuracy /12','Twelve paired image groups; internal research holdout from labeled public development data',out/'cultural_image_use.png')
 print('Generated two recorded-count PNG figures; no model calls.')
if __name__=='__main__':main()
