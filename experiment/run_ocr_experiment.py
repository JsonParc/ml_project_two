#!/usr/bin/env python3
"""Reproducible TrOCR/Tesseract printed-line recognition experiment.

The local test data are *synthetically generated* and NOT a published benchmark.
Run this script with --engine both after installing requirements and with internet access.
Never fills in missing predictions or fabricates metrics.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pytesseract

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
RESULTS=ROOT/'results'
FONT_REG='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_MONO='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_SERIF='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
FONT_LATO='/usr/share/fonts/truetype/lato/Lato-Regular.ttf'
SAMPLES=[
    {'id':'s01','kind':'영수증 합계','gt':'TOTAL 12.50','font':FONT_BOLD,'size':52},
    {'id':'s02','kind':'잡음 영수증','gt':'CASH CHANGE 7.50','font':FONT_MONO,'size':45,'distort':('noise',25)},
    {'id':'s03','kind':'기울어진 청구서','gt':'Invoice No. 4821','font':FONT_REG,'size':48,'distort':('rotate',7)},
    {'id':'s04','kind':'흐릿한 문장','gt':'Reading text from pixels','font':FONT_SERIF,'size':42,'distort':('blur',2.2)},
    {'id':'s05','kind':'잡음 안내문','gt':'PLEASE KEEP YOUR RECEIPT','font':FONT_LATO,'size':44,'distort':('noise',45)},
    {'id':'s06','kind':'회전 상품 번호','gt':'ORDER 2026-11-25','font':FONT_MONO,'size':45,'distort':('rotate',15)},
    {'id':'s07','kind':'흐릿한 혼합문','gt':'The Quick Brown Fox','font':FONT_REG,'size':48,'distort':('blur',3.2)},
    {'id':'s08','kind':'강한 잡음','gt':'DATA MATRIX 3.14','font':FONT_BOLD,'size':48,'distort':('noise',60)},
]
DEG_SPEC={
  'blur':{'name':'흐림','unit':'px','levels':[0,1,2,3,4]},
  'rotate':{'name':'회전','unit':'°','levels':[0,5,10,15,20]},
  'noise':{'name':'잡음','unit':'σ','levels':[0,15,30,45,60]},
}

def mkfont(path:str,size:int):
    if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.truetype('DejaVuSans.ttf',size)

def draw_line(text:str,font:str,size:int,seed:int=0):
    ft=mkfont(font,size)
    bbox=ft.getbbox(text)
    w=max(500, bbox[2]-bbox[0]+96)
    h=116
    out=Image.new('RGB',(w,h),'white')
    d=ImageDraw.Draw(out)
    d.text((48, (h-(bbox[3]-bbox[1]))/2-bbox[1]),text,font=ft,fill=(22,22,22))
    return out

def make_degraded(img, kind, amount, seed=20261125):
    if kind=='blur': return img.filter(ImageFilter.GaussianBlur(float(amount)))
    if kind=='rotate':return img.rotate(float(amount),resample=Image.Resampling.BICUBIC,expand=True,fillcolor='white')
    if kind=='noise':
        rng=np.random.default_rng(seed)
        a=np.array(img,dtype=np.float32)
        n=rng.normal(0,float(amount),(a.shape[0],a.shape[1],1))
        return Image.fromarray(np.clip(a+n,0,255).astype(np.uint8),'RGB')
    return img

def levenshtein(s,t):
    prev=list(range(len(t)+1))
    for i,c in enumerate(s,1):
        curr=[i]+[0]*len(t)
        for j,d in enumerate(t,1):
            curr[j]=min(curr[j-1]+1,prev[j]+1,prev[j-1]+(c!=d))
        prev=curr
    return prev[-1]

def cer(gt,pred):return levenshtein(gt,pred)/max(1,len(gt))
def wer(gt,pred):return levenshtein(gt.split(),pred.split())/max(1,len(gt.split()))

def setup_images():
    ASSETS.mkdir(parents=True,exist_ok=True)
    base=[]
    for row in SAMPLES:
        path=ASSETS/(row['id']+'.png')
        if not path.exists():
            im=draw_line(row['gt'],row['font'],row['size'])
            if 'distort' in row:
                kind,amount=row['distort'];im=make_degraded(im,kind,amount)
            im.save(path)
        base.append({**{k:v for k,v in row.items() if k not in ('font','size')},'image':str(path.relative_to(ROOT))})
    src=Image.open(ASSETS/'s01.png').convert('RGB')
    degraded=[]
    for kind,spec in DEG_SPEC.items():
        for index,amount in enumerate(spec['levels']):
            path=ASSETS/f'degrade_{kind}_{index}.png'
            if not path.exists():make_degraded(src,kind,amount).save(path)
            degraded.append({'kind':kind,'level':amount,'index':index,'gt':SAMPLES[0]['gt'],'image':str(path.relative_to(ROOT))})
    return base,degraded

def clean_pred(s):
    # Preserve case, punctuation, and internal whitespace for reproducible CER/WER.
    # Remove only OCR output's terminal line endings.
    return s.rstrip('\r\n')

def tesseract_predict(im):
    return clean_pred(pytesseract.image_to_string(im,lang='eng',config='--psm 7'))

def load_trocr(model_id):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    processor=TrOCRProcessor.from_pretrained(model_id)
    model=VisionEncoderDecoderModel.from_pretrained(model_id).eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    def predict(im):
        with torch.no_grad():
            pv=processor(images=im.convert('RGB'),return_tensors='pt').pixel_values.to(device)
            ids=model.generate(pv,max_new_tokens=64,num_beams=1,do_sample=False)
            return clean_pred(processor.batch_decode(ids,skip_special_tokens=True)[0])
    return predict,device

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',choices=['tesseract','both'],default='both')
    ap.add_argument('--model',default='microsoft/trocr-small-printed')
    args=ap.parse_args()
    base,deg=setup_images()
    RESULTS.mkdir(exist_ok=True)
    pred=None;device=None
    if args.engine=='both':
        try: pred,device=load_trocr(args.model)
        except Exception as e:
            msg=f'TrOCR unavailable: {type(e).__name__}: {str(e)[:300]}'
            print(msg)
            (RESULTS/'TROCR_NOT_MEASURED.txt').write_text(msg+'\nNo TrOCR numbers were invented. Rerun on an internet-connected machine.\n',encoding='utf8')
    rows=[]
    for cohort,records in [('samples',base),('degrade',deg)]:
        for row in records:
            im=Image.open(ROOT/row['image']).convert('RGB')
            t0=time.perf_counter();baseline=tesseract_predict(im);dt=time.perf_counter()-t0
            out={**row,'group':cohort,'tesseract_pred':baseline,'tesseract_cer':cer(row['gt'],baseline),'tesseract_wer':wer(row['gt'],baseline),'tesseract_seconds':round(dt,4),'trocr_pred':None,'trocr_cer':None,'trocr_wer':None}
            if pred is not None:
                t0=time.perf_counter();a=pred(im);dt=time.perf_counter()-t0
                out.update(trocr_pred=a,trocr_cer=cer(row['gt'],a),trocr_wer=wer(row['gt'],a),trocr_seconds=round(dt,4))
            rows.append(out)
            print(f"{cohort} {row['image']} | true={row['gt']!r} | Tesseract={baseline!r} ({out['tesseract_cer']:.1%}) | TrOCR={out['trocr_pred']!r}",flush=True)
    fieldnames=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with (RESULTS/'individual_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fieldnames);wr.writeheader();wr.writerows(rows)
    def aggregate(engine,cohort):
        subset=[r for r in rows if r['group']==cohort and r[f'{engine}_pred'] is not None]
        if not subset:return None
        # Corpus CER: total edit distance / total reference chars (not unweighted per-line mean)
        return {'n':len(subset),'cer':sum(levenshtein(r['gt'],r[f'{engine}_pred']) for r in subset)/sum(len(r['gt']) for r in subset),
                'wer':sum(levenshtein(r['gt'].split(),r[f'{engine}_pred'].split()) for r in subset)/sum(len(r['gt'].split()) for r in subset)}
    complete=all(r['trocr_pred'] is not None for r in rows)
    result={'meta':{'date':'2026-11-25','model_id':args.model,'tesseract_version':str(pytesseract.get_tesseract_version()),'device':device,'inference_mode':'num_beams=1 max_new_tokens=64; --psm 7','dataset':'eight synthetic English printed single-line PNGs + fifteen distortions to one base line','trOCR_measured':complete,'synthetic':True,'test_images':len(base),'perturbations':len(deg)},
            'summary':{'samples':{m:aggregate(m,'samples') for m in ('tesseract','trocr')},'degrade':{m:aggregate(m,'degrade') for m in ('tesseract','trocr')}},
            'rows':rows}
    (RESULTS/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    summ=[]
    summ.append('# 자체 OCR 실험 — 실제 실행 결과\n')
    summ.append('입력: 합성 인쇄체 한 줄 이미지 8장, 추가 변형 15장(기준 문장 동일). 이미지 생성에 정답 레이블은 사용했지만 TrOCR/Tesseract 학습은 하지 않았음.\n')
    summ.append('TrOCR는 사전학습된 microsoft/trocr-small-printed, greedy(num_beams=1), Tesseract는 --psm 7. TrOCR 성능은 '+('측정 완료.' if complete else '**미측정(가중치 다운로드/실행 불가)**; 비어 있는 값은 0%가 아님.')+'\n')
    summ.append('CER은 편집거리 합 / 정답 문자 수 합. 소문자화나 구두점 제거 없이 정확히 비교.\n')
    summ.append('| 모델 | 샘플 n | Corpus CER | Corpus WER |\n|---|---:|---:|---:|')
    for m in ('tesseract','trocr'):
        a=result['summary']['samples'][m]
        summ.append(f'| {m} | {a["n"] if a else "미측정"} | {a["cer"]:.1%} | {a["wer"]:.1%} |' if a else f'| {m} | - | 미측정 | 미측정 |')
    summ.append('\n## 품질 변화: Tesseract CER (TrOCR 미측정일 수 있음)\n')
    for kind in DEG_SPEC:
        summ.append(f'- {kind}: '+', '.join(f"{r['level']}={r['tesseract_cer']:.1%}" for r in rows if r['group']=='degrade' and r['kind']==kind))
    summ.append('\n**해석 한계:** n=8 합성 인쇄체, 동일 글자에서 변형을 15개 생성했으므로 독립 표본 23개가 아니다. 일반적인 OCR 성능 우위를 주장할 수 없다.\n')
    (RESULTS/'summary.md').write_text('\n'.join(summ),encoding='utf-8')
    print('\n'.join(summ))
if __name__=='__main__':main()
