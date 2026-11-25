"""
TrOCR 실측 (로컬 CPU) — run_ocr_experiment.py와 같은 이미지·같은 추론 설정

- 이미지: experiment/assets/ 의 23장 (s01~s08, degrade_*), Tesseract 실측에 쓴 것과 같은 파일
- 모델: microsoft/trocr-small-printed, 재학습 없음
- 추론: num_beams=1 (탐욕적), max_new_tokens=64, do_sample=False. 출력은 끝 줄바꿈만 제거
- Tesseract 결과는 results/results.json(GPT 쪽 실측)을 그대로 두고, TrOCR 결과만 채워
  results/results_with_trocr.json 으로 저장한다.

실행 (transformers 4.x 필요, 5.x는 이 모델의 토크나이저를 읽지 못함):
  python experiment/run_trocr_local.py [모델 경로 또는 이름]
Windows에서 사용자 폴더 경로에 한글이 있으면 sentencepiece가 캐시 파일을 못 여므로,
모델 폴더를 영문 경로에 복사해 그 경로를 인자로 넘긴다.
"""
import json
import platform
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import transformers

ROOT = Path(__file__).resolve().parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/trocr-small-printed"


def levenshtein(s, t):
    prev = list(range(len(t) + 1))
    for i, c in enumerate(s, 1):
        cur = [i] + [0] * len(t)
        for j, d in enumerate(t, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (c != d))
        prev = cur
    return prev[-1]


def main():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    processor = TrOCRProcessor.from_pretrained(MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL).eval()

    data = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    for r in rows:
        im = Image.open(ROOT / r["image"]).convert("RGB")
        t0 = time.perf_counter()
        with torch.no_grad():
            pv = processor(images=im, return_tensors="pt").pixel_values
            ids = model.generate(pv, max_new_tokens=64, num_beams=1, do_sample=False)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0].rstrip("\r\n")
        dt = time.perf_counter() - t0
        gt = r["gt"]
        r.update(trocr_pred=pred, trocr_cer=levenshtein(gt, pred) / max(1, len(gt)),
                 trocr_wer=levenshtein(gt.split(), pred.split()) / max(1, len(gt.split())),
                 trocr_seconds=round(dt, 4))
        print(f"{r['image']}\tgt={gt!r}\ttesseract={r['tesseract_pred']!r}\ttrocr={pred!r}\tCER={r['trocr_cer']:.1%}", flush=True)

    def agg(engine, group):
        sub = [r for r in rows if r["group"] == group and r.get(f"{engine}_pred") is not None]
        return {"n": len(sub),
                "cer": sum(levenshtein(r["gt"], r[f"{engine}_pred"]) for r in sub) / sum(len(r["gt"]) for r in sub),
                "wer": sum(levenshtein(r["gt"].split(), r[f"{engine}_pred"].split()) for r in sub) / sum(len(r["gt"].split()) for r in sub)}

    data["meta"].update(trOCR_measured=True, trocr_run={
        "where": "local CPU", "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "model": "microsoft/trocr-small-printed",
        "settings": "num_beams=1 max_new_tokens=64 do_sample=False"})
    data["summary"] = {g: {e: agg(e, g) for e in ("tesseract", "trocr")} for g in ("samples", "degrade")}
    (ROOT / "results" / "results_with_trocr.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
