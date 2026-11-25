# AI는 어떻게 사진 속 글자를 읽을까? — TrOCR와 사전학습 기반 문자 인식

머신러닝 수업 논문 발표 자료.
Li et al., "TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models", *AAAI* 2023 · 공식 코드 [microsoft/unilm/trocr](https://github.com/microsoft/unilm/tree/master/trocr) · 공개 모델 [microsoft/trocr-small-printed](https://huggingface.co/microsoft/trocr-small-printed)

- **발표 사이트**: `site/index.html` (파일 하나, 17장. → 키를 누를 때마다 장 안의 단계가 하나씩 나오고 마지막 단계 뒤에 다음 장으로, T 키로 밝게/어둡게, F 키로 전체 화면, 오른쪽 아래 "대본 N번"은 대본 사이트의 같은 구간으로 이동)
- **발표 대본**: docs/발표대본.txt (슬라이드별 대본, 논문 실험 상세, 예상 질문)
- **용어 정리**: docs/용어정리.txt
- **대본 사이트**: `python scripts/build_script_site.py` → `script_site/index.html` (별도 저장소 [JsonParc/ml_scrpit_two](https://github.com/JsonParc/ml_scrpit_two)로 올림)

## 슬라이드 구성

| 부 | 슬라이드 |
|---|---|
| 배경 | 2 글자를 읽는 AI · 3 논문 정보와 연구 질문 |
| 독창성 | 4 기존 OCR · 5 TrOCR의 독창성 (Table 1) |
| 모델 | 6 전체 구조 · 7 Encoder (패치 분할 애니메이션) · 8 Decoder (토큰 생성 애니메이션) |
| 학습 | 9 사전학습과 Fine-tuning · 10 손실함수 (슬라이더) |
| 평가·결과 | 11 CER·WER (직접 입력해 계산) · 12 원논문 결과 (Table 2·3·4) |
| 자체 실험 | 13 TrOCR vs Tesseract · 14 이미지 품질 변화 (흐림·회전·잡음 슬라이더) |
| 한계 | 15 한계와 AI 치팅과의 관계 |

7·8·10번의 확률, 토큰 분할, Attention 연결선은 설명용 개념도이고 화면에 그렇게 표시한다.

## 자체 실험 결과 넣기

`site/index.html` 위쪽의 `window.EXP` 객체만 고친다.

- `samples`: 이미지별 정답(`gt`), TrOCR 결과(`trocr`), 비교 모델 결과(`base`). CER/WER은 자동 계산. 실제 사진은 `img: "img/s1.png"`처럼 넣고 `site/img/`에 둔다
- `degrade`: 흐림·회전·잡음 단계별 TrOCR 인식 결과(`pred`)
- 다 채우면 `placeholder: false` → "예시값" 표시가 사라진다

## 대본 고치기

docs/발표대본.txt, docs/용어정리.txt를 고친 뒤

```bash
python scripts/build_script_site.py
cd script_site && git add -A && git commit -m "대본 수정" && git push
```

## 배포 (Render Static Site)

- Root Directory: `site`
- Build Command: `echo "no build"` (또는 비움)
- Publish Directory: `.`

Blueprint로 만들 때는 `render.yaml`에 같은 설정이 들어 있다. 공개되는 것은 `site/` 폴더뿐이다. `site/index.html`은 파일을 더블클릭해도 열린다(인터넷이 없으면 글꼴만 기본 글꼴로 바뀜).

배포 주소가 정해지면 두 곳을 맞춘다.
- `site/index.html`의 `SCRIPT_URL` (대본 사이트 주소)
- `scripts/build_script_site.py`의 `SLIDES_URL` (발표 사이트 주소)
