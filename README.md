# AI는 어떻게 사진 속 글자를 읽을까? — TrOCR와 사전학습 기반 문자 인식

머신러닝 수업 논문 발표 자료.
Li et al., "TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models", *AAAI* 2023 · 공식 코드 [microsoft/unilm/trocr](https://github.com/microsoft/unilm/tree/master/trocr) · 공개 모델 [microsoft/trocr-small-printed](https://huggingface.co/microsoft/trocr-small-printed)

- **발표 사이트**: `site/index.html` (파일 하나, 본편 21장 + 부록 2장. → 키를 누를 때마다 장 안의 단계가 하나씩 나오고 마지막 단계 뒤에 다음 장으로, T 키로 밝게/어둡게, F 키로 전체 화면, 오른쪽 아래 "대본 N번"은 대본 사이트의 같은 구간으로 이동)
- **발표 대본**: docs/발표대본.txt (슬라이드별 대본, 논문 실험 상세, 예상 질문)
- **용어 정리**: docs/용어정리.txt
- **대본 사이트**: `python scripts/build_script_site.py` → `script_site/index.html` (별도 저장소 [JsonParc/ml_scrpit_two](https://github.com/JsonParc/ml_scrpit_two)로 올림)

## 슬라이드 구성

초파리 발표(JsonParc/ML_project)와 같은 6부 순서. 제목은 주제명, 본문은 개조식.

| 부 | 슬라이드 |
|---|---|
| 개요 | 1 표지 · 2 논문 요약 |
| 1 선택 이유 | 3 논문을 고른 이유 |
| 2 독창성 | 4 배경 · 5 기존 방법과 비교 · 6 독창성 |
| 3 모델 | 7 TrOCR 구조 · 8 인코더 (패치 분할 애니메이션) · 9 디코더 (토큰 생성 애니메이션) · 10 이 모델을 사용한 이유 |
| 4 파이프라인 | 11 학습 파이프라인 · 12 사전학습 데이터와 데이터 증강 · 13 손실함수 |
| 5 결과 | 14 평가 방법 (CER·WER 계산) · 15 사전학습과 데이터 증강의 효과 · 16 벤치마크 결과 |
| 6 한계와 극복 아이디어 | 17 논문의 한계 · 18 이미지 품질에 따른 인식 변화 (자체 실험) |
| 마무리 | 19 공개 모델로 직접 비교 · 20 정리 · 21 감사합니다 · 부록 A, B |

8·9·13번의 확률, 토큰 분할, Attention 연결선은 설명용 개념도이고 화면에 그렇게 표시한다.

## 자체 실험 결과 넣기

`site/index.html` 위쪽의 `window.EXP` 객체만 고친다.

- `samples`: 이미지별 정답(`gt`), Tesseract 결과(`tess`), TrOCR 결과(`trocr`). `null`은 미측정, `""`는 아무것도 읽지 못함(CER 100%). CER/WER은 자동 계산
- `degrade`: 흐림·회전·잡음 단계별 인식 결과
- 실험 이미지는 `site/img/s01.png`…`s08.png`, `site/img/degrade_{blur,rotate,noise}_{0..4}.png`. 파일이 없으면 같은 조건을 브라우저에서 그린 이미지로 대신 보여준다

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
