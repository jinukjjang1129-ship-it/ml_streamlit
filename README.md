# OCV 분류 Streamlit 실습 앱

Excel로 입력한 OCV 측정값으로 분류 모델을 학습하고, 임곗값에 따른 미검출·과검을 살펴보는 실습용 대시보드입니다.

## 기능

- Excel 헤더·입력 컬럼 선택
- OCV1~3 입력과 OCV1~4 입력 비교
- GradientBoostingClassifier 학습과 Precision·Recall·혼동행렬 표시
- 변수 중요도, OCV3/OCV4 산점도, 임곗값별 결과 확인

OCV4가 정답 판정 기준인 데이터에서는 OCV4 포함 결과를 **사전 예측 성능으로 사용할 수 없습니다.** 이 저장소는 입력·시각화·임곗값 실습을 보존하는 용도입니다. 다른 모델과의 우열이나 배포 안정성은 별도 비교 실험 없이 주장하지 않습니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

사용 권한이 있는 Excel을 준비하고 실제 헤더 위치를 지정합니다. 원문 제조 데이터나 업로드 파일을 저장소에 공개하지 않습니다. 공개 서비스에 비공개 업무 데이터를 업로드하는 방식은 이 프로젝트의 제출 범위에 포함하지 않습니다.

## 검증용 프로젝트와 관계

생산일/그룹 분할, 학습 내부 전처리, 4개 모델 비교, 평가셋과 검증셋 분리는 [OCV 검증 변경](https://github.com/jinukjjang1129-ship-it/ocv__/pull/1)에서 별도로 다룹니다. 제출용 `battery-ocv-classification`은 해당 검증 코드를 선별한 신규 저장소로 준비 중입니다.

기존 업무 성과의 원자료는 비공개이며, 제공된 강의용 샘플과 같은 데이터인지 확인되지 않아 직접 재현 검증은 보류합니다. 이번 README 정리를 새로운 모델 성능 검증으로 표시하지 않습니다.
