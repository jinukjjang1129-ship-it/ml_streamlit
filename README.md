# OCV 저전압 불량 분류 대시보드

엑셀(OCV 측정 데이터)을 올리면 OK/NG를 분류하는 Streamlit 앱입니다.
**OCV1~3(사전 예측)** 과 **OCV1~4(측정 후)** 를 버튼으로 비교해,
불량을 언제부터 잡을 수 있는지 보여줍니다.

## 기능
- 엑셀 업로드 (헤더 행 조정 가능)
- 컬럼 자동 인식 + 수동 지정
- 입력 변수 시나리오 토글: OCV1~3 vs OCV1~4
- 결과: 오검/과검/recall/precision, 혼동행렬, 변수 중요도
- OCV3 vs OCV4 산점도
- 임계값에 따른 오검↔과검 트레이드오프 표

## 로컬에서 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
브라우저에서 http://localhost:8501 로 열립니다. (본인 PC에서만 보임)

## 인터넷 링크로 공유하기 (Streamlit Community Cloud, 무료)

1. 이 파일들(`app.py`, `requirements.txt`, `packages.txt`, `README.md`)을
   GitHub 저장소(repository)에 올립니다.
2. https://share.streamlit.io 에 접속해 GitHub 계정으로 로그인합니다.
3. **"New app"** → 저장소(repo)와 브랜치 선택, Main file path에 `app.py` 입력.
4. **Deploy** 클릭. 잠시 후 `https://<앱이름>.streamlit.app` 링크가 생깁니다.
5. 이 링크를 카톡으로 보내면 누구나 브라우저로 열어 자기 엑셀을 올려볼 수 있습니다.

## 참고
- 모델은 sklearn `GradientBoostingClassifier`를 사용합니다
  (xgboost와 성능이 비슷하면서 배포가 더 안정적입니다).
- 한글 폰트는 `packages.txt`의 `fonts-nanum`으로 클라우드에 자동 설치됩니다.
- 엑셀 헤더가 2번째 행에 있으면 앱에서 "헤더 행 번호"를 1로 두세요.
