"""
OCV 저전압 불량 분류 대시보드
- 엑셀 업로드 → 변수 선택 → 머신러닝 → 결과/그래프
- 핵심: OCV1~3(사전예측) vs OCV1~4(사후) 비교
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ────────────────────────────────────────────────
# 한글 폰트 (배포 환경에 없을 수 있어 안전하게 처리)
# ────────────────────────────────────────────────
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
        rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="OCV 불량 분류 대시보드", layout="wide")

# ────────────────────────────────────────────────
# 헤더
# ────────────────────────────────────────────────
st.title("OCV 저전압 불량 분류 대시보드")
st.caption("엑셀을 올리면 OCV 측정값으로 OK/NG를 분류합니다. "
           "OCV1~3(사전 예측)과 OCV1~4(측정 후)를 비교해 "
           "'언제부터 불량을 잡을 수 있는지'를 보여줍니다.")

# ────────────────────────────────────────────────
# 1. 데이터 업로드
# ────────────────────────────────────────────────
st.header("1. 데이터 업로드")
up = st.file_uploader("OCV 엑셀 파일 (.xlsx)", type=["xlsx"])

st.markdown(
    "헤더가 2번째 행에 있는 경우 아래에서 조정하세요. "
    "(예: 1행이 병합셀이고 2행이 '생산일자, 제품명, OCV1...'인 경우 header=1)"
)
header_row = st.number_input("헤더 행 번호 (0부터 시작)", min_value=0, max_value=10, value=1)

if up is None:
    st.info("엑셀을 업로드하면 분석이 시작됩니다.")
    st.stop()

try:
    df = pd.read_excel(up, header=int(header_row))
except Exception as e:
    st.error(f"파일을 읽지 못했어요: {e}")
    st.stop()

st.success(f"불러왔어요 — {len(df):,}행 × {df.shape[1]}열")
with st.expander("데이터 미리보기"):
    st.dataframe(df.head(20), use_container_width=True)

# ────────────────────────────────────────────────
# 2. 컬럼 지정
# ────────────────────────────────────────────────
st.header("2. 컬럼 지정")
cols = df.columns.tolist()

def guess(name_options, fallback_idx):
    for i, c in enumerate(cols):
        if str(c).strip().upper() in name_options:
            return i
    return min(fallback_idx, len(cols) - 1)

c1, c2, c3 = st.columns(3)
with c1:
    ocv1 = st.selectbox("OCV1", cols, index=guess({"OCV1"}, 2))
    ocv2 = st.selectbox("OCV2", cols, index=guess({"OCV2"}, 3))
with c2:
    ocv3 = st.selectbox("OCV3", cols, index=guess({"OCV3"}, 4))
    ocv4 = st.selectbox("OCV4", cols, index=guess({"OCV4"}, 5))
with c3:
    label_col = st.selectbox("판정 컬럼 (OK/NG)", cols, index=guess({"OK_NG", "OKNG", "판정"}, 6))

# 라벨을 0/1로
raw = df[label_col].astype(str).str.upper().str.strip()
is_ng = (raw == "NG").astype(int)
n_ng, n_ok = int(is_ng.sum()), int((is_ng == 0).sum())

if n_ng == 0:
    st.error("NG로 인식된 행이 없어요. 판정 컬럼이 맞는지 확인해주세요 "
             "(값이 'NG'/'OK' 형태여야 해요).")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("전체", f"{len(df):,}")
m2.metric("불량(NG)", f"{n_ng:,}")
m3.metric("불량 비율", f"{n_ng/len(df)*100:.2f}%")

# ────────────────────────────────────────────────
# 3. 분석 조건
# ────────────────────────────────────────────────
st.header("3. 분석 조건")
cc1, cc2 = st.columns(2)
with cc1:
    scenario = st.radio(
        "입력 변수 (언제 시점까지 볼까요?)",
        ["OCV1~3 (사전 예측 — 3번 검사)", "OCV1~4 (측정 후 — 4번 검사)"],
        help="OCV1~3은 OCV4를 측정하기 전에 미리 거를 수 있는지를 봅니다."
    )
with cc2:
    test_size = st.slider("테스트셋 비율", 0.1, 0.4, 0.2, 0.05)
    use_weight = st.checkbox("불균형 보정 (불량에 가중치)", value=True,
                             help="불량이 드물어서 모델이 무시하지 않게 가중치를 줍니다.")

run = st.button("머신러닝 실행", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# 학습 함수
# ────────────────────────────────────────────────
def build_features(d, use_ocv4):
    f = pd.DataFrame(index=d.index)
    f["OCV1"] = d[ocv1]; f["OCV2"] = d[ocv2]; f["OCV3"] = d[ocv3]
    f["dV_12"] = d[ocv1] - d[ocv2]
    f["dV_23"] = d[ocv2] - d[ocv3]
    f["accel"] = f["dV_23"] - f["dV_12"]
    f["std_123"] = d[[ocv1, ocv2, ocv3]].std(axis=1)
    if use_ocv4:
        f["OCV4"] = d[ocv4]
        f["dV_34"] = d[ocv3] - d[ocv4]
        f["std_1234"] = d[[ocv1, ocv2, ocv3, ocv4]].std(axis=1)
    return f

@st.cache_data(show_spinner=False)
def train(d, y, use_ocv4, test_size, use_weight):
    X = build_features(d, use_ocv4)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    sw = None
    if use_weight:
        n0, n1 = (ytr == 0).sum(), (ytr == 1).sum()
        sw = np.where(ytr == 1, n0 / max(n1, 1), 1.0)
    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(Xtr, ytr, sample_weight=sw)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return Xte, yte, pred, proba, imp

if not run:
    st.stop()

use_ocv4 = scenario.startswith("OCV1~4")
data = df[[ocv1, ocv2, ocv3, ocv4]].copy()
Xte, yte, pred, proba, imp = train(data, is_ng, use_ocv4, test_size, use_weight)

# ────────────────────────────────────────────────
# 4. 결과
# ────────────────────────────────────────────────
st.header("4. 결과")
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
recall = tp / (tp + fn) if (tp + fn) else 0
precision = tp / (tp + fp) if (tp + fp) else 0

r1, r2, r3, r4 = st.columns(4)
r1.metric("오검 (불량 놓침)", f"{fn}", help="실제 불량인데 양품으로 통과 — 가장 위험")
r2.metric("과검 (양품 버림)", f"{fp}", help="멀쩡한 셀을 불량으로 폐기 — 수율 손실")
r3.metric("불량 검출률 (recall)", f"{recall:.3f}")
r4.metric("정밀도 (precision)", f"{precision:.3f}")

if use_ocv4:
    st.success("OCV4까지 본 결과입니다. 보통 검출률이 크게 오릅니다 "
               "(급락형 불량이 OCV4에서 드러나기 때문).")
else:
    st.warning("OCV1~3만 본 결과입니다. OCV4에서 갑자기 떨어지는 '급락형' 불량은 "
               "이 시점에선 잡히지 않아 검출률이 낮을 수 있어요.")

# 혼동행렬 + 변수중요도
g1, g2 = st.columns(2)

with g1:
    st.subheader("혼동 행렬")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                fontsize=15, color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["예측 OK", "예측 NG"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실제 OK", "실제 NG"])
    fig.tight_layout()
    st.pyplot(fig)

with g2:
    st.subheader("변수 중요도")
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    imp_sorted = imp.sort_values()
    ax2.barh(imp_sorted.index, imp_sorted.values, color="#2563eb")
    ax2.set_xlabel("중요도")
    fig2.tight_layout()
    st.pyplot(fig2)

# OCV3 vs OCV4 산점도
st.subheader("OCV3 vs OCV4 — OK/NG 분포")
fig3, ax3 = plt.subplots(figsize=(6.5, 5.5))
okm = is_ng == 0
ax3.scatter(df.loc[okm, ocv3], df.loc[okm, ocv4],
            s=6, alpha=0.2, c="#3b82f6", label="OK")
ax3.scatter(df.loc[~okm, ocv3], df.loc[~okm, ocv4],
            s=22, alpha=0.8, c="#ef4444", label="NG")
lo = float(min(df[ocv3].min(), df[ocv4].min()))
hi = float(max(df[ocv3].max(), df[ocv4].max()))
ax3.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="OCV3=OCV4")
ax3.set_xlabel("OCV3"); ax3.set_ylabel("OCV4")
ax3.legend()
fig3.tight_layout()
st.pyplot(fig3)
st.caption("정상(파랑)은 OCV3·OCV4가 모두 높은 영역에 모여 있고, "
           "불량(빨강)은 그 바깥과, 박스 안에도 섞여 있습니다. "
           "박스 안에 섞인 불량이 단순 임계값으로는 안 잡히는, 머신러닝이 필요한 케이스입니다.")

# 임계값 트레이드오프
st.subheader("임계값에 따른 오검 ↔ 과검")
rows = []
for t in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
    pn = (proba >= t)
    tn2, fp2, fn2, tp2 = confusion_matrix(yte, pn).ravel()
    rows.append({
        "임계값": t, "오검(FN)": int(fn2), "과검(FP)": int(fp2),
        "recall": round(tp2 / (tp2 + fn2), 3) if (tp2 + fn2) else 0,
        "precision": round(tp2 / (tp2 + fp2), 3) if (tp2 + fp2) else 0,
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("임계값을 낮추면 불량을 더 적극적으로 잡지만(오검↓) 멀쩡한 셀도 더 버립니다(과검↑). "
           "공정에서 무엇이 더 비싼지에 따라 운영점을 정하세요.")

with st.expander("상세 분류 리포트"):
    st.text(classification_report(yte, pred, target_names=["OK", "NG"]))
