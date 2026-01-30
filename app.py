
import streamlit as st

# 위치 고정!
st.set_page_config(
    page_title="영양정보 통합 트래커 (Upstage IE)",
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import base64
import json
import re
from PIL import Image
from openai import OpenAI

# =========================================
# 0) Upstage Information Extract 설정
# =========================================
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")  # export UPSTAGE_API_KEY="up_..."
IE_BASE_URL = "https://api.upstage.ai/v1/information-extraction"

ie_client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url=IE_BASE_URL
)

def encode_bytes_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")

def _safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        # "12g", "120kcal" 같은 문자열도 대비
        if isinstance(x, str):
            x = re.sub(r"[^\d\.]", "", x)
        return float(x)
    except Exception:
        return default

def extract_nutrition_with_ie(image_bytes: bytes) -> dict:
    """
    Upstage Information Extract로 영양성분을 JSON으로 추출
    반환 예:
      {
        "product_name": "...",
        "serving_size": "...",
        "calories_kcal": 120,
        "carbs_g": 15,
        "sugar_g": 5,
        "protein_g": 3,
        "fat_g": 4,
        "cholesterol_mg": 0,
        "sodium_mg": 200
      }
    """
    if not UPSTAGE_API_KEY:
        raise RuntimeError("UPSTAGE_API_KEY 환경변수가 설정되어 있지 않습니다.")

    b64 = encode_bytes_to_base64(image_bytes)

    # ✅ 너무 빡빡하게 required 걸면 실패가 늘어서, optional로 두고 후처리에서 0 처리
    schema = {
        "name": "nutrition_label_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "product_name": {"type": "string", "description": "제품명(가능하면)"},
                "serving_size": {"type": "string", "description": "1회 제공량(예: 30g, 1봉지)"},
                "calories_kcal": {"type": "number", "description": "열량(kcal)"},
                "carbs_g": {"type": "number", "description": "탄수화물(g)"},
                "sugar_g": {"type": "number", "description": "당류(g)"},
                "protein_g": {"type": "number", "description": "단백질(g)"},
                "fat_g": {"type": "number", "description": "지방(g)"},
                "cholesterol_mg": {"type": "number", "description": "콜레스테롤(mg)"},
                "sodium_mg": {"type": "number", "description": "나트륨(mg)"},
            }
        }
    }

    resp = ie_client.chat.completions.create(
        model="information-extract",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:application/octet-stream;base64,{b64}"}
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": schema
        }
    )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Information Extract 응답이 비어있습니다.")

    # 혹시 잡텍스트가 섞여도 최대한 JSON만 추출
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start:end+1])
        else:
            raise RuntimeError(f"IE 응답 JSON 파싱 실패:\n{content}")

    return data

# =========================================
# 1) 음식 추천 함수 (유사도 기반)
# =========================================
def recommend_foods(input_nutrition, food_df, top_n=5):
    """
    input_nutrition: {'탄수화물': float, '당류': float, '지방': float, '단백질': float}
    food_df: pandas.DataFrame (필수 컬럼: '음식','탄수화물','당류','단백질','지방')
    """
    required_cols = {"음식","탄수화물","당류","단백질","지방"}
    if not required_cols.issubset(set(food_df.columns)):
        return pd.DataFrame()

    def compute_similarity(row):
        return sum(abs(_safe_float(row[k]) - input_nutrition[k]) for k in input_nutrition)

    tmp = food_df.copy()
    tmp["유사도"] = tmp.apply(compute_similarity, axis=1)
    return tmp.sort_values(by="유사도").head(top_n)

# =========================================
# 2) Streamlit UI 스타일
# =========================================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #198754 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    [data-testid="stSidebar"] button {
        background-color: white !important;
        color: #198754 !important;
        font-weight: bold !important;
    }
    h1, h2, h3 {
        color: #198754 !important;
    }
    .stButton>button {
        background-color: #198754 !important;
        color: white !important;
    }
    .stProgress>div>div {
        background-color: #198754 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# 3) 세션 상태 초기화
# =========================================
if "nutrition_history" not in st.session_state:
    st.session_state.nutrition_history = []

if "daily_total" not in st.session_state:
    st.session_state.daily_total = {
        "칼로리": 0.0,
        "탄수화물": 0.0,
        "단백질": 0.0,
        "지방": 0.0,
        "당류": 0.0,
        "콜레스테롤": 0.0,
        "나트륨": 0.0
    }

if "current_date" not in st.session_state:
    st.session_state.current_date = datetime.date.today()

# =========================================
# 4) 사이드바
# =========================================
with st.sidebar:
    st.markdown("<h3 style='color: white;'>📊 영양 대시보드</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>영양성분표 사진을 올려 자동 추출하세요 (Upstage IE)</p>", unsafe_allow_html=True)

    selected_date = st.date_input("", st.session_state.current_date, label_visibility="collapsed")
    if selected_date != st.session_state.current_date:
        st.session_state.current_date = selected_date
        if selected_date != datetime.date.today():
            st.session_state.daily_total = {k: 0.0 for k in st.session_state.daily_total}
            st.session_state.nutrition_history = []

    st.markdown("<h3 style='color: white;'>⚙️ 일일 목표 설정</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: white; font-weight: bold;'>목표 칼로리 (kcal)</p>", unsafe_allow_html=True)
    daily_calorie_goal = st.number_input("", min_value=0, value=2000, label_visibility="collapsed")

    if st.button("🔄 오늘 초기화"):
        st.session_state.nutrition_history = []
        st.session_state.daily_total = {k: 0.0 for k in st.session_state.daily_total}
        st.success("오늘의 기록이 초기화되었습니다!")

# =========================================
# 5) 요약 패널 함수
# =========================================
def display_nutrition_summary():
    st.markdown("<h2 style='color: #198754;'>📊 오늘의 영양 요약</h2>", unsafe_allow_html=True)

    total_cal = float(np.nan_to_num(st.session_state.daily_total.get("칼로리", 0.0), nan=0.0))
    if total_cal <= 0:
        st.info("아직 오늘 기록된 음식이 없습니다.")
        return

    carbs_val   = float(np.nan_to_num(st.session_state.daily_total.get("탄수화물", 0.0), nan=0.0))
    protein_val = float(np.nan_to_num(st.session_state.daily_total.get("단백질", 0.0), nan=0.0))
    fat_val     = float(np.nan_to_num(st.session_state.daily_total.get("지방", 0.0), nan=0.0))
    sugar_val   = float(np.nan_to_num(st.session_state.daily_total.get("당류", 0.0), nan=0.0))

    values = [carbs_val, protein_val, fat_val, sugar_val]
    labels = ["Carbs", "Protein", "Fat", "Sugar"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white"),
        textprops={"color": "black"}
    )
    ax.axis("equal")
    plt.title("Nutrition Ratio", fontsize=14, fontweight="bold", color="black")
    st.pyplot(fig)

    st.markdown(f"<h3 style='color: #198754;'>칼로리 진행 상황: {total_cal:.1f} / {daily_calorie_goal} kcal</h3>", unsafe_allow_html=True)
    progress = min(total_cal / max(daily_calorie_goal, 1), 1.0)
    st.progress(progress)

    summary_df = pd.DataFrame({
        "영양소": ["칼로리 (kcal)", "탄수화물 (g)", "단백질 (g)", "지방 (g)", "당류 (g)", "콜레스테롤 (mg)", "나트륨 (mg)"],
        "섭취량": [
            f"{total_cal:.1f}",
            f"{carbs_val:.1f}",
            f"{protein_val:.1f}",
            f"{fat_val:.1f}",
            f"{sugar_val:.1f}",
            f"{float(np.nan_to_num(st.session_state.daily_total.get('콜레스테롤', 0.0), nan=0.0)):.1f}",
            f"{float(np.nan_to_num(st.session_state.daily_total.get('나트륨', 0.0), nan=0.0)):.1f}",
        ]
    })
    st.table(summary_df)

# =========================================
# 6) 메인 화면
# =========================================
st.title("🍎 영양정보 통합 트래커 (Upstage Information Extract)")

tab1, tab2 = st.tabs(["📸 사진으로 분석 (IE)", "✏️ 직접 입력"])

# -------------------------------
# 탭 1: 사진 업로드 → Upstage IE → 기록/추천
# -------------------------------
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<h2 style='color: #198754;'>📸 영양성분표(라벨) 사진 업로드</h2>", unsafe_allow_html=True)
        st.caption("✅ '음식 사진'이 아니라, **영양성분표가 보이는 포장지/라벨 사진**을 올려야 정확합니다.")

        uploaded_file = st.file_uploader("PNG/JPG 업로드", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="업로드된 이미지", use_container_width=True)

            food_name = st.text_input("음식 이름(선택)", value="")

            if st.button("🔍 분석 실행", key="btn_ie_run"):
                with st.spinner("Upstage Information Extract로 영양정보 추출 중..."):
                    try:
                        img_bytes = uploaded_file.getvalue()
                        ie = extract_nutrition_with_ie(img_bytes)
                    except Exception as e:
                        st.error(f"IE 호출/파싱 실패: {e}")
                        st.stop()

                # IE 결과 후처리 (없으면 0 처리)
                product_name = ie.get("product_name") or ""
                serving_size = ie.get("serving_size") or ""

                cal = _safe_float(ie.get("calories_kcal"), 0.0)
                carbs = _safe_float(ie.get("carbs_g"), 0.0)
                sugar = _safe_float(ie.get("sugar_g"), 0.0)
                protein = _safe_float(ie.get("protein_g"), 0.0)
                fat = _safe_float(ie.get("fat_g"), 0.0)
                chol = _safe_float(ie.get("cholesterol_mg"), 0.0)
                sodium = _safe_float(ie.get("sodium_mg"), 0.0)

                # 화면 표시용 nutrient_info (기존 UI 유지)
                nutrient_info = [
                    {"칼로리": f"{cal}kcal" if cal else None},
                    {"나트륨": f"{sodium}mg" if sodium else None},
                    {"탄수화물": f"{carbs}g" if carbs else None},
                    {"당류": f"{sugar}g" if sugar else None},
                    {"지방": f"{fat}g" if fat else None},
                    {"콜레스테롤": f"{chol}mg" if chol else None},
                    {"단백질": f"{protein}g" if protein else None},
                ]

                st.success("영양정보 추출 완료!")
                if product_name or serving_size:
                    st.info(f"제품명: {product_name if product_name else '—'} / 1회 제공량: {serving_size if serving_size else '—'}")

                # 추출 결과 표
                display_list = []
                for item in nutrient_info:
                    for k, v in item.items():
                        display_list.append((k, v if v else "–"))
                df_nutr = pd.DataFrame(display_list, columns=["영양소", "값"])
                st.table(df_nutr)

                # 기록 추가
                record = {
                    "음식명": food_name if food_name else (product_name if product_name else "사진 자동분석"),
                    "칼로리": cal,
                    "탄수화물": carbs,
                    "단백질": protein,
                    "지방": fat,
                    "당류": sugar,
                    "콜레스테롤": chol,
                    "나트륨": sodium,
                    "시간": datetime.datetime.now().strftime("%H:%M"),
                    "날짜": st.session_state.current_date.strftime("%Y-%m-%d"),
                    "1회제공량": serving_size
                }
                st.session_state.nutrition_history.append(record)

                # daily_total 업데이트 (record 기준으로만 누적)
                for k in ["칼로리", "탄수화물", "단백질", "지방", "당류", "콜레스테롤", "나트륨"]:
                    st.session_state.daily_total[k] += float(record.get(k, 0.0) or 0.0)

                # 추천(옵션): food_data.csv 있으면
                input_nutri = {"탄수화물": carbs, "당류": sugar, "지방": fat, "단백질": protein}
                food_df = None
                for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
                    try:
                        food_df = pd.read_csv("food_data.csv", encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                    except FileNotFoundError:
                        food_df = None
                        break

                if food_df is None:
                    st.warning("food_data.csv가 없어 추천 기능은 생략됩니다. (앱 폴더에 food_data.csv를 넣으면 추천 활성화)")
                else:
                    rec_df = recommend_foods(input_nutri, food_df, top_n=5)
                    if rec_df is not None and not rec_df.empty:
                        if st.button("🍽️ 음식 추천 보기", key="btn_show_rec"):
                            st.markdown("#### 🍽️ 음식 추천 Top 5")
                            st.table(rec_df[["음식", "탄수화물", "당류", "단백질", "지방"]])
                    else:
                        st.info("추천 데이터(컬럼)가 맞지 않아 추천을 표시할 수 없습니다. food_data.csv 컬럼을 확인하세요.")

    with col2:
        display_nutrition_summary()

# -------------------------------
# 탭 2: 수동 입력
# -------------------------------
with tab2:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<h2 style='color: #198754;'>✏️ 영양정보 직접 입력</h2>", unsafe_allow_html=True)

        with st.form("manual_nutrition_form_tab"):
            manual_food_name = st.text_input("음식 이름", "")

            col_a, col_b = st.columns(2)
            with col_a:
                manual_calories = st.number_input("칼로리 (kcal)", min_value=0.0, format="%.1f")
                manual_carbs    = st.number_input("탄수화물 (g)", min_value=0.0, format="%.1f")
                manual_protein  = st.number_input("단백질 (g)", min_value=0.0, format="%.1f")
                manual_fat      = st.number_input("지방 (g)", min_value=0.0, format="%.1f")
            with col_b:
                manual_sugar       = st.number_input("당류 (g)", min_value=0.0, format="%.1f")
                manual_cholesterol = st.number_input("콜레스테롤 (mg)", min_value=0.0, format="%.1f")
                manual_sodium      = st.number_input("나트륨 (mg)", min_value=0.0, format="%.1f")
                manual_time        = st.time_input("섭취 시간", datetime.datetime.now().time())

            submitted = st.form_submit_button("저장하기")
            if submitted:
                if not manual_food_name:
                    st.error("음식 이름을 입력해주세요!")
                else:
                    rec = {
                        "음식명": manual_food_name,
                        "칼로리": manual_calories,
                        "탄수화물": manual_carbs,
                        "단백질": manual_protein,
                        "지방": manual_fat,
                        "당류": manual_sugar,
                        "콜레스테롤": manual_cholesterol,
                        "나트륨": manual_sodium,
                        "시간": manual_time.strftime("%H:%M"),
                        "날짜": st.session_state.current_date.strftime("%Y-%m-%d"),
                    }
                    st.session_state.nutrition_history.append(rec)
                    for k in ["칼로리","탄수화물","단백질","지방","당류","콜레스테롤","나트륨"]:
                        st.session_state.daily_total[k] += float(rec.get(k, 0.0) or 0.0)
                    st.success(f"{manual_food_name}의 영양정보가 추가되었습니다!")

        st.markdown("<h3 style='color: #198754;'>자주 먹는 음식 바로 추가</h3>", unsafe_allow_html=True)
        common_foods = {
            "사과 1개":      {"칼로리":95,  "탄수화물":25,  "단백질":0.5, "지방":0.3, "당류":19, "콜레스테롤":0,   "나트륨":2},
            "바나나 1개":    {"칼로리":105, "탄수화물":27,  "단백질":1.3, "지방":0.4, "당류":14, "콜레스테롤":0,   "나트륨":1},
            "계란 1개":      {"칼로리":70,  "탄수화물":0.6, "단백질":6.3, "지방":5,   "당류":0.6, "콜레스테롤":186, "나트륨":70},
            "우유 200ml":    {"칼로리":124, "탄수화물":12,  "단백질":6.6, "지방":6.6, "당류":12, "콜레스테롤":24,  "나트륨":100},
            "닭가슴살 100g": {"칼로리":165, "탄수화물":0,   "단백질":31,  "지방":3.6, "당류":0,  "콜레스테롤":85,  "나트륨":74},
        }

        cols = st.columns(3)
        buttons = [
            ("🍎 사과 1개", "사과 1개"),
            ("🍌 바나나 1개", "바나나 1개"),
            ("🥚 계란 1개", "계란 1개"),
            ("🥛 우유 200ml", "우유 200ml"),
            ("🍗 닭가슴살 100g", "닭가슴살 100g"),
        ]
        for i, (label, keyname) in enumerate(buttons):
            with cols[i % 3]:
                if st.button(label, key=f"btn_common_{i}"):
                    fd = common_foods[keyname]
                    rec = {
                        "음식명": keyname,
                        "칼로리": fd["칼로리"],
                        "탄수화물": fd["탄수화물"],
                        "단백질": fd["단백질"],
                        "지방": fd["지방"],
                        "당류": fd["당류"],
                        "콜레스테롤": fd["콜레스테롤"],
                        "나트륨": fd["나트륨"],
                        "시간": datetime.datetime.now().strftime("%H:%M"),
                        "날짜": st.session_state.current_date.strftime("%Y-%m-%d"),
                    }
                    st.session_state.nutrition_history.append(rec)
                    for k in ["칼로리","탄수화물","단백질","지방","당류","콜레스테롤","나트륨"]:
                        st.session_state.daily_total[k] += float(rec.get(k, 0.0) or 0.0)
                    st.success(f"{keyname}가 추가되었습니다!")

    with col2:
        display_nutrition_summary()

# =========================================
# 7) 하단: 오늘의 식단 기록
# =========================================
st.markdown("<h2 style='color: #198754;'>📝 오늘의 식단 기록</h2>", unsafe_allow_html=True)

if st.session_state.nutrition_history:
    history_df = pd.DataFrame(st.session_state.nutrition_history)

    # 표시할 컬럼이 없을 수도 있으니 안전하게
    display_cols = [c for c in ["음식명","시간","칼로리","탄수화물","단백질","지방","당류","나트륨"] if c in history_df.columns]
    formatted_df = history_df[display_cols].copy()

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    csv = formatted_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 식단 기록 다운로드 (CSV)",
        data=csv,
        file_name=f"식단기록_{st.session_state.current_date.strftime('%Y-%m-%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("오늘 기록된 식단이 없습니다.")

# 푸터
st.markdown("---")
st.markdown("""
<div style="background-color: #e8f4ea; padding: 15px; border-radius: 10px; border-left: 5px solid #198754;">
    <h3 style="color: #198754;">💡 팁: 이렇게 업로드하세요!</h3>
    <p style="color: #000000 !important;">
      (1) 영양성분표가 프레임 안에 크게 나오게 촬영<br/>
      (2) 반사광이 없게 조명 각도 조절<br/>
      (3) 흐림/손떨림 방지 (가능하면 정면에서)<br/>
    </p>
</div>
""", unsafe_allow_html=True)
