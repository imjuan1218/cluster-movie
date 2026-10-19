import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 설정 및 브라우저 탭 제목/아이콘 설정
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

# 헤더 표시
st.title("🎬 영화 유형 나누기 (K-Means 군집화)")

# 2. 데이터 로드 및 전처리 함수
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_and_preprocess_data():
    # 데이터 읽기
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 전체 원래 편수
    total_raw_count = len(df)

    # 파생 변수 및 파생 속성 생성 전 결측치/0 처리 대상 필터링
    # 롱런 지수 = 누적 관객(total_audi) / 첫 주 관객(first_week_audi)
    # first_week_audi가 0이거나 NaN인 경우 제외
    valid_mask = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["days_in_top10"].notna()
        & df["first_week_audi"].notna()
        & (df["first_week_audi"] > 0)
    )

    df_clean = df[valid_mask].copy()

    # 속성 생성 및 변환
    # 1) 스크린 수 상용로그 (log10)
    df_clean["스크린 수(log10)"] = np.log10(df_clean["first_scrn"])

    # 2) 누적 관객 상용로그 (log10)
    df_clean["누적 관객(log10)"] = np.log10(df_clean["total_audi"])

    # 3) 10위권 일수
    df_clean["10위권 일수"] = df_clean["days_in_top10"]

    # 4) 롱런 지수 (20을 넘으면 20으로 클리핑)
    long_run = df_clean["total_audi"] / df_clean["first_week_audi"]
    df_clean["롱런 지수"] = np.minimum(long_run, 20.0)

    used_count = len(df_clean)

    return df_clean, total_raw_count, used_count


df_data, total_count, used_count = load_and_preprocess_data()

# 전체 편수와 묶은 편수를 화면에 한 줄로 표시
st.write(
    f"**전체 데이터 수:** {total_count:,}편 | **분석에 사용된 영화 수:** {used_count:,}편"
)
st.markdown("---")

# 3. 속성 선택
feature_options = [
    "스크린 수(log10)",
    "누적 관객(log10)",
    "10위권 일수",
    "롱런 지수",
]

st.subheader("1. 군집화에 사용할 속성 선택")
selected_features = st.multiselect(
    "속성을 선택하세요 (2개 이상 선택 필수):",
    options=feature_options,
    default=feature_options,
)

if len(selected_features) < 2:
    st.error("군집화를 진행하려면 속성을 최소 2개 이상 선택해야 합니다.")
    st.stop()

# 4. 데이터 표준화 및 K-Means 군집화
X = df_data[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K=3 난수 고정
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_data["raw_cluster"] = cluster_labels

# 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 레이블 부여
cluster_audi_mean = (
    df_data.groupby("raw_cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
)
rank_map = {
    old_cluster: new_symbol
    for old_cluster, new_symbol in zip(cluster_audi_mean.index, ["㉮", "㉯", "㉰"])
}

df_data["묶음"] = df_data["raw_cluster"].map(rank_map)

# 범주 정렬을 위해 카테고리형으로 설정
category_order = ["㉮", "㉯", "㉰"]
df_data["묶음"] = pd.Categorical(
    df_data["묶음"], categories=category_order, ordered=True
)

st.markdown("---")

# 5. 2차원 산점도
st.subheader("2. 2차원 산점도")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_axis_2d = st.selectbox(
        "2차원 X축 속성 선택:", selected_features, index=0, key="2d_x"
    )
with col2d_2:
    y_axis_2d = st.selectbox(
        "2차원 Y축 속성 선택:",
        selected_features,
        index=1 if len(selected_features) > 1 else 0,
        key="2d_y",
    )

fig_2d = px.scatter(
    df_data,
    x=x_axis_2d,
    y=y_axis_2d,
    color="묶음",
    hover_name="movieNm",
    category_orders={"묶음": category_order},
    title=f"2차원 산점도 ({x_axis_2d} vs {y_axis_2d})",
    color_discrete_sequence=px.colors.qualitative.Set1,
)
fig_2d.update_traces(marker=dict(size=6))
st.plotly_chart(fig_2d, use_container_width=True)

# 6. 3차원 산점도
st.subheader("3. 3차원 산점도")
if len(selected_features) >= 3:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_axis_3d = st.selectbox(
            "3차원 X축 속성 선택:", selected_features, index=0, key="3d_x"
        )
    with col3d_2:
        y_axis_3d = st.selectbox(
            "3차원 Y축 속성 선택:", selected_features, index=1, key="3d_y"
        )
    with col3d_3:
        z_axis_3d = st.selectbox(
            "3차원 Z축 속성 선택:", selected_features, index=2, key="3d_z"
        )

    fig_3d = px.scatter_3d(
        df_data,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color="묶음",
        hover_name="movieNm",
        category_orders={"묶음": category_order},
        title=f"3차원 산점도 ({x_axis_3d} vs {y_axis_3d} vs {z_axis_3d})",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_3d.update_traces(
        marker=dict(size=3)
    )  # 요구사항: 마우스 조작 가능, 작은 점크기
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("💡 3차원 산점도를 그리려면 속성을 3개 이상 선택해 주세요.")

st.markdown("---")

# 7. 묶음별 통계 표 (원래 단위 평균)
st.subheader("4. 묶음별 편수 및 속성 평균 (원래 단위)")

# 원래 단위 속성 값 생성
# first_scrn, total_audi, days_in_top10, 롱런 지수
df_data["스크린 수(원래 단위)"] = df_data["first_scrn"]
df_data["누적 관객(원래 단위)"] = df_data["total_audi"]

raw_feature_cols = [
    "스크린 수(원래 단위)",
    "누적 관객(원래 단위)",
    "days_in_top10",
    "롱런 지수",
]

# 묶음별 aggregations
stats_list = []
for label in category_order:
    sub_df = df_data[df_data["묶음"] == label]
    if len(sub_df) > 0:
        row = {
            "묶음": label,
            "편수": len(sub_df),
            "스크린 수 평균": round(sub_df["first_scrn"].mean(), 1),
            "누적 관객 평균": round(sub_df["total_audi"].mean(), 1),
            "10위권 일수 평균": round(sub_df["days_in_top10"].mean(), 1),
            "롱런 지수 평균": round(sub_df["롱런 지수"].mean(), 2),
        }
        stats_list.append(row)

stats_df = pd.DataFrame(stats_list)
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# 8. 묶음별 누적 관객 Top 5 영화
st.subheader("5. 묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)

for idx, label in enumerate(category_order):
    with cols[idx]:
        st.markdown(f"### 묶음 {label}")
        sub_df = df_data[df_data["묶음"] == label]
        top5 = sub_df.sort_values(by="total_audi", ascending=False).head(5)

        top5_display = top5[["movieNm", "total_audi"]].copy()
        top5_display["total_audi"] = top5_display["total_audi"].apply(
            lambda x: f"{x:,.0f}명"
        )
        top5_display.columns = ["영화 제목", "누적 관객수"]
        top5_display.index = range(1, len(top5_display) + 1)

        st.table(top5_display)


# 1. 앱 제목 및 브라우저 탭 설정 (아이콘 포함)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")


# 2. 데이터 전처리 및 롱런 지수(20 클리핑) 적용
df['log_first_scrn'] = np.log10(df['first_scrn'].astype(float))
df['log_total_audi'] = np.log10(df['total_audi'].astype(float))
df['days_in_top10'] = df['days_in_top10'].astype(float)
df['long_run_idx'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20.0)


# 3. K-Means 난수 고정 및 누적 관객 평균 기준 ㉮·㉯·㉰ 매핑
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
rank_to_label = {raw_cluster: label for raw_cluster, label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['cluster_raw'].map(rank_to_label)


# 4. 고른 속성이 3개 미만일 때 3차원 산점도 예외 처리 안내 문구
if len(selected_feature_labels) < 3:
    st.info("ℹ️ 고른 속성이 셋보다 적어 3차원 산점도를 그릴 수 없습니다. 속성을 3개 이상 선택해 주세요.")


# 5. 원래 단위 기준의 속성 평균 표 생성
summary_list = []
for c in ['㉮', '㉯', '㉰']:
    c_df = df[df['cluster'] == c]
    summary_list.append({
        '묶음': c,
        '편수': len(c_df),
        '스크린 수 평균': round(c_df['first_scrn'].mean(), 2),
        '누적 관객 평균': round(c_df['total_audi'].mean(), 2),
        '10위권 일수 평균': round(c_df['days_in_top10'].mean(), 2),
        '롱런 지수 평균': round(c_df['long_run_idx'].mean(), 2)
    })
st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
