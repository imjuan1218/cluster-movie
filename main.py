import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# 1. 페이지 및 브라우저 탭 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    return pd.read_csv(url, encoding="utf-8")

raw_df = load_data()
total_raw_count = len(raw_df)

# 3. 데이터 전처리 및 변수 생성 (NameError 수정 부분)
# 필수 속성 열 결측치 및 첫 주 관객이 0인 영화를 빼서 df 변수 정의
required_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi', 'movieNm']
df = raw_df.dropna(subset=required_cols).copy()
df = df[df['first_week_audi'] > 0].copy()

# 파생 속성 생성 (상용로그 및 롱런 지수 20 클리핑)
df['log_first_scrn'] = np.log10(df['first_scrn'].astype(float))
df['log_total_audi'] = np.log10(df['total_audi'].astype(float))
df['days_in_top10'] = df['days_in_top10'].astype(float)
df['long_run_idx'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20.0)

used_count = len(df)

# 전체 편수와 묶은 편수 한 줄 표시
st.markdown(f"**전체 편수:** {total_raw_count}편 | **묶은 편수:** {used_count}편")
st.divider()

# 4. 속성 선택 UI (기본값: 4개 모두 선택)
feature_options = {
    '스크린 수 (로그)': 'log_first_scrn',
    '누적 관객 (로그)': 'log_total_audi',
    '10위권 일수': 'days_in_top10',
    '롱런 지수': 'long_run_idx'
}

selected_feature_labels = st.multiselect(
    "묶는 데 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=list(feature_options.keys()),
    default=list(feature_options.keys())
)

if len(selected_feature_labels) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_options[label] for label in selected_feature_labels]

# 5. 선택된 속성 표준화 및 K-Means 군집화 (k=3, 난수 고정)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 6. 누적 관객 평균이 큰 순서대로 ㉮·㉯·㉰ 라벨 부여
cluster_audi_mean = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
rank_to_label = {raw_cluster: label for raw_cluster, label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['cluster_raw'].map(rank_to_label)
cluster_order = ['㉮', '㉯', '㉰']

# 7. 2차원 산점도
st.subheader("📊 2차원 산점도")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_2d_label = st.selectbox("가로축 속성:", selected_feature_labels, index=0)
with col2d_2:
    y_2d_label = st.selectbox("세로축 속성:", selected_feature_labels, index=min(1, len(selected_feature_labels)-1))

fig2d = px.scatter(
    df,
    x=feature_options[x_2d_label],
    y=feature_options[y_2d_label],
    color='cluster',
    category_orders={'cluster': cluster_order},
    hover_name='movieNm',
    labels={
        feature_options[x_2d_label]: x_2d_label,
        feature_options[y_2d_label]: y_2d_label,
        'cluster': '묶음'
    },
    title=f"2차원 산점도 ({x_2d_label} vs {y_2d_label})"
)
st.plotly_chart(fig2d, use_container_width=True)

st.divider()

# 8. 3차원 산점도
st.subheader("🧊 3차원 산점도")
if len(selected_feature_labels) < 3:
    st.info("ℹ️ 고른 속성이 셋보다 적어 3차원 산점도를 그릴 수 없습니다. 속성을 3개 이상 선택해 주세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_3d_label = st.selectbox("x축 속성:", selected_feature_labels, index=0, key="3d_x")
    with col3d_2:
        y_3d_label = st.selectbox("y축 속성:", selected_feature_labels, index=1, key="3d_y")
    with col3d_3:
        z_3d_label = st.selectbox("z축 속성:", selected_feature_labels, index=2, key="3d_z")

    fig3d = px.scatter_3d(
        df,
        x=feature_options[x_3d_label],
        y=feature_options[y_3d_label],
        z=feature_options[z_3d_label],
        color='cluster',
        category_orders={'cluster': cluster_order},
        hover_name='movieNm',
        labels={
            feature_options[x_3d_label]: x_3d_label,
            feature_options[y_3d_label]: y_3d_label,
            feature_options[z_3d_label]: z_3d_label,
            'cluster': '묶음'
        },
        title=f"3차원 산점도 ({x_3d_label} x {y_3d_label} x {z_3d_label})"
    )
    fig3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig3d, use_container_width=True)

st.divider()

# 9. 묶음마다 편수와 네 속성의 평균(원래 단위) 표
st.subheader("📈 묶음별 편수 및 속성 평균 (원래 단위)")

summary_list = []
for c in cluster_order:
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

# 10. 묶음마다 누적 관객이 많은 순서로 영화 5편의 제목
st.subheader("🏆 묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)
for idx, c in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"### 묶음 {c}")
        top5 = df[df['cluster'] == c].sort_values(by='total_audi', ascending=False).head(5)
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            st.write(f"**{rank}. {row['movieNm']}** ({int(row['total_audi']):,}명)")
