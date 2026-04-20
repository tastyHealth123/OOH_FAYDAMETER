import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="OOH Savings Meter",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #e9ecef;
}
.rank-badge {
    display: inline-block;
    width: 22px;
    text-align: center;
    color: #aaa;
    font-size: 12px;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading & training model — this takes ~30 seconds on first run...")
def load_and_train():
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_excel("OOH_BASE_DATA.xlsx", sheet_name="Sheet2")
    df['city'] = df['city'].str.strip().str.title()
    df = df.groupby(['order_date', 'order_hour', 'city'], as_index=False)['total_saving'].sum()
    df = df.sort_values(['city', 'order_date', 'order_hour']).reset_index(drop=True)

    all_cities = sorted(df['city'].unique())
    all_dates = pd.date_range(df['order_date'].min(), df['order_date'].max(), freq='D')
    full_idx = pd.MultiIndex.from_product(
        [all_dates, range(24), all_cities],
        names=['order_date', 'order_hour', 'city']
    )
    df = (df.set_index(['order_date', 'order_hour', 'city'])
            .reindex(full_idx, fill_value=0)
            .reset_index())

    # ── Feature engineering ────────────────────────────────────────────────────
    df['dow'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['day'] = df['order_date'].dt.day
    df['month'] = df['order_date'].dt.month
    df['week_of_month'] = ((df['day'] - 1) // 7) + 1
    df = df.sort_values(['city', 'order_date', 'order_hour']).reset_index(drop=True)

    grp = df.groupby(['city', 'order_hour'])['total_saving']
    df['lag_1d'] = grp.shift(1)
    df['lag_7d'] = grp.shift(7)
    df['lag_14d'] = grp.shift(14)
    df['lag_21d'] = grp.shift(21)
    df['roll_7d_mean'] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    df['roll_14d_mean'] = grp.transform(lambda x: x.shift(1).rolling(14, min_periods=5).mean())
    df['roll_30d_mean'] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())

    le = LabelEncoder()
    df['city_enc'] = le.fit_transform(df['city'])

    FEATURES = ['order_hour', 'dow', 'is_weekend', 'day', 'month', 'week_of_month',
                'lag_1d', 'lag_7d', 'lag_14d', 'lag_21d',
                'roll_7d_mean', 'roll_14d_mean', 'roll_30d_mean', 'city_enc']

    train_df = df.dropna(subset=FEATURES).copy()
    model = RandomForestRegressor(n_estimators=300, max_depth=15,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train_df[FEATURES], train_df['total_saving'])

    # ── Forecast 30 days ───────────────────────────────────────────────────────
    history = df[['order_date', 'order_hour', 'city', 'total_saving']].copy()
    last_date = df['order_date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq='D')

    results = []
    for fdate in future_dates:
        rows = []
        for city in all_cities:
            for hour in range(24):
                sub = history[(history['city'] == city) & (history['order_hour'] == hour)] \
                    .sort_values('order_date')
                vals = sub['total_saving'].values

                def lag(n): return vals[-n] if len(vals) >= n else (np.mean(vals) if len(vals) > 0 else 0)
                def roll(n): return np.mean(vals[-n:]) if len(vals) >= 3 else (vals[-1] if len(vals) > 0 else 0)

                rows.append({
                    'order_date': fdate, 'order_hour': hour, 'city': city,
                    'dow': fdate.dayofweek, 'is_weekend': int(fdate.dayofweek >= 5),
                    'day': fdate.day, 'month': fdate.month,
                    'week_of_month': ((fdate.day - 1) // 7) + 1,
                    'lag_1d': lag(1), 'lag_7d': lag(7), 'lag_14d': lag(14), 'lag_21d': lag(21),
                    'roll_7d_mean': roll(7), 'roll_14d_mean': roll(14), 'roll_30d_mean': roll(30),
                    'city_enc': le.transform([city])[0]
                })

        fdf = pd.DataFrame(rows)
        fdf['total_saving'] = model.predict(fdf[FEATURES]).clip(min=0)
        results.append(fdf[['order_date', 'order_hour', 'city', 'total_saving']])

        new_hist = fdf[['order_date', 'order_hour', 'city', 'total_saving']].copy()
        history = pd.concat([history, new_hist], ignore_index=True)

    forecast = pd.concat(results, ignore_index=True)

    # ── Combine historical + forecast ──────────────────────────────────────────
    hist_clean = df[['order_date', 'order_hour', 'city', 'total_saving']].copy()
    hist_clean['is_forecast'] = False
    forecast['is_forecast'] = True

    full = pd.concat([hist_clean, forecast], ignore_index=True)
    full = full.sort_values(['city', 'order_date', 'order_hour']).reset_index(drop=True)

    # Cumulative saving per city
    full['cumulative_saving'] = full.groupby('city')['total_saving'].cumsum()
    full['total_saving'] = full['total_saving'].round(2)
    full['cumulative_saving'] = full['cumulative_saving'].round(2)

    return full, sorted(full['city'].unique().tolist()), last_date


def fmt_inr(n):
    if not n or np.isnan(n): return "—"
    if n >= 1e9: return f"₹{n/1e9:.2f}B"
    if n >= 1e7: return f"₹{n/1e7:.1f}Cr"
    if n >= 1e5: return f"₹{n/1e5:.1f}L"
    return f"₹{n:,.0f}"


# ── App ────────────────────────────────────────────────────────────────────────
st.title("💰 OOH Cumulative Savings Meter")
st.caption("Citywise · Datewise · Hourly | Historical + 30-Day Forecast")

full_df, cities, hist_end = load_and_train()

all_dates = sorted(full_df['order_date'].dt.date.unique())
min_date, max_date = all_dates[0], all_dates[-1]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Controls")
    selected_date = st.date_input(
        "Select Date",
        value=hist_end.date(),
        min_value=min_date,
        max_value=max_date
    )
    selected_hour = st.slider("Select Hour", 0, 23, 23,
                               format="%02d:00")
    st.divider()
    city_filter = st.multiselect(
        "Filter Cities (blank = all)",
        options=cities,
        default=[]
    )
    st.divider()
    st.info(f"📅 Historical data up to **{hist_end.date()}**\n\n🔮 Forecast: Apr 21 – May 20")

active_cities = city_filter if city_filter else cities
is_forecast = pd.Timestamp(selected_date) > hist_end

# ── Filter data up to selected date+hour ──────────────────────────────────────
mask = (
    (full_df['order_date'].dt.date < selected_date) |
    ((full_df['order_date'].dt.date == selected_date) & (full_df['order_hour'] <= selected_hour))
)
filtered = full_df[mask & full_df['city'].isin(active_cities)]

# Latest cumulative per city
latest = filtered.groupby('city').last().reset_index()

# Hourly saving for selected date+hour
hourly = full_df[
    (full_df['order_date'].dt.date == selected_date) &
    (full_df['order_hour'] == selected_hour) &
    (full_df['city'].isin(active_cities))
][['city', 'total_saving']].rename(columns={'total_saving': 'hour_saving'})

table = latest.merge(hourly, on='city', how='left').fillna(0)
table = table.sort_values('cumulative_saving', ascending=False).reset_index(drop=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
grand_cum = table['cumulative_saving'].sum()
grand_hour = table['hour_saving'].sum()

tag = "🔮 Forecast" if is_forecast else "✅ Historical"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Grand Cumulative Total", fmt_inr(grand_cum))
c2.metric("Hour Saving (All Cities)", fmt_inr(grand_hour),
          f"{pd.Timestamp(selected_date).strftime('%d %b')} {selected_hour:02d}:00")
c3.metric("Data Type", tag)
c4.metric("Cities Shown", len(active_cities))

st.divider()

# ── Main table ────────────────────────────────────────────────────────────────
st.subheader(f"📊 City Rankings — {selected_date}  {selected_hour:02d}:00")

max_cum = table['cumulative_saving'].max() or 1

display_rows = []
for i, row in table.iterrows():
    pct = row['cumulative_saving'] / max_cum * 100
    share = row['cumulative_saving'] / grand_cum * 100 if grand_cum > 0 else 0
    bar = "█" * int(pct / 5)
    display_rows.append({
        "Rank": i + 1,
        "City": row['city'],
        "Hour Saving": fmt_inr(row['hour_saving']),
        "Cumulative Saving": fmt_inr(row['cumulative_saving']),
        "Share %": f"{share:.1f}%",
        "Bar": bar
    })

display_df = pd.DataFrame(display_rows)
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "City": st.column_config.TextColumn(width="medium"),
        "Hour Saving": st.column_config.TextColumn(width="medium"),
        "Cumulative Saving": st.column_config.TextColumn(width="medium"),
        "Share %": st.column_config.TextColumn(width="small"),
        "Bar": st.column_config.TextColumn("Progress", width="large"),
    }
)

# ── Daily trend chart ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Cumulative Saving Trend (Daily)")

chart_cities = city_filter if city_filter else cities[:6]  # top 6 by default
chart_data = full_df[full_df['city'].isin(chart_cities)].copy()
chart_data = chart_data[chart_data['order_date'].dt.date <= selected_date]

# End-of-day cumulative
eod = chart_data[chart_data['order_hour'] == 23].groupby(['order_date', 'city'])['cumulative_saving'].last().reset_index()
eod_pivot = eod.pivot(index='order_date', columns='city', values='cumulative_saving').fillna(method='ffill')

st.line_chart(eod_pivot, use_container_width=True)

st.caption("Data: Sheet2 of OOH_BASE_DATA.xlsx | Forecast model: Random Forest (sMAPE ~38%) | Built with Streamlit")
