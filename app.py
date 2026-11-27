import io

import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st

st.set_page_config(page_title="Data Analysis App", layout="wide")


@st.cache_data
def sample_data(n: int = 1000) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        pl.date(2024, 1, 1), pl.date(2024, 1, 1) + pl.duration(days=n - 1), eager=True
    )
    return pl.DataFrame(
        {
            "date": dates,
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": rng.choice(["A", "B", "C"], n),
            "units": rng.integers(1, 100, n),
            "price": rng.normal(50, 12, n).round(2),
        }
    ).with_columns(revenue=(pl.col("units") * pl.col("price")).round(2))


@st.cache_data
def load_file(data: bytes, name: str) -> pl.DataFrame:
    buf = io.BytesIO(data)
    if name.endswith(".parquet"):
        return pl.read_parquet(buf)
    if name.endswith((".xlsx", ".xls")):
        return pl.read_excel(buf)
    return pl.read_csv(buf, try_parse_dates=True, infer_schema_length=10000)


st.title("Data Analysis App")
st.caption("Streamlit + Polars + Plotly")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV or Parquet file", type=["csv", "parquet"])

if uploaded:
    try:
        df = load_file(uploaded.getvalue(), uploaded.name.lower())
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    st.info("No file uploaded. Using a generated sample sales dataset.")
    df = sample_data()

numeric_cols = [c for c, t in df.schema.items() if t.is_numeric()]
temporal_cols = [c for c, t in df.schema.items() if t.is_temporal()]
categorical_cols = [c for c, t in df.schema.items() if t in (pl.String, pl.Categorical, pl.Boolean)]

with st.sidebar:
    st.header("Filters")
    filtered = df
    for col in categorical_cols:
        values = df[col].drop_nulls().unique().sort().to_list()
        if 1 < len(values) <= 50:
            chosen = st.multiselect(col, values, default=values)
            filtered = filtered.filter(pl.col(col).is_in(chosen))
    for col in numeric_cols:
        lo, hi = df[col].min(), df[col].max()
        if lo is not None and hi is not None and lo < hi:
            r = st.slider(col, float(lo), float(hi), (float(lo), float(hi)))
            filtered = filtered.filter(pl.col(col).is_between(r[0], r[1]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{filtered.height:,}", f"{filtered.height - df.height:,}" if filtered.height != df.height else None)
c2.metric("Columns", filtered.width)
c3.metric("Numeric columns", len(numeric_cols))
c4.metric("Missing values", f"{sum(filtered.null_count().row(0)):,}")

tab_data, tab_stats, tab_charts, tab_group = st.tabs(["Data", "Statistics", "Charts", "Group by"])

with tab_data:
    st.dataframe(filtered, width="stretch", height=450)
    st.download_button(
        "Download filtered CSV", filtered.write_csv().encode(), "filtered.csv", "text/csv"
    )

with tab_stats:
    st.subheader("Summary statistics")
    st.dataframe(filtered.describe(), width="stretch")
    st.subheader("Schema")
    st.dataframe(
        pl.DataFrame(
            {
                "column": filtered.columns,
                "dtype": [str(t) for t in filtered.dtypes],
                "nulls": filtered.null_count().row(0),
                "unique": [filtered[c].n_unique() for c in filtered.columns],
            }
        ),
        width="stretch",
    )
    if len(numeric_cols) >= 2:
        st.subheader("Correlation matrix")
        corr = filtered.select(numeric_cols).drop_nulls().corr()
        fig = px.imshow(
            corr.to_numpy(), x=numeric_cols, y=numeric_cols, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        st.plotly_chart(fig, width="stretch")

with tab_charts:
    chart = st.selectbox("Chart type", ["Histogram", "Scatter", "Line", "Box", "Bar"])
    color_opts = [None] + categorical_cols
    pdf = filtered.to_pandas()
    fig = None
    if chart == "Histogram" and numeric_cols:
        x = st.selectbox("Column", numeric_cols)
        color = st.selectbox("Color by", color_opts)
        fig = px.histogram(pdf, x=x, color=color, nbins=40, marginal="box")
    elif chart == "Scatter" and len(numeric_cols) >= 2:
        x = st.selectbox("X", numeric_cols, index=0)
        y = st.selectbox("Y", numeric_cols, index=1)
        color = st.selectbox("Color by", color_opts)
        fig = px.scatter(pdf, x=x, y=y, color=color, opacity=0.7)
    elif chart == "Line" and numeric_cols:
        x_opts = temporal_cols + numeric_cols
        x = st.selectbox("X", x_opts)
        y = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols) - 1))
        color = st.selectbox("Color by", color_opts)
        fig = px.line(pdf.sort_values(x), x=x, y=y, color=color)
    elif chart == "Box" and numeric_cols:
        y = st.selectbox("Value", numeric_cols)
        x = st.selectbox("Group by", color_opts)
        fig = px.box(pdf, x=x, y=y, color=x)
    elif chart == "Bar" and categorical_cols:
        x = st.selectbox("Category", categorical_cols)
        counts = filtered.group_by(x).len().sort("len", descending=True)
        fig = px.bar(counts.to_pandas(), x=x, y="len", labels={"len": "count"})
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Not enough suitable columns for this chart type.")

with tab_group:
    if categorical_cols and numeric_cols:
        g1, g2, g3 = st.columns(3)
        keys = g1.multiselect("Group by", categorical_cols, default=categorical_cols[:1])
        vals = g2.multiselect("Values", numeric_cols, default=numeric_cols[-1:])
        agg = g3.selectbox("Aggregation", ["sum", "mean", "median", "min", "max", "count"])
        if keys and vals:
            exprs = [getattr(pl.col(v), agg)().alias(f"{v}_{agg}") for v in vals]
            grouped = filtered.group_by(keys).agg(exprs).sort(keys)
            st.dataframe(grouped, width="stretch")
            fig = px.bar(
                grouped.to_pandas(), x=keys[0], y=f"{vals[0]}_{agg}",
                color=keys[1] if len(keys) > 1 else None, barmode="group",
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Need at least one categorical and one numeric column.")
