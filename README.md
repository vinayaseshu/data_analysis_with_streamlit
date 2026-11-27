# Data Analysis App

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinayaseshu/data_analysis_with_streamlit/blob/main/run_app.ipynb)

Simple data analysis app built with Streamlit, Polars and Plotly.

Features: CSV/Parquet upload (or built-in sample data), sidebar filters, summary stats, schema, correlation heatmap, interactive charts (histogram, scatter, line, box, bar), group-by aggregations, and filtered CSV download.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run from a notebook

[`run_app.ipynb`](run_app.ipynb) clones this repo, installs the dependencies, launches the app with a shareable link (Colab port proxy, or a Cloudflare quick tunnel elsewhere), and runs the same analysis inline. Click the **Open in Colab** badge above to run it.
