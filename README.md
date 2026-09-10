# Cyber Threats
# Hosting Link Dashboards --->
https://cyber-threatsrxbtech.streamlit.app/
# 🌐 Global Cybersecurity Threats Dashboard

An interactive data visualization dashboard that explores worldwide cyber threat intelligence, revealing attack patterns, geographic hotspots, and threat actor behavior. Built with Streamlit and Plotly Express.

**Live Demo:** [https://cyber-threatsrxbtech.streamlit.app/](https://cyber-threatsrxbtech.streamlit.app/)

---

## 📊 Key Features

- **10 TB of Threat Intelligence Data** – Processed and aggregated global cyber threat records.
- **Attack Pattern Analysis** – Identify emerging trends, common attack vectors, and targeted industries.
- **Geographic Hotspots** – Visualize threat origins and targets on interactive maps.
- **Threat Actor Clustering** – K-Means clustering (pre-computed) groups threat actors by behavior, revealing 5 distinct TTP groups.
- **Interactive Filters** – Slice data by time period, region, attack type, and severity.
- **Dynamic Visualizations** – Line charts, bar charts, heatmaps, and scatter plots powered by Plotly Express.

---

## 🛠️ Tech Stack

- **Python** – Core data processing and logic.
- **Pandas & NumPy** – Data cleaning, transformation, and aggregation.
- **PySpark & Hive** – Large-scale data processing (10 TB) for the underlying dataset.
- **Streamlit** – Web application framework for the interactive dashboard.
- **Plotly Express** – Interactive, publication-quality visualizations.
- **AWS (S3, Lambda, SNS)** – Automated alerting pipeline for real-time threat detection.

---

## 🚀 How to Run Locally

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
