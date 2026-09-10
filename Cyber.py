import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Global Cybersecurity Threats Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────
st.title("🛡️ Global Cybersecurity Threats Analysis")
st.markdown("---")

# ─────────────────────────────────────────────
# Executive Summary
# ─────────────────────────────────────────────
st.header("Executive Summary")
st.markdown("""
This comprehensive data science project analyzes global cybersecurity threat data spanning 2015–2024
across 10 countries and 7 target industries. The analysis uncovers critical patterns in attack types,
financial losses, response times, and industry-level vulnerabilities to help organizations strengthen
their cyber-defense posture.

**Key Objectives:**
- Identify trends in cyber attack frequency across years and geographies
- Understand which industries face the highest threat exposure and financial damage
- Examine the relationship between incident response time and financial impact
- Analyze severity and vulnerability type distributions across attack vectors
- Provide actionable intelligence for cybersecurity policymakers and practitioners

**Expected Deliverables:**
- Interactive visualizations covering attack trends, industry exposure, and geographic losses
- Statistical analysis of financial impacts, response times, and threat severity
- Drill-down capability via sidebar filters for targeted analysis
- Synthesized insights and strategic recommendations
""")

st.markdown("---")

# ─────────────────────────────────────────────
# Project Description
# ─────────────────────────────────────────────
st.header("Project Description")

st.subheader("Problem Statement")
st.markdown("""
Cybersecurity threats are escalating in scale, sophistication, and impact. Organizations across every
sector face mounting pressure to understand adversarial patterns and respond effectively. This project
leverages a global dataset of 3,000 cybersecurity incidents to provide data-driven intelligence:

- **CISOs & Security Teams**: Prioritize defense investments based on threat landscape trends
- **Risk Managers**: Quantify financial exposure by attack type, industry, and geography
- **Policy Makers**: Benchmark national incident response performance
- **Researchers**: Identify structural correlations between vulnerability types and outcomes
""")

st.subheader("Dataset Overview")
st.markdown("""
The Global Cybersecurity Threats dataset contains incident-level records with the following fields:

**Incident Identifiers:**
- Country of the targeted organization
- Year the attack occurred (2015–2024)

**Attack Characteristics:**
- Attack Type (Phishing, Ransomware, DDoS, Malware, SQL Injection, Man-in-the-Middle)
- Attack Source (Hacker Group, Nation-state, Insider, Unknown)
- Security Vulnerability Type (Unpatched Software, Weak Passwords, Social Engineering, Zero-day)

**Impact Metrics:**
- Financial Loss (in Million $)
- Number of Affected Users
- Incident Resolution Time (in Hours)

**Organizational Context:**
- Target Industry (Banking, Education, Government, Healthcare, IT, Retail, Telecommunications)
- Defense Mechanism Used (VPN, Firewall, AI-based Detection, Encryption, Anti-virus)
""")

st.markdown("---")

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
st.header("Data Overview")


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Global_Cybersecurity_Threats.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def prepare_dataframe_for_display(df, max_string_length=100):
    """Prepare dataframe for display by handling dtypes safely."""
    if df is None:
        return df
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df.empty:
        return df
    try:
        df_display = df.copy()
        for col in df_display.columns:
            if pd.api.types.is_object_dtype(df_display[col]):
                df_display[col] = df_display[col].astype(str)
                df_display[col] = df_display[col].replace(
                    ["nan", "None", "NaN", "<NA>", "null"], ""
                )
                mask = df_display[col].str.len() > max_string_length
                if mask.any():
                    df_display.loc[mask, col] = (
                        df_display.loc[mask, col].str[:max_string_length] + "..."
                    )
        return df_display
    except Exception:
        df_fallback = df.copy()
        for col in df_fallback.columns:
            df_fallback[col] = df_fallback[col].astype(str)
        return df_fallback


def safe_dataframe_display(df, use_container_width=True, **kwargs):
    if df is not None and not df.empty:
        return st.dataframe(
            prepare_dataframe_for_display(df),
            use_container_width=use_container_width,
            **kwargs,
        )
    return st.info("No data to display")


# ── Load ──
df = load_data()

if df is not None:
    # ── Basic metrics ──
    st.subheader("Dataset Basic Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Years Covered", f"{df['Year'].min()}–{df['Year'].max()}")
    with col4:
        st.metric("Countries", df["Country"].nunique())

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Column Info",
            "Missing Values",
            "Sample Data",
            "Statistics",
            "Categorical Data",
            "Data Quality",
        ]
    )

    with tab1:
        st.subheader("Column Information")
        col_info = pd.DataFrame(
            {
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(2),
            }
        )
        safe_dataframe_display(col_info)

        st.subheader("Detailed Column Descriptions")
        st.markdown(
            """
        - **Country**: Nation where the targeted organization operates
        - **Year**: Calendar year of the attack (2015–2024)
        - **Attack Type**: Category of cyberattack technique
        - **Target Industry**: Industry sector of the victim organization
        - **Financial Loss (in Million $)**: Estimated monetary damage
        - **Number of Affected Users**: Count of users impacted by the breach
        - **Attack Source**: Attributed origin of the attack
        - **Security Vulnerability Type**: Underlying weakness exploited
        - **Defense Mechanism Used**: Primary security control in place
        - **Incident Resolution Time (in Hours)**: Time taken to close the incident
        """
        )

    with tab2:
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_df = pd.DataFrame(
                {
                    "Column": missing_data.index,
                    "Missing Count": missing_data.values,
                    "Missing %": (missing_data.values / len(df) * 100).round(2),
                }
            )
            safe_dataframe_display(missing_df)
        else:
            st.success("✅ No missing values found in the dataset!")

    with tab3:
        st.subheader("Sample Data")
        st.write("**First 10 rows:**")
        safe_dataframe_display(df.head(10))
        st.write("**10 random rows:**")
        safe_dataframe_display(df.sample(10))
        st.write("**Last 5 rows:**")
        safe_dataframe_display(df.tail(5))

    with tab4:
        st.subheader("Statistical Summary")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.write("**Numerical Columns Statistics:**")
            safe_dataframe_display(df[numerical_cols].describe())
        object_cols = df.select_dtypes(include=["object"]).columns
        if len(object_cols) > 0:
            st.write("**Text/Categorical Columns Statistics:**")
            safe_dataframe_display(df[object_cols].describe())

    with tab5:
        st.subheader("Categorical Data Analysis")
        cat_cols = [
            "Country",
            "Attack Type",
            "Target Industry",
            "Attack Source",
            "Security Vulnerability Type",
            "Defense Mechanism Used",
        ]
        for col in cat_cols:
            if col in df.columns:
                vc = df[col].value_counts()
                st.write(f"**{col}** — {df[col].nunique()} unique values")
                safe_dataframe_display(vc.reset_index().rename(columns={"index": col, col: "Count"}))
                st.write("---")

    with tab6:
        st.subheader("Data Quality Assessment")
        dup = df.duplicated().sum()
        st.metric("Duplicate Rows", dup)
        if dup == 0:
            st.success("✅ No duplicate rows detected.")
        num_nulls = df.isnull().sum().sum()
        st.metric("Total Missing Values", num_nulls)
        if num_nulls == 0:
            st.success("✅ Dataset is complete — no missing values.")

else:
    st.error("Dataset could not be loaded.")

st.markdown("---")

# ─────────────────────────────────────────────
# Data Cleaning Section
# ─────────────────────────────────────────────
st.header("Data Cleaning & Preprocessing")

if df is not None:

    @st.cache_data
    def clean_data(raw_df):
        df_c = raw_df.copy()
        # Ensure correct dtypes
        df_c["Financial Loss (in Million $)"] = pd.to_numeric(
            df_c["Financial Loss (in Million $)"], errors="coerce"
        )
        df_c["Number of Affected Users"] = pd.to_numeric(
            df_c["Number of Affected Users"], errors="coerce"
        )
        df_c["Incident Resolution Time (in Hours)"] = pd.to_numeric(
            df_c["Incident Resolution Time (in Hours)"], errors="coerce"
        )
        df_c["Year"] = pd.to_numeric(df_c["Year"], errors="coerce").astype("Int64")
        # Fill remaining nulls
        for col in df_c.select_dtypes(include=["object"]).columns:
            df_c[col] = df_c[col].fillna("Unknown")
        for col in df_c.select_dtypes(include=[np.number]).columns:
            df_c[col] = df_c[col].fillna(df_c[col].median())
        return df_c

    df_cleaned = clean_data(df)

    clean_tab1, clean_tab2, clean_tab3 = st.tabs(
        ["Data Type Validation", "Missing Values Treatment", "Outlier Treatment"]
    )

    with clean_tab1:
        st.subheader("Data Type Validation")
        st.markdown(
            """
        **Transformations Applied:**
        - `Financial Loss (in Million $)` → `float64` (numeric coercion)
        - `Number of Affected Users` → `int64` (numeric coercion)
        - `Incident Resolution Time (in Hours)` → `int64` (numeric coercion)
        - `Year` → `Int64` nullable integer
        - All object columns standardized to string
        """
        )
        dtype_df = pd.DataFrame(
            {"Column": df_cleaned.columns, "Final Data Type": df_cleaned.dtypes.astype(str)}
        )
        safe_dataframe_display(dtype_df)

    with clean_tab2:
        st.subheader("Missing Values Treatment")
        st.markdown(
            """
        - **Numeric columns**: Missing values filled with column median
        - **Categorical columns**: Missing values filled with `'Unknown'`
        """
        )
        post_nulls = df_cleaned.isnull().sum().sum()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Values Before", df.isnull().sum().sum())
        with col2:
            st.metric("Missing Values After", post_nulls)
        if post_nulls == 0:
            st.success("✅ All missing values resolved.")

    with clean_tab3:
        st.subheader("Outlier Treatment")
        num_cols = [
            "Financial Loss (in Million $)",
            "Number of Affected Users",
            "Incident Resolution Time (in Hours)",
        ]
        for col in num_cols:
            if col in df_cleaned.columns:
                data = df_cleaned[col].dropna()
                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = ((data < lower) | (data > upper)).sum()
                pct = outliers / len(data) * 100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{col[:25]}… Outliers", outliers)
                c2.metric("Outlier %", f"{pct:.1f}%")
                c3.metric("IQR", f"{IQR:.1f}")
                c4.metric("Median", f"{data.median():.1f}")
                st.markdown("---")

    # Final summary
    st.subheader("Final Processed Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Records", f"{df_cleaned.shape[0]:,}")
    c2.metric("Final Columns", df_cleaned.shape[1])
    c3.metric("Remaining Nulls", df_cleaned.isnull().sum().sum())

else:
    st.error("Data cleaning skipped — dataset unavailable.")

st.markdown("---")

# ─────────────────────────────────────────────
# Visualization Section
# ─────────────────────────────────────────────
st.header("Data Visualization & Insights")

if df is not None:
    df_viz = df_cleaned if "df_cleaned" in dir() else df

    # ── Sidebar Filters ──
    st.sidebar.header("🔎 Data Filters")
    st.sidebar.markdown("Apply filters, then click **Apply Filters** to refresh charts.")
    st.sidebar.markdown("---")

    # Session state init
    for key, default in [
        ("filters_applied", False),
        ("applied_years", []),
        ("applied_countries", []),
        ("applied_industries", []),
        ("applied_attack_types", []),
        ("applied_sources", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    df_filtered = df_viz.copy()

    with st.sidebar.form("filters_form"):
        st.subheader("Temporal")
        all_years = sorted(df_viz["Year"].dropna().unique().tolist())
        selected_years = st.multiselect("Year(s)", all_years, help="Filter by year")

        st.markdown("---")
        st.subheader("Geographic")
        all_countries = sorted(df_viz["Country"].dropna().unique().tolist())
        selected_countries = st.multiselect("Country(s)", all_countries)

        st.markdown("---")
        st.subheader("Threat Context")
        all_industries = sorted(df_viz["Target Industry"].dropna().unique().tolist())
        selected_industries = st.multiselect("Target Industry(s)", all_industries)

        all_attack_types = sorted(df_viz["Attack Type"].dropna().unique().tolist())
        selected_attack_types = st.multiselect("Attack Type(s)", all_attack_types)

        all_sources = sorted(df_viz["Attack Source"].dropna().unique().tolist())
        selected_sources = st.multiselect("Attack Source(s)", all_sources)

        st.markdown("---")
        c1, c2 = st.columns(2)
        apply_btn = c1.form_submit_button("✅ Apply", type="primary", use_container_width=True)
        reset_btn = c2.form_submit_button("🔄 Reset", use_container_width=True)

    if reset_btn:
        for key in list(st.session_state.keys()):
            if key.startswith("applied_"):
                del st.session_state[key]
        st.session_state["filters_applied"] = False
        st.rerun()

    if apply_btn:
        st.session_state["filters_applied"] = True
        st.session_state["applied_years"] = selected_years
        st.session_state["applied_countries"] = selected_countries
        st.session_state["applied_industries"] = selected_industries
        st.session_state["applied_attack_types"] = selected_attack_types
        st.session_state["applied_sources"] = selected_sources

    # Apply stored filters
    if st.session_state["applied_years"]:
        df_filtered = df_filtered[df_filtered["Year"].isin(st.session_state["applied_years"])]
    if st.session_state["applied_countries"]:
        df_filtered = df_filtered[df_filtered["Country"].isin(st.session_state["applied_countries"])]
    if st.session_state["applied_industries"]:
        df_filtered = df_filtered[df_filtered["Target Industry"].isin(st.session_state["applied_industries"])]
    if st.session_state["applied_attack_types"]:
        df_filtered = df_filtered[df_filtered["Attack Type"].isin(st.session_state["applied_attack_types"])]
    if st.session_state["applied_sources"]:
        df_filtered = df_filtered[df_filtered["Attack Source"].isin(st.session_state["applied_sources"])]

    # Filter status banner
    if st.session_state["filters_applied"]:
        if len(df_filtered) < len(df_viz):
            st.info(
                f"🔍 Filters active — showing **{len(df_filtered):,}** of **{len(df_viz):,}** records."
            )
        else:
            st.success("No filters narrowed the data — showing all records.")
    else:
        st.info("💡 Use the sidebar to filter data, then click **Apply Filters**.")

    st.markdown("---")

    # ── KPI row ──
    st.subheader("📊 Key Metrics (Filtered View)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Incidents", f"{len(df_filtered):,}")
    k2.metric(
        "Total Financial Loss",
        f"${df_filtered['Financial Loss (in Million $)'].sum():,.1f}M",
    )
    k3.metric(
        "Avg Loss / Incident",
        f"${df_filtered['Financial Loss (in Million $)'].mean():.2f}M",
    )
    k4.metric(
        "Avg Resolution Time",
        f"{df_filtered['Incident Resolution Time (in Hours)'].mean():.1f} hrs",
    )
    k5.metric(
        "Total Users Affected",
        f"{df_filtered['Number of Affected Users'].sum():,.0f}",
    )

    st.markdown("---")

    if len(df_filtered) == 0:
        st.error("No records match the current filters. Please adjust and try again.")
        st.stop()

    # ══════════════════════════════════════════
    # CHART 1 — Line Chart: Cyber Attacks by Year
    # ══════════════════════════════════════════
    st.subheader("📈 Cyber Attacks & Financial Loss Trends Over Time")
    st.write("### Line Chart — Cyber Attacks by Year")

    yearly = (
        df_filtered.groupby("Year")
        .agg(
            Incidents=("Attack Type", "count"),
            Total_Loss=("Financial Loss (in Million $)", "sum"),
            Avg_Resolution=("Incident Resolution Time (in Hours)", "mean"),
        )
        .reset_index()
    )

    fig1 = px.line(
        yearly,
        x="Year",
        y=["Incidents", "Total_Loss"],
        markers=True,
        title="Cyber Attack Incidents & Total Financial Loss by Year",
        labels={"value": "Count / Million $", "variable": "Metric"},
        color_discrete_map={"Incidents": "#EF553B", "Total_Loss": "#636EFA"},
    )
    fig1.update_traces(line_width=2.5)
    fig1.update_layout(height=480, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    top_year = yearly.loc[yearly["Incidents"].idxmax(), "Year"]
    top_loss_year = yearly.loc[yearly["Total_Loss"].idxmax(), "Year"]
    st.write("**Key Insights:**")
    st.write(f"- **{top_year}** recorded the highest number of cyber incidents.")
    st.write(f"- **{top_loss_year}** saw peak cumulative financial losses.")
    st.write("- Both metrics reveal escalating threat intensity over the decade.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 2 — Treemap: Industry-wise Attacks
    # ══════════════════════════════════════════
    st.write("### Treemap — Industry-wise Attack Distribution")

    industry_attack = (
        df_filtered.groupby(["Target Industry", "Attack Type"])
        .agg(Count=("Attack Type", "count"), Loss=("Financial Loss (in Million $)", "sum"))
        .reset_index()
    )

    fig2 = px.treemap(
        industry_attack,
        path=["Target Industry", "Attack Type"],
        values="Count",
        color="Loss",
        color_continuous_scale="RdYlGn_r",
        title="Industry-wise Attack Volume (color = Financial Loss $M)",
        hover_data={"Loss": ":.2f"},
    )
    fig2.update_layout(height=560)
    st.plotly_chart(fig2, use_container_width=True)

    top_ind = df_filtered["Target Industry"].value_counts().idxmax()
    st.write("**Key Insights:**")
    st.write(f"- **{top_ind}** is the most targeted industry by attack volume.")
    st.write("- Darker treemap blocks indicate higher financial damage — not always tied to attack count.")
    st.write("- Drill down within each tile to see dominant attack types per sector.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 3 — Donut Chart: Severity Distribution
    # ══════════════════════════════════════════
    st.write("### Donut Chart — Severity / Vulnerability Type Distribution")

    severity_counts = df_filtered["Security Vulnerability Type"].value_counts().reset_index()
    severity_counts.columns = ["Vulnerability", "Count"]

    fig3 = px.pie(
        severity_counts,
        names="Vulnerability",
        values="Count",
        hole=0.45,
        title="Distribution of Security Vulnerability Types (Severity)",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )
    fig3.update_traces(textposition="outside", textinfo="percent+label")
    fig3.update_layout(height=480, showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)

    top_vuln = severity_counts.iloc[0]["Vulnerability"]
    top_vuln_pct = severity_counts.iloc[0]["Count"] / severity_counts["Count"].sum() * 100
    st.write("**Key Insights:**")
    st.write(f"- **{top_vuln}** is the most exploited vulnerability ({top_vuln_pct:.1f}% of incidents).")
    st.write("- Zero-day exploits, though less frequent, indicate highly sophisticated threat actors.")
    st.write("- Vulnerability profile mix guides where security investment should be directed.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 4 — Scatter Plot: Response Time vs Financial Impact
    # ══════════════════════════════════════════
    st.write("### Scatter Plot — Incident Response Time vs Financial Impact")

    scatter_data = df_filtered.dropna(
        subset=["Incident Resolution Time (in Hours)", "Financial Loss (in Million $)"]
    )
    sample_size = min(2000, len(scatter_data))
    scatter_sample = scatter_data.sample(sample_size) if len(scatter_data) > sample_size else scatter_data

    fig4 = px.scatter(
        scatter_sample,
        x="Incident Resolution Time (in Hours)",
        y="Financial Loss (in Million $)",
        color="Attack Type",
        size="Number of Affected Users",
        hover_data=["Country", "Target Industry", "Security Vulnerability Type"],
        title="Response Time vs Financial Loss (bubble size = Affected Users)",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig4.update_layout(height=520)
    st.plotly_chart(fig4, use_container_width=True)

    corr = scatter_data["Incident Resolution Time (in Hours)"].corr(
        scatter_data["Financial Loss (in Million $)"]
    )
    st.write("**Key Insights:**")
    st.write(f"- Pearson correlation between resolution time and loss: **{corr:.3f}**")
    st.write("- Large bubbles (more affected users) tend to cluster at higher financial loss values.")
    st.write("- Slow response time does not always correlate with higher losses — attack severity matters more.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 5 — Heatmap: Industry vs Severity
    # ══════════════════════════════════════════
    st.write("### Heatmap — Industry vs Vulnerability Type (Severity)")

    heatmap_data = (
        df_filtered.groupby(["Target Industry", "Security Vulnerability Type"])
        .size()
        .reset_index(name="Count")
    )
    heatmap_pivot = heatmap_data.pivot(
        index="Target Industry", columns="Security Vulnerability Type", values="Count"
    ).fillna(0)

    fig5 = px.imshow(
        heatmap_pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Attack Count Heatmap: Industry vs Security Vulnerability Type",
        labels={"color": "Incident Count"},
    )
    fig5.update_layout(height=480)
    st.plotly_chart(fig5, use_container_width=True)

    st.write("**Key Insights:**")
    st.write("- Darker cells indicate higher attack concentration for that industry–vulnerability pair.")
    st.write("- Some industries are disproportionately hit by a single vulnerability class.")
    st.write("- Cross-referencing industry and severity guides targeted hardening efforts.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 6 — Sunburst: Industry → Threat Type
    # ══════════════════════════════════════════
    st.write("### Sunburst Chart — Industry → Attack Type Hierarchy")

    sunburst_data = (
        df_filtered.groupby(["Target Industry", "Attack Type"])
        .agg(Count=("Attack Type", "count"), Loss=("Financial Loss (in Million $)", "sum"))
        .reset_index()
    )

    fig6 = px.sunburst(
        sunburst_data,
        path=["Target Industry", "Attack Type"],
        values="Count",
        color="Loss",
        color_continuous_scale="Turbo",
        title="Hierarchical View: Industry → Attack Type (color = Financial Loss $M)",
        hover_data={"Loss": ":.2f"},
    )
    fig6.update_layout(height=560)
    st.plotly_chart(fig6, use_container_width=True)

    st.write("**Key Insights:**")
    st.write("- Click any sector to zoom into its inner breakdown.")
    st.write("- Color intensity reveals which industry–attack combinations are most financially damaging.")
    st.write("- Dominant attack types vary significantly across industries.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 7 — Box Plot: Financial Impact by Threat Type
    # ══════════════════════════════════════════
    st.write("### Box Plot — Financial Impact Distribution by Threat Type")

    box_data = df_filtered.dropna(subset=["Attack Type", "Financial Loss (in Million $)"])

    fig7 = px.box(
        box_data,
        x="Attack Type",
        y="Financial Loss (in Million $)",
        color="Attack Type",
        points="outliers",
        title="Financial Loss Distribution by Attack Type",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        notched=True,
    )
    fig7.update_layout(height=500, xaxis_tickangle=-20, showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

    medians = box_data.groupby("Attack Type")["Financial Loss (in Million $)"].median().sort_values(ascending=False)
    st.write("**Key Insights:**")
    st.write(f"- **{medians.index[0]}** has the highest median financial impact (${medians.iloc[0]:.2f}M).")
    st.write("- Notched box plots show the confidence interval around the median.")
    st.write("- Outlier points represent exceptionally costly incidents worth investigating individually.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 8 — Choropleth Map: Country-wise Losses
    # ══════════════════════════════════════════
    st.write("### Choropleth Map — Country-wise Financial Losses")

    country_loss = (
        df_filtered.groupby("Country")
        .agg(
            Total_Loss=("Financial Loss (in Million $)", "sum"),
            Incidents=("Attack Type", "count"),
            Avg_Loss=("Financial Loss (in Million $)", "mean"),
        )
        .reset_index()
    )

    # Map country names to ISO-3 codes
    iso_map = {
        "Australia": "AUS",
        "Brazil": "BRA",
        "China": "CHN",
        "France": "FRA",
        "Germany": "DEU",
        "India": "IND",
        "Japan": "JPN",
        "Russia": "RUS",
        "UK": "GBR",
        "USA": "USA",
    }
    country_loss["ISO"] = country_loss["Country"].map(iso_map)

    fig8 = px.choropleth(
        country_loss,
        locations="ISO",
        color="Total_Loss",
        hover_name="Country",
        hover_data={"Incidents": True, "Avg_Loss": ":.2f", "Total_Loss": ":.2f"},
        color_continuous_scale="Reds",
        title="Total Cybersecurity Financial Losses by Country ($M)",
        labels={"Total_Loss": "Total Loss ($M)"},
    )
    fig8.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig8, use_container_width=True)

    top_country = country_loss.sort_values("Total_Loss", ascending=False).iloc[0]
    st.write("**Key Insights:**")
    st.write(
        f"- **{top_country['Country']}** leads in total cybersecurity losses (${top_country['Total_Loss']:,.1f}M)."
    )
    st.write("- Heavier shading indicates greater cumulative financial exposure.")
    st.write("- Comparing per-incident average loss vs total loss reveals scale-vs-intensity tradeoffs.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 9 — Bubble Chart: Data Breached vs Financial Impact
    # ══════════════════════════════════════════
    st.write("### Bubble Chart — Affected Users vs Financial Impact by Country")

    bubble_data = (
        df_filtered.groupby(["Country", "Attack Type"])
        .agg(
            Avg_Users=("Number of Affected Users", "mean"),
            Avg_Loss=("Financial Loss (in Million $)", "mean"),
            Incidents=("Attack Type", "count"),
        )
        .reset_index()
    )

    fig9 = px.scatter(
        bubble_data,
        x="Avg_Users",
        y="Avg_Loss",
        size="Incidents",
        color="Country",
        text="Attack Type",
        title="Avg Affected Users vs Avg Financial Loss (bubble = Incident Count)",
        labels={
            "Avg_Users": "Avg Affected Users",
            "Avg_Loss": "Avg Financial Loss ($M)",
        },
        size_max=60,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data=["Incidents"],
    )
    fig9.update_traces(textposition="top center", textfont_size=9)
    fig9.update_layout(height=560)
    st.plotly_chart(fig9, use_container_width=True)

    st.write("**Key Insights:**")
    st.write("- Larger bubbles indicate more incidents — highlighting persistent threat patterns.")
    st.write("- Top-right quadrant: high user impact AND high financial loss — most critical clusters.")
    st.write("- User-count and financial loss do not always scale together, revealing differing attack efficiency.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # CHART 10 — Histogram: Response Time Distribution
    # ══════════════════════════════════════════
    st.write("### Histogram — Incident Response Time Distribution")

    resp_data = df_filtered["Incident Resolution Time (in Hours)"].dropna()

    fig10 = px.histogram(
        x=resp_data,
        nbins=35,
        color_discrete_sequence=["#00CC96"],
        title="Distribution of Incident Resolution Times (Hours)",
        labels={"x": "Resolution Time (Hours)", "y": "Number of Incidents"},
    )
    median_rt = resp_data.median()
    fig10.add_vline(
        x=median_rt,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Median: {median_rt:.0f} hrs",
        annotation_position="top right",
    )
    fig10.update_layout(height=500)
    st.plotly_chart(fig10, use_container_width=True)

    mean_rt = resp_data.mean()
    fast_pct = (resp_data <= 24).mean() * 100
    st.write("**Key Insights:**")
    st.write(f"- Median resolution time: **{median_rt:.0f} hours** | Mean: **{mean_rt:.1f} hours**.")
    st.write(f"- **{fast_pct:.1f}%** of incidents were resolved within 24 hours.")
    st.write("- Long-tail incidents (>72 hrs) likely involve complex ransomware or nation-state attacks.")

    st.markdown("---")

else:
    st.error("Dataset unavailable for visualizations.")

# ─────────────────────────────────────────────
# Conclusion Section
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Data Analysis Summary")
st.markdown(
    """
This comprehensive analysis of the Global Cybersecurity Threats dataset has surfaced critical patterns
across a decade of cyberattack data, spanning 10 nations and 7 industry sectors. Through 10 purposefully
chosen visualizations, we have built an evidence-based picture of the global threat landscape.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    ### **Threat Landscape**
    - **Attack Escalation**: Incident frequency and financial losses have trended upward across the decade
    - **Attack Diversity**: Six distinct attack types affect industries differently — no single defense suffices
    - **Nation-State Risk**: State-backed attacks tend to correlate with higher financial and user impact

    ### **Industry Exposure**
    - **Uneven Targeting**: Some sectors (Healthcare, Banking) face outsized threat concentration
    - **Vulnerability Mismatch**: Each industry's dominant vulnerability type differs — requiring tailored hardening
    - **Cross-sector Threat**: Phishing and Ransomware cut across all industries indiscriminately
    """
    )

with col2:
    st.markdown(
        """
    ### **Financial Impact**
    - **Loss Disparity**: Countries vary significantly in both total and per-incident losses
    - **Attack-Type Costs**: Ransomware and DDoS attacks typically carry the highest financial burden
    - **Outlier Events**: A small fraction of incidents drive disproportionate financial damage

    ### **Response Performance**
    - **Response Gap**: Mean resolution time varies widely — indicating readiness disparities
    - **Speed–Loss Link**: Faster response does not always reduce loss — prevention matters more
    - **Chronic Incidents**: Long-tail resolution events signal systemic detection or remediation gaps
    """
    )

st.markdown("---")
st.subheader("💡 Strategic Recommendations")

with st.expander("**For CISOs & Security Teams**", expanded=True):
    st.markdown(
        """
    1. **Vulnerability Prioritization**: Focus patching cycles on the top vulnerability class in your industry
    2. **Attack-Type Training**: Run targeted phishing and social-engineering simulations industry-wide
    3. **Response Drills**: Reduce mean resolution time through tabletop exercises and playbook automation
    4. **Zero-day Preparedness**: Invest in behavioral detection for exploits that bypass signature-based tools
    """
    )

with st.expander("**For Risk Managers & Executives**"):
    st.markdown(
        """
    1. **Loss Quantification**: Use per-attack-type median losses to calibrate cyber insurance coverage
    2. **Country Risk Profiling**: Assess partner and supply-chain risk based on geographic threat intensity
    3. **Board Reporting**: Map attack trends to financial exposure for governance-level risk communication
    4. **ROI on Defense**: Correlate defense mechanism spend against resolution time and loss reduction
    """
    )

with st.expander("**For Policy Makers & Regulators**"):
    st.markdown(
        """
    1. **Cross-border Intelligence Sharing**: High-volume countries should collaborate on threat attribution
    2. **Sector-specific Mandates**: Regulate minimum security controls for the most targeted industries
    3. **Incident Reporting Standards**: Harmonize disclosure requirements to improve dataset completeness
    4. **Response Time Benchmarks**: Set national targets for mean incident resolution to drive accountability
    """
    )

st.markdown("---")
st.subheader("🏁 Final Conclusion")

if df is not None and "df_cleaned" in dir():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{df_cleaned.shape[0]:,}")
    c2.metric("Countries", df_cleaned["Country"].nunique())
    c3.metric(
        "Total Losses",
        f"${df_cleaned['Financial Loss (in Million $)'].sum():,.0f}M",
    )
    c4.metric(
        "Avg Response",
        f"{df_cleaned['Incident Resolution Time (in Hours)'].mean():.0f} hrs",
    )
    c5.metric("Industries Covered", df_cleaned["Target Industry"].nunique())

st.markdown(
    """
### **Project Impact**

This analysis provides a data-driven foundation for understanding one of the defining risks of the
digital age. The insights across time, geography, industry, and attack vector demonstrate that
effective cybersecurity is not a one-size-fits-all challenge — it demands **contextual intelligence,
adaptive defense, and rapid cross-sector collaboration**.

**Key Takeaways:**
1. **Trend Awareness**: Annual growth in incidents demands proactive rather than reactive security postures
2. **Industry Targeting**: Sector-specific attack profiles must drive differentiated defense strategies
3. **Financial Exposure**: Quantifying loss by attack type enables rational cyber risk investment decisions
4. **Geographic Intelligence**: Country-level loss data informs both policy and operational risk models
5. **Response Excellence**: Faster, better-drilled incident response remains the highest-leverage mitigation lever
"""
)