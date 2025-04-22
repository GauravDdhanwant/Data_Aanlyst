#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# 1) SQLITE HOT‑FIX (required on Streamlit Cloud)
###############################################################################
try:
    import pysqlite3, sys              # bundled with SQLite ≥ 3.45
    sys.modules["sqlite3"] = pysqlite3  # patch stdlib before anything else
except ModuleNotFoundError:
    # Local dev machine already has a modern SQLite → nothing to patch
    pass

###############################################################################
# 2) STANDARD LIBS & DATA LIBS
###############################################################################
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

###############################################################################
# 3) CREWAI & LLM
###############################################################################
from crewai import Agent, Task, Crew, LLM

###############################################################################
# 4) GLOBAL CONFIG
###############################################################################
st.set_page_config(page_title="Data Analysis Assistant", page_icon="📊", layout="wide")

LLM_CONFIGS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini"]
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20241022",
                   "claude-3-5-haiku-20241022",
                   "claude-3-opus-20240229"]
    },
    "Gemini": {
        "models": ["gemini-2.0-flash-exp",
                   "gemini-1.5-flash",
                   "gemini-1.5-flash-8b",
                   "gemini-1.5-pro"]
    },
    "Groq": {
        "models": ["groq/deepseek-r1-distill-llama-70b",
                   "groq/llama3-70b-8192",
                   "groq/llama-3.1-8b-instant",
                   "groq/llama-3.3-70b-versatile",
                   "groq/gemma2-9b-it",
                   "groq/mixtral-8x7b-32768"]
    }
}

###############################################################################
# 5) BUSINESS LOGIC
###############################################################################
class DataAnalyzer:
    """Builds CrewAI agents, creates tasks and runs the analysis workflow."""

    def __init__(self, llm_provider: str, model_name: str):
        self.llm_provider = llm_provider
        self.model_name = model_name

    # ------------------------------------------------------------------ #
    # 5‑A  DATA SUMMARY                                                   #
    # ------------------------------------------------------------------ #
    def get_data_context(self, data_path: str) -> str:
        """Return a human‑readable summary of the dataset for prompt‑engineering."""
        df = pd.read_csv(data_path)

        context = (
            f"\nDataset Summary:\n"
            f"- Total Records: {len(df)}\n"
            f"- Columns: {', '.join(df.columns.tolist())}\n\n"
            f"Column Information:\n"
        )

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                context += (
                    f"\n{col}:\n"
                    f"- Type: Numeric\n"
                    f"- Range: {df[col].min()} to {df[col].max()}\n"
                    f"- Average: {df[col].mean():.2f}\n"
                    f"- Missing Values: {df[col].isnull().sum()}\n"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                context += (
                    f"\n{col}:\n"
                    f"- Type: Date/Time\n"
                    f"- Range: {df[col].min()} to {df[col].max()}\n"
                    f"- Missing Values: {df[col].isnull().sum()}\n"
                )
            else:
                context += (
                    f"\n{col}:\n"
                    f"- Type: Categorical\n"
                    f"- Unique Values: {df[col].nunique()}\n"
                    f"- Top Values: {', '.join(df[col].value_counts().nlargest(3).index.astype(str))}\n"
                    f"- Missing Values: {df[col].isnull().sum()}\n"
                )
        return context

    # ------------------------------------------------------------------ #
    # 5‑B  LLM INITIALISER                                               #
    # ------------------------------------------------------------------ #
    def initialize_llm(self, llm_provider: str, model_name: str) -> LLM:
        """Return an LLM instance with credentials pulled from env vars / Streamlit secrets."""
        # Map provider → expected environment variable
        provider_keys = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Gemini": "GOOGLE_API_KEY",
            "Groq": "GROQ_API_KEY",
        }

        key_name = provider_keys.get(llm_provider)
        if key_name and key_name not in os.environ:
            raise RuntimeError(
                f"Environment variable '{key_name}' is not set. "
                "Store it as a Git/Streamlit secret."
            )

        return LLM(
            model=model_name,
            temperature=0.7,
            timeout=120,
            max_tokens=4000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

    # ------------------------------------------------------------------ #
    # 5‑C  CREW FACTORY                                                 #
    # ------------------------------------------------------------------ #
    def create_crew(self, data_path: str) -> Crew:
        llm = self.initialize_llm(self.llm_provider, self.model_name)
        data_context = self.get_data_context(data_path)

        # ---------- Agent definitions ----------
        data_analyst = Agent(
            role="Data Analyst",
            goal="Provide comprehensive statistical analysis with actionable insights",
            backstory=(
                "Expert data analyst with strong statistical background and "
                "business intelligence expertise."
            ),
            llm=llm,
            verbose=True,
        )

        insights_generator = Agent(
            role="Business Intelligence Specialist",
            goal="Generate strategic business insights and recommendations",
            backstory=(
                "Senior business analyst experienced in converting data patterns "
                "into actionable strategies."
            ),
            llm=llm,
            verbose=True,
        )

        # ---------- Task definitions ----------
        analysis_task = Task(
            description=f"""
            Analyze the provided dataset using this context:
            {data_context}

            Deliver:
            1. Key Statistical Findings
            2. Data Distribution Analysis
            3. Relationship Analysis
            4. Anomaly detection
            """,
            expected_output="""
            Detailed analysis report with:
            • Statistical summary
            • Pattern analysis
            • Relationship insights
            • Outlier details
            """,
            agent=data_analyst,
        )

        insights_task = Task(
            description=f"""
            Using the analysis and context below:
            {data_context}

            Produce business insights:
            • Key findings affecting performance
            • Actionable recommendations
            • Implementation roadmap
            • Risk assessment
            """,
            expected_output="""
            Comprehensive insights report with:
            • Data‑backed findings
            • Prioritised recommendations
            • Short‑ & long‑term actions
            • Risk mitigation
            """,
            agent=insights_generator,
        )

        return Crew(
            agents=[data_analyst, insights_generator],
            tasks=[analysis_task, insights_task],
            verbose=True,
        )

###############################################################################
# 6) UTILITY FUNCTIONS
###############################################################################
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cast nullable Int64 to int64 and attempt to parse date‑like columns."""
    for col in df.select_dtypes(include=["Int64"]).columns:
        df[col] = df[col].astype("int64")

    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


def create_visualizations(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Return a dict of Plotly figures keyed by name."""
    visualizations: Dict[str, go.Figure] = {}
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    try:
        # Histograms
        for col in numeric_cols:
            visualizations[f"{col}_hist"] = px.histogram(
                df, x=col, title=f"Distribution of {col}"
            )

        # Bar‑chart counts
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            visualizations[f"{col}_bar"] = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {col}",
            )

        # Correlation heatmap
        if len(numeric_cols) > 1:
            visualizations["correlation"] = px.imshow(
                df[numeric_cols].corr(),
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
            )
    except Exception as e:
        st.warning(f"Some visualizations failed: {e}")

    return visualizations


def display_dataframe_info(df: pd.DataFrame) -> None:
    """Interactive preview + quality metrics."""
    with st.expander("📊 Data Preview", expanded=True):
        column_config = {
            col: st.column_config.Column(
                width="auto",
                help=f"Type: {df[col].dtype}",
            )
            for col in df.columns
        }
        st.dataframe(df.head(10), use_container_width=True, column_config=column_config)

    st.subheader("📈 Dataset Overview")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("**Dimensions**")
        st.info(
            f"• Rows: {df.shape[0]:,}\n"
            f"• Columns: {df.shape[1]}\n"
            f"• Memory: {df.memory_usage().sum()/1024**2:.2f} MB"
        )
    with col2:
        st.markdown("**Data Quality**")
        total_missing = df.isnull().sum().sum()
        pct = total_missing / (df.size) * 100
        st.info(
            f"• Missing cells: {total_missing:,} ({pct:.2f}%)\n"
            f"• Duplicate rows: {df.duplicated().sum():,}"
        )
    with col3:
        st.markdown("**Column Types**")
        st.info(
            "\n".join(f"• {k}: {v}" for k, v in df.dtypes.value_counts().items())
        )

    # Column details
    with st.expander("🔍 Detailed Column Info", expanded=False):
        details = []
        for col in df.columns:
            miss = df[col].isnull().sum()
            pct = miss / len(df) * 100
            info = {
                "Column": col,
                "Type": str(df[col].dtype),
                "Unique": df[col].nunique(),
                "Missing": f"{miss} ({pct:.1f}%)",
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                info.update(
                    {
                        "Min": df[col].min(),
                        "Max": df[col].max(),
                        "Mean": round(df[col].mean(), 2),
                    }
                )
            details.append(info)
        st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)

    if (num := df.select_dtypes(include=["int64", "float64"]).columns).any():
        with st.expander("📊 Numerical Statistics", expanded=False):
            st.dataframe(df[num].describe().round(2), use_container_width=True)


def render_workflow_diagram() -> None:
    with st.expander("📖 System Workflow", expanded=False):
        current_dir = Path(__file__).parent
        img_dir = current_dir / "images"
        st.image(img_dir / "data_analysis_architecture.png", caption="High‑level Architecture")
        st.image(img_dir / "data_analysis_sequence_diagram.png", caption="Sequence Diagram")

###############################################################################
# 7) MAIN APP
###############################################################################
def main() -> None:
    st.header("📊 Data Analysis Assistant")

    # 7‑A  Provider / model selection
    with st.sidebar:
        st.header("⚙️ LLM Configuration")
        provider = st.selectbox("Provider", list(LLM_CONFIGS.keys()))
        model = st.selectbox("Model", LLM_CONFIGS[provider]["models"])

    render_workflow_diagram()

    # 7‑B  Upload
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            df = preprocess_dataframe(df)
            display_dataframe_info(df)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return
    else:
        st.stop()

    # 7‑C  Analyse
    if st.button("Start Analysis"):
        with st.spinner("Analysing…"):
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / "data.csv"
            df.to_csv(temp_path, index=False)

            try:
                crew = DataAnalyzer(provider, model).create_crew(str(temp_path))
                result = crew.kickoff()
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                    Path(temp_dir).rmdir()
                except Exception:
                    pass

        st.success("Analysis complete!")

        # ----- Tabs -----
        viz_tab, analysis_tab = st.tabs(["📈 Visualisations", "📊 Analysis"])

        with viz_tab:
            for fig in create_visualizations(df).values():
                st.plotly_chart(fig, use_container_width=True)

        with analysis_tab:
            st.markdown(str(result))

        # ----- Download -----
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": str(result),
            "dataset_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_types": {c: str(t) for c, t in df.dtypes.items()},
            },
        }
        st.download_button(
            "Download JSON report",
            data=pd.json.dumps(report, indent=2),
            file_name="analysis_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
