"""Streamlit Data Analyst – fully self‑contained app
────────────────────────────────────────────────────────────────────────
Features
- Reads API keys from **Streamlit secrets** first, then environment vars
- Displays architecture & sequence diagrams (fails gracefully if missing)
- Upload CSV / Excel → lightweight EDA (summary + basic chart)
- Optionally invokes a CrewAI pipeline to generate deeper insights
"""

from __future__ import annotations

# ─── Standard Library ────────────────────────────────────────────────────────
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Third‑party ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image  # for safe image handling

# CrewAI stack
from crewai import Agent, Task, Crew, LLM

# ─── Environment helpers ────────────────────────────────────────────────────

def _get_key(var_name: str) -> str | None:
    """Return key from Streamlit secrets if present, else from env. """
    return st.secrets.get(var_name) if var_name in st.secrets else os.getenv(var_name)

# populate the variables CrewAI / OpenAI expect – even if empty (prevents KeyErrors)
os.environ["OPENAI_API_KEY"]     = _get_key("OPENAI_API_KEY") or ""
os.environ["ANTHROPIC_API_KEY"] = _get_key("ANTHROPIC_API_KEY") or ""
os.environ["GOOGLE_API_KEY"]    = _get_key("GOOGLE_API_KEY")  or ""
os.environ["GROQ_API_KEY"]      = _get_key("GROQ_API_KEY")    or ""

# ─── LLM options shown in the sidebar ───────────────────────────────────────
LLM_CONFIGS: Dict[str, Dict[str, List[str]]] = {
    "OpenAI":      {"models": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]},
    "Anthropic":   {"models": ["claude-3-sonnet-20240229", "claude-3-opus-20240229"]},
    "Groq":        {"models": ["Llama3-70B-8192", "Mixtral-8x22B"]},
}

# ─── UI helpers ─────────────────────────────────────────────────────────────

def _safe_image(path: Path, caption: str):
    """Render an image if it exists; otherwise show a friendly warning."""
    if path.exists():
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(f"⚠️ Missing image: **{path.name}** – add it to your repo or update `render_workflow_diagram()`. ")

def render_workflow_diagram() -> None:
    with st.expander("📖 System Workflow", expanded=False):
        img_dir = Path(__file__).parent / "images"
        _safe_image(img_dir / "data_analysis_architecture.png", "High‑level architecture")
        _safe_image(img_dir / "data_analysis_sequence_diagram.png", "Sequence diagram")

# ─── Data utilities ─────────────────────────────────────────────────────────

def _read_uploaded(file) -> pd.DataFrame:
    """Return a DataFrame from uploaded CSV or Excel file."""
    suffix = Path(file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file)
    elif suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type – upload CSV or Excel")

def _basic_eda(df: pd.DataFrame):
    """Show shape, null counts, and descriptive stats inline."""
    st.subheader("📊 Basic EDA")
    st.write("**Shape:**", df.shape)

    with st.expander("Null values by column"):
        st.write(df.isna().sum())

    with st.expander("Descriptive statistics"):
        st.write(df.describe())

    # example quick chart for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Distribution of first numeric column")
        fig = px.histogram(df, x=numeric_cols[0])
        st.plotly_chart(fig, use_container_width=True)

# ─── CrewAI pipeline (optional) ─────────────────────────────────────────────

def _run_crewai_analysis(df: pd.DataFrame, provider: str, model: str) -> str:
    """Run a very simple CrewAI workflow and return the generated report."""
    # 1️⃣  Set the LLM to use
    llm = LLM(provider=provider.lower(), model=model)

    # 2️⃣  Define Agents
    analyst = Agent(
        role="Data Analyst",
        goal="Generate a concise but insightful analysis of the provided dataset",
        llm=llm,
        backstory="You are a seasoned data analyst who explains things clearly for stakeholders with varied technical backgrounds."
    )

    # 3️⃣  Persist the dataset to a temporary CSV so the agent can "read" it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as tmp:
        df.to_csv(tmp.name, index=False)
        dataset_path = tmp.name

    # 4️⃣  Create Task and Crew
    task = Task(
        agent=analyst,
        description=(
            "Read the CSV located at '" + dataset_path + "' (it's small). "
            "Produce a JSON with keys: 'summary', 'interesting_findings', 'recommended_visuals'."
        ),
        expected_output="JSON"
    )
    crew = Crew(agents=[analyst], tasks=[task])

    # 5️⃣  Execute
    result: Dict[str, str] = json.loads(crew.run())  # type: ignore[arg-type]
    return (
        "### 📝 AI‑generated report\n" +
        result.get("summary", "") + "\n\n" +
        "**Interesting findings:** " + result.get("interesting_findings", "") + "\n\n" +
        "**Recommended visuals:** " + result.get("recommended_visuals", "")
    )

# ─── Main app ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="📈 Data‑Analyst", layout="wide")
    st.title("📈 Data‑Analyst – LLM‑powered exploratory tool")

    # Sidebar – choose LLM provider & model
    with st.sidebar:
        st.header("🔑 LLM settings")
        provider = st.selectbox("Provider", list(LLM_CONFIGS.keys()))
        model = st.selectbox("Model", LLM_CONFIGS[provider]["models"])
        run_ai = st.checkbox("Run AI insights with CrewAI", value=False)
        st.markdown("---")
        st.markdown("### ⚙️ App settings")
        st.info(
            "Add your API keys under **⋮ → Settings → Secrets**.\n"
            "Env vars will be used as fallback for local runs."
        )

    render_workflow_diagram()

    # File uploader
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    if not uploaded:
        st.warning("👆 Upload a dataset to get started.")
        return

    df = _read_uploaded(uploaded)
    st.success("Dataset loaded successfully!")
    st.dataframe(df.head())

    # Basic EDA always shown
    _basic_eda(df)

    # Optional CrewAI analysis
    if run_ai:
        with st.spinner("Calling CrewAI – this might take a minute…"):
            report_md = _run_crewai_analysis(df, provider, model)
        st.markdown(report_md, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
