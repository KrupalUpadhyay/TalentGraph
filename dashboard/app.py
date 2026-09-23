from __future__ import annotations

import os
from typing import Any, Dict

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

RECOMMEND_URL = (
    API_URL.rstrip("/")
    + "/recommend"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="TalentGraph",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .job-card {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 18px;
    }

    .score {
        font-size: 32px;
        font-weight: 800;
    }

    .muted {
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎯 TalentGraph</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Intelligent Talent Discovery & Workforce Intelligence Engine
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Candidate Profile")

    name = st.text_input(
        "Name",
        value="Candidate",
    )

    roles_text = st.text_input(
        "Target Roles",
        value="Machine Learning Engineer | Data Scientist",
        help="Separate multiple roles with |",
    )

    primary_skills_text = st.text_area(
        "Primary Skills",
        value="Python, Machine Learning, SQL",
        help="Separate skills with commas",
    )

    secondary_skills_text = st.text_area(
        "Secondary Skills",
        value="PyTorch, Scikit-Learn, FastAPI, Docker",
        help="Separate skills with commas",
    )

    experience = st.number_input(
        "Experience (years)",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=0.5,
    )

    seniority = st.selectbox(
        "Seniority",
        [
            "",
            "Intern",
            "Entry",
            "Junior",
            "Associate",
            "Mid",
            "Mid-level",
            "Senior",
            "Lead",
            "Principal",
            "Staff",
        ],
        index=0,
    )

    domains_text = st.text_input(
        "Domains",
        value="Artificial Intelligence | Data Science",
        help="Separate domains with |",
    )

    location = st.text_input(
        "Preferred Location",
        value="Bangalore",
    )

    preferences = st.text_input(
        "Preferences",
        value="",
    )

    career_intent = st.text_input(
        "Career Intent",
        value="Build ML and AI systems",
    )

    dealbreakers = st.text_input(
        "Dealbreakers",
        value="",
    )

    recommend_button = st.button(
        "🔎 Find Matching Jobs",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# HELPERS
# ============================================================

def comma_list(
    text: str,
):
    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def pipe_list(
    text: str,
):
    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def build_payload() -> Dict[str, Any]:

    return {
        "name": name,

        "roles": pipe_list(
            roles_text
        ),

        "skills_primary": comma_list(
            primary_skills_text
        ),

        "skills_secondary": comma_list(
            secondary_skills_text
        ),

        "experience_years": experience,

        "seniority": seniority,

        "domains": pipe_list(
            domains_text
        ),

        "preferences": preferences,

        "career_intent": career_intent,

        "dealbreakers": dealbreakers,

        "location": location,
    }


def feature_label(
    key: str,
) -> str:

    labels = {
        "bm25_score":
            "Keyword Retrieval",

        "dense_score":
            "Semantic Similarity",

        "skill_coverage":
            "Skill Coverage",

        "role_family_match":
            "Role Match",

        "domain_match":
            "Domain Match",

        "experience_compatibility":
            "Experience Compatibility",

        "seniority_compatibility":
            "Seniority Compatibility",

        "location_compatibility":
            "Location Compatibility",
    }

    return labels.get(
        key,
        key.replace("_", " ").title(),
    )


# ============================================================
# API HEALTH
# ============================================================

try:

    health_response = requests.get(
        API_URL.rstrip("/")
        + "/health",
        timeout=3,
    )

    if health_response.ok:

        health = (
            health_response.json()
        )

        with st.sidebar:

            st.success(
                "API Connected"
            )

            st.caption(
                f"Indexed jobs: "
                f"{health.get('jobs', 0)}"
            )

    else:

        with st.sidebar:
            st.warning(
                "API is reachable but unhealthy."
            )

except Exception:

    with st.sidebar:
        st.error(
            "API Offline"
        )

        st.caption(
            f"Start FastAPI at "
            f"{API_URL}"
        )


# ============================================================
# EMPTY STATE
# ============================================================

if not recommend_button:

    st.info(
        "Enter a candidate profile in the sidebar "
        "and click **Find Matching Jobs**."
    )

    st.markdown(
        """
        ### How TalentGraph works

        **Candidate Profile**

        ↓

        **BM25 + MiniLM Dense Retrieval**

        ↓

        **8 Ranking Features**

        ↓

        **XGBoost Learning-to-Rank**

        ↓

        **Top-K Job Recommendations**

        ↓

        **Deterministic Match Explanation**
        """
    )

    st.stop()


# ============================================================
# API REQUEST
# ============================================================

payload = build_payload()

with st.spinner(
    "Running TalentGraph inference..."
):

    try:

        response = requests.post(
            RECOMMEND_URL,
            json=payload,
            timeout=120,
        )

    except requests.exceptions.ConnectionError:

        st.error(
            f"""
            Could not connect to TalentGraph API.

            Start FastAPI with:

            `uvicorn src.api.main:app --reload`
            """
        )

        st.stop()

    except requests.exceptions.Timeout:

        st.error(
            "The API took too long to respond."
        )

        st.stop()


# ============================================================
# ERROR
# ============================================================

if not response.ok:

    try:
        error = response.json()
    except Exception:
        error = response.text

    st.error(
        f"TalentGraph API error: {error}"
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

data = response.json()

results = data.get(
    "results",
    [],
)

st.subheader(
    f"Top Matches for {data.get('candidate', name)}"
)

if not results:

    st.warning(
        "No matching jobs were returned."
    )

    st.stop()


st.caption(
    "Results generated by BM25 + MiniLM + "
    "XGBoost Learning-to-Rank"
)


# ============================================================
# RESULT CARDS
# ============================================================

for result in results:

    st.markdown(
        '<div class="job-card">',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [4, 1]
    )

    with left:

        title = result.get(
            "title",
            "Untitled Role",
        )

        company = result.get(
            "company",
            "Unknown Company",
        )

        location_value = result.get(
            "location",
            "",
        )

        seniority_value = result.get(
            "seniority",
            "",
        )

        domain_value = result.get(
            "domain",
            "",
        )

        st.markdown(
            f"### {result.get('rank', '')}. {title}"
        )

        st.write(
            f"**{company}**"
        )

        details = []

        if location_value:
            details.append(
                f"📍 {location_value}"
            )

        if seniority_value:
            details.append(
                f"👤 {seniority_value}"
            )

        if domain_value:
            details.append(
                f"🏷️ {domain_value}"
            )

        if details:
            st.caption(
                "  •  ".join(details)
            )

    with right:

        score = result.get(
            "match_score",
            0,
        )

        st.metric(
            "Match Score",
            f"{score:.1f}%",
        )

    # --------------------------------------------------------
    # Matched Skills
    # --------------------------------------------------------

    matched = result.get(
        "matched_skills",
        [],
    )

    if matched:

        st.markdown(
            "**Matched Skills**"
        )

        st.write(
            " · ".join(matched)
        )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reasons = result.get(
        "reasons",
        [],
    )

    if reasons:

        st.markdown(
            "**Why this match?**"
        )

        for reason in reasons:

            st.write(
                f"✓ {reason}"
            )

    # --------------------------------------------------------
    # Feature Breakdown
    # --------------------------------------------------------

    features = result.get(
        "features",
        {},
    )

    if features:

        with st.expander(
            "View ranking feature breakdown"
        ):

            feature_items = list(
                features.items()
            )

            # Four columns
            columns = st.columns(4)

            for index, (
                key,
                value,
            ) in enumerate(
                feature_items
            ):

                column = columns[
                    index % 4
                ]

                with column:

                    st.metric(
                        feature_label(key),
                        f"{float(value):.1f}%",
                    )

    # --------------------------------------------------------
    # Job Link
    # --------------------------------------------------------

    url = result.get(
        "url",
        "",
    )

    if url and url.startswith(
        ("http://", "https://")
    ):

        st.link_button(
            "View Job",
            url,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )