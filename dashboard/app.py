import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="TalentGraph",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# Header
# ============================================================

st.title(
    "TalentGraph"
)

st.caption(
    "Intelligent Talent Discovery & Workforce Intelligence Engine"
)

st.markdown(
    """
    **Hybrid retrieval + Learning-to-Rank**

    TalentGraph combines lexical retrieval, semantic similarity,
    candidate-job compatibility features, and XGBoost ranking.
    """
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "Candidate Profile"
)

api_url = st.sidebar.text_input(
    "API URL",
    "http://127.0.0.1:8000",
)


# ============================================================
# Candidate input
# ============================================================

profile_text = st.text_area(
    "Candidate profile",

    """
    M.Tech student in Computer Engineering and
    Cyber Physical Systems focused on machine learning,
    computer vision, information retrieval and AI systems.
    """.strip(),

    height=140,
)


skills_text = st.text_input(
    "Skills",

    (
        "Python, C++, SQL, Machine Learning, "
        "Deep Learning, NLP, Computer Vision, "
        "FastAPI, Docker"
    ),
)

roles_text = st.text_input("Target roles (optional)", "Machine Learning Engineer, Data Scientist")
domains_text = st.text_input("Domains (optional)", "AI, Machine Learning")
seniority = st.selectbox("Seniority", ["", "Intern", "Entry", "Junior", "Associate", "Mid-level", "Senior", "Lead"])


experience = st.number_input(
    "Experience (years)",

    min_value=0.0,

    max_value=20.0,

    value=1.0,

    step=0.5,
)


education = st.text_input(
    "Education",
    "M.Tech",
)


location = st.text_input(
    "Preferred location",
    "India",
)


top_k = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
)


# ============================================================
# Recommendation
# ============================================================

if st.button(
    "Find matching jobs",
    type="primary",
):

    skills = [
        skill.strip()
        for skill
        in skills_text.split(",")
        if skill.strip()
    ]

    payload = {

        "profile_text":
            profile_text,

        "skills":
            skills,

        "roles": [value.strip() for value in roles_text.split(",") if value.strip()],

        "domains": [value.strip() for value in domains_text.split(",") if value.strip()],

        "seniority": seniority,

        "experience_years":
            experience,

        "education":
            education,

        "location":
            location,
    }

    try:

        response = requests.post(
            f"{api_url}/recommend",
            params={
                "top_k": top_k,
            },
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        st.success(
            f"Found {data['count']} recommendations."
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        for rank, job in enumerate(
            data["recommendations"],
            start=1,
        ):

            with st.container(
                border=True
            ):

                st.subheader(
                    f"{rank}. "
                    f"{job['title']}"
                )

                st.write(
                    f"**{job['company']}** "
                    f"· {job['location']}"
                )

                st.metric(
                    "LTR Ranking Score",
                    job["score"],
                )

                if job.get("matched_skills"):
                    st.caption("Matched skills: " + ", ".join(job["matched_skills"]))
                st.caption(job.get("explanation", ""))

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Skill Coverage",
                    job[
                        "skill_coverage"
                    ],
                )

                col2.metric(
                    "Semantic Score",
                    job[
                        "semantic_score"
                    ],
                )

                col3.metric(
                    "Experience",
                    job[
                        "experience_compatibility"
                    ],
                )

                col4.metric(
                    "Location",
                    job[
                        "location_compatibility"
                    ],
                )

    except Exception as error:

        st.error(
            f"API request failed: {error}"
        )
