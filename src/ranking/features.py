import re


def normalize(text):
    text = str(text).lower()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9+#.\-/ ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_candidate_skills(row):

    primary = str(
        row["candidate_primary_skills"]
    )

    secondary = str(
        row["candidate_secondary_skills"]
    )

    skills = []

    for value in [primary, secondary]:

        if value and value != "nan":

            skills.extend(
                [
                    normalize(x)
                    for x in value.split("|")
                    if x.strip()
                ]
            )

    return list(set(skills))


def skill_matches(skill, job_text):

    skill = normalize(skill)
    job_text = normalize(job_text)

    if not skill:
        return False

    # Exact phrase match first.
    if skill in job_text:
        return True

    # Handle simple variants such as:
    # python / python3
    # node.js / node
    if skill.replace(".", "") in job_text.replace(".", ""):
        return True

    return False



def skill_coverage(row):

    skills = get_candidate_skills(row)

    if not skills:
        return 0.0

    job_text = normalize(
        row["job_text"]
    )

    matched = sum(
        skill_matches(
            skill,
            job_text
        )
        for skill in skills
    )

    return matched / len(skills)


def role_family_match(row):

    candidate_roles = normalize(
        row["candidate_roles"]
    )

    job_role = normalize(
        row["job_role_family"]
    )

    if not candidate_roles or not job_role:
        return 0.0

    roles = [
        x.strip()
        for x in candidate_roles.split("|")
    ]

    for role in roles:

        if role == job_role:
            return 1.0

        if role in job_role:
            return 1.0

        if job_role in role:
            return 1.0

    return 0.0


def domain_match(row):

    candidate_domains = [
        normalize(x)
        for x in str(
            row["candidate_domains"]
        ).split("|")
        if x.strip()
    ]

    job_domain = normalize(
        row["job_domain"]
    )

    if not candidate_domains or not job_domain:
        return 0.0

    for domain in candidate_domains:

        if domain == job_domain:
            return 1.0

        if domain in job_domain:
            return 1.0

        if job_domain in domain:
            return 1.0

    return 0.0


def experience_compatibility(row):

    candidate_exp = float(
        row["candidate_experience"]
    )

    job_exp = float(
        row["job_experience"] or 0
    )

    if job_exp <= 0:
        return 1.0

    # Fully satisfies requirement.
    if candidate_exp >= job_exp:
        return 1.0

    gap = job_exp - candidate_exp

    # Smooth penalty.
    return max(
        0.0,
        1.0 - gap / max(job_exp, 1.0)
    )


def seniority_to_level(value):

    value = normalize(value)

    mapping = {
        "intern": 0,
        "entry": 1,
        "junior": 1,
        "associate": 2,
        "mid": 2,
        "mid-level": 2,
        "senior": 3,
        "lead": 4,
        "principal": 5,
        "staff": 5,
    }

    for key, level in mapping.items():

        if key in value:
            return level

    return None


def seniority_compatibility(row):

    candidate_level = seniority_to_level(
        row["candidate_seniority"]
    )

    job_level = seniority_to_level(
        row["job_seniority"]
    )

    if (
        candidate_level is None
        or job_level is None
    ):
        return 0.0

    difference = (
        candidate_level - job_level
    )

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.75

    if difference == -1:
        return 0.50

    if difference > 1:
        return 0.50

    return 0.0


def extract_locations(candidate_text):

    text = normalize(
        candidate_text
    )

    # Common Indian location vocabulary
    locations = [
        "bangalore",
        "bengaluru",
        "mumbai",
        "delhi",
        "new delhi",
        "gurgaon",
        "gurugram",
        "noida",
        "hyderabad",
        "pune",
        "chennai",
        "kolkata",
        "ahmedabad",
        "jaipur",
        "jodhpur",
        "indore",
        "chandigarh",
        "kochi",
        "remote",
    ]

    return {
        location
        for location in locations
        if location in text
    }


def location_compatibility(row):

    candidate_locations = (
        extract_locations(
            row["candidate_text"]
        )
    )

    job_location = normalize(
        row["job_location"]
    )

    if not candidate_locations:
        return 0.0

    if "remote" in job_location:
        return 1.0

    for location in candidate_locations:

        if location in job_location:
            return 1.0

    return 0.0


def build_features(row):

    return {
        "bm25_score": float(
            row["bm25_score"]
        ),

        "dense_score": float(
            row["dense_score"]
        ),

        "skill_coverage": skill_coverage(
            row
        ),

        "role_family_match":
            role_family_match(row),

        "domain_match":
            domain_match(row),

        "experience_compatibility":
            experience_compatibility(row),

        "seniority_compatibility":
            seniority_compatibility(row),

        "location_compatibility":
            location_compatibility(row),
    }