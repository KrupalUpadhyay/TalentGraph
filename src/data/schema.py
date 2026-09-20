from dataclasses import dataclass


@dataclass
class Candidate:
    candidate_id: str
    profile_text: str
    skills: list[str]
    experience_years: float
    education: str
    location: str


@dataclass
class Job:
    job_id: str
    title: str
    description: str
    skills: list[str]
    experience_required: float
    education_required: str
    location: str