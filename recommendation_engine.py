"""
recommendation_engine.py
------------------------
Generates personalised career/placement recommendations based on a student's
metrics, skills, job-role matches, and project quality.

Rule-based, explainable, and fully local.
"""

import re
from typing import List, Dict, Any

# =================================================================
# CONFIGURABLE THRESHOLDS
# =================================================================

# Coding
CODING_STRONG_THRESHOLD = 80
CODING_GOOD_THRESHOLD = 70
CODING_MODERATE_THRESHOLD = 70
CODING_LOW_THRESHOLD = 50

# Aptitude
APTITUDE_STRONG_THRESHOLD = 80
APTITUDE_GOOD_THRESHOLD = 70
APTITUDE_MODERATE_THRESHOLD = 70
APTITUDE_LOW_THRESHOLD = 50

# CGPA
CGPA_STRONG_THRESHOLD = 8.0
CGPA_GOOD_THRESHOLD = 7.0
CGPA_MODERATE_THRESHOLD = 7.5
CGPA_LOW_THRESHOLD = 6.5

# Attendance
ATTENDANCE_EXCELLENT_THRESHOLD = 85
ATTENDANCE_GOOD_THRESHOLD = 75
ATTENDANCE_MODERATE_THRESHOLD = 75
ATTENDANCE_LOW_THRESHOLD = 60

# Internships
INTERNSHIPS_MULTIPLE = 2
INTERNSHIPS_ONE = 1

# Project Score
PROJECT_STRONG_THRESHOLD = 70
PROJECT_GOOD_THRESHOLD = 55
PROJECT_MODERATE_THRESHOLD = 60
PROJECT_LOW_THRESHOLD = 40
PROJECT_IMPROVE_THRESHOLD = 75

# Placement Probability
PLACEMENT_HIGH_THRESHOLD = 75
PLACEMENT_MODERATE_THRESHOLD = 60
PLACEMENT_LOW_THRESHOLD = 40

# Job role match
JOB_ROLE_HIGH_MATCH = 70

MAX_RECOMMENDATIONS = 7

# =================================================================

# -------------------------------------------------------------------
# Static mapping of common job roles → typical required skills
# -------------------------------------------------------------------
ROLE_SKILLS = {
    "Data Analyst": [
        "SQL", "Data Visualization", "Tableau", "Power BI", "Excel",
        "Statistics", "Data Cleaning", "Python", "Communication"
    ],
    "Machine Learning Engineer": [
        "Deep Learning", "Model Deployment", "MLOps", "Cloud",
        "PyTorch", "TensorFlow", "Data Pipelines", "Scalability"
    ],
    "Artificial Intelligence Engineer": [
        "Deep Learning", "Python", "Machine Learning", "TensorFlow",
        "PyTorch", "Neural Networks", "Model Deployment", "NLP"
    ],
    "Data Scientist": [
        "Statistics", "Python", "R", "Machine Learning", "Data Visualization",
        "Big Data", "SQL", "Feature Engineering", "Experimentation"
    ],
    "Software Developer": [
        "Programming", "Git", "Agile", "Data Structures",
        "Algorithms", "Web Frameworks", "SQL", "Unit Testing"
    ],
    "Cloud Engineer": [
        "AWS", "Azure", "GCP", "Terraform", "Docker", "Kubernetes",
        "CI/CD", "Networking", "Linux", "Scripting"
    ],
    "DevOps Engineer": [
        "Docker", "Kubernetes", "Jenkins", "Git", "CI/CD", "Linux",
        "Scripting", "Monitoring", "Cloud"
    ],
    "Computer Vision Engineer": [
        "Computer Vision", "Deep Learning", "Python", "OpenCV",
        "TensorFlow", "PyTorch", "Image Processing", "CNNs"
    ],
    "Product Manager": [
        "Product Strategy", "Market Research", "Data Analysis", "Communication",
        "Project Management", "Agile", "User Experience", "Roadmapping"
    ],
    "Business Analyst": [
        "Data Analysis", "SQL", "Excel", "Tableau", "Business Process",
        "Requirements Gathering", "Stakeholder Management", "Communication"
    ]
}

# -------------------------------------------------------------------
# Synonym mapping: alternative names for the same skill
# -------------------------------------------------------------------
SKILL_SYNONYMS = {
    "git": ["github", "gitlab", "version control"],
    "github": ["git", "gitlab", "version control"],
    "programming": ["python", "java", "c++", "c", "javascript", "coding"],
    "python": ["programming", "coding"],
    "machine learning": ["ml", "deep learning", "ai"],
    "deep learning": ["dl", "machine learning", "neural networks", "ai"],
    "artificial intelligence": ["ai", "ml", "machine learning"],
    "sql": ["mysql", "postgresql", "database", "db"],
    "data visualization": ["visualization", "plotly", "matplotlib", "tableau", "power bi"],
    "nlp": ["natural language processing", "text mining"],
    "computer vision": ["cv", "image processing", "opencv"],
    "cloud": ["aws", "azure", "gcp", "cloud computing"],
    "aws": ["cloud", "amazon web services"],
    "docker": ["containerization", "containers"],
    "kubernetes": ["k8s", "container orchestration"],
    "data analysis": ["data analytics", "analytics", "eda"],
}


# -------------------------------------------------------------------
# Helper to add recommendations with duplicate prevention
# -------------------------------------------------------------------
def _add_rec(rec_list, title, reason, priority, existing_titles):
    if title in existing_titles:
        return
    rec_list.append({
        "title": title,
        "reason": reason,
        "priority": priority
    })
    existing_titles.add(title)


# -------------------------------------------------------------------
# Fuzzy skill matching helpers
# -------------------------------------------------------------------
def _normalize_skill(skill: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    s = skill.lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _skill_matches(student_skill_set: set, student_keyword_set: set, required_skill: str) -> bool:
    """
    Check if student has a skill matching the required skill.
    Uses exact match, keyword match, and synonym match.
    """
    req_norm = _normalize_skill(required_skill)
    
    # 1. Exact normalized match
    if req_norm in student_skill_set:
        return True
    
    # 2. Check synonyms
    synonyms = SKILL_SYNONYMS.get(req_norm, [])
    for syn in synonyms:
        if syn in student_skill_set:
            return True
    
    # 3. Keyword match - break required skill into words
    req_words = [w for w in req_norm.split() if len(w) > 2]
    
    # If any meaningful word from required skill is in student's skills, consider matched
    if req_words:
        matched_words = sum(1 for w in req_words if w in student_keyword_set)
        # If at least half of the words match, consider it learned
        if matched_words >= max(1, len(req_words) // 2):
            return True
    
    return False


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def generate_recommendations(
    placement_probability: float,
    skills: List[str],
    job_role_matches: List[Dict[str, Any]],
    project_score: float,
    internships: int,
    coding_score: float,
    aptitude_score: float,
    cgpa: float,
    attendance: float
) -> Dict[str, Any]:
    """
    Generate personalized recommendations.
    """
    recommendations = []
    seen_titles = set()
    strengths = []

    # ---------- STRENGTHS ----------
    if cgpa >= CGPA_STRONG_THRESHOLD:
        strengths.append("Strong CGPA")
    elif cgpa >= CGPA_GOOD_THRESHOLD:
        strengths.append("Good CGPA")

    if coding_score >= CODING_STRONG_THRESHOLD:
        strengths.append("Strong coding skills")
    elif coding_score >= CODING_GOOD_THRESHOLD:
        strengths.append("Good coding skills")

    if aptitude_score >= APTITUDE_STRONG_THRESHOLD:
        strengths.append("Strong aptitude")
    elif aptitude_score >= APTITUDE_GOOD_THRESHOLD:
        strengths.append("Good aptitude")

    if attendance >= ATTENDANCE_EXCELLENT_THRESHOLD:
        strengths.append("Excellent attendance")
    elif attendance >= ATTENDANCE_GOOD_THRESHOLD:
        strengths.append("Good attendance")

    if internships >= INTERNSHIPS_MULTIPLE:
        strengths.append("Multiple internships")
    elif internships == INTERNSHIPS_ONE:
        strengths.append("Has internship experience")

    if project_score >= PROJECT_STRONG_THRESHOLD:
        strengths.append("Strong project quality")
    elif project_score >= PROJECT_GOOD_THRESHOLD:
        strengths.append("Good project quality")

    if placement_probability >= PLACEMENT_HIGH_THRESHOLD:
        strengths.append("High placement probability")

    # ---------- RECOMMENDATIONS ----------
    # 1. Coding
    if coding_score < CODING_MODERATE_THRESHOLD:
        if coding_score < CODING_LOW_THRESHOLD:
            _add_rec(recommendations, "Improve coding skills",
                     "Coding score is very low; focus on DSA and practice on LeetCode/HackerRank.",
                     "High", seen_titles)
        else:
            _add_rec(recommendations, "Enhance coding skills",
                     f"Coding score is moderate. Aim for {CODING_MODERATE_THRESHOLD}+ to improve placement chances.",
                     "Medium", seen_titles)

    # 2. Aptitude
    if aptitude_score < APTITUDE_MODERATE_THRESHOLD:
        if aptitude_score < APTITUDE_LOW_THRESHOLD:
            _add_rec(recommendations, "Improve aptitude skills",
                     "Aptitude score is low; practice quantitative and logical reasoning regularly.",
                     "High", seen_titles)
        else:
            _add_rec(recommendations, "Enhance aptitude skills",
                     "Aptitude score is moderate. Practice more to boost placement test performance.",
                     "Medium", seen_titles)

    # 3. Internships
    if internships == 0:
        _add_rec(recommendations, "Gain internship experience",
                 "You have no internship experience. Internships significantly boost placement chances.",
                 "High", seen_titles)
    elif internships == 1:
        _add_rec(recommendations, "Complete at least one more internship",
                 "One internship is good, but 2+ internships make your resume stand out.",
                 "Medium", seen_titles)

    # 4. Attendance
    if attendance < ATTENDANCE_MODERATE_THRESHOLD:
        priority = "High" if attendance < ATTENDANCE_LOW_THRESHOLD else "Medium"
        _add_rec(recommendations, "Improve attendance",
                 f"Current attendance {attendance:.0f}%. Maintain above {ATTENDANCE_MODERATE_THRESHOLD}% for better placement eligibility.",
                 priority, seen_titles)

    # 5. CGPA
    if cgpa < CGPA_MODERATE_THRESHOLD:
        priority = "High" if cgpa < CGPA_LOW_THRESHOLD else "Medium"
        _add_rec(recommendations, "Improve CGPA",
                 f"Current CGPA {cgpa:.1f}. Aim for {CGPA_MODERATE_THRESHOLD}+ to increase shortlisting chances.",
                 priority, seen_titles)

    # 6. Project quality
    if project_score < PROJECT_MODERATE_THRESHOLD:
        if project_score < PROJECT_LOW_THRESHOLD:
            _add_rec(recommendations, "Rebuild project with real-world focus",
                     "Project quality is low. Add real-world data, deployment, or ML/AI components.",
                     "High", seen_titles)
        else:
            _add_rec(recommendations, "Enhance project quality",
                     "Project is decent but could be improved. Consider adding deployment, documentation, or scalability.",
                     "Medium", seen_titles)
    elif project_score < PROJECT_IMPROVE_THRESHOLD:
        _add_rec(recommendations, "Add deployment and documentation",
                 "Your project is good. Adding deployment (e.g., Streamlit/Flask) and proper documentation will make it more impressive.",
                 "Medium", seen_titles)

    # 7. Missing skills for high-matching job roles
    high_match_roles = [role for role in job_role_matches if role.get("match_score", 0) >= JOB_ROLE_HIGH_MATCH]
    
    if high_match_roles:
        # Build student skill sets (normalized + keywords)
        student_skill_set = set()
        student_keyword_set = set()
        for s in skills:
            norm = _normalize_skill(s)
            student_skill_set.add(norm)
            # Add individual words as keywords
            for word in norm.split():
                if len(word) > 2:
                    student_keyword_set.add(word)
            # Add synonym expansions
            for syn in SKILL_SYNONYMS.get(norm, []):
                student_skill_set.add(syn)
                for word in syn.split():
                    if len(word) > 2:
                        student_keyword_set.add(word)

        # Track skills already suggested to avoid duplicates
        suggested_skills = set()

        for role in high_match_roles:
            role_title = role.get("role", "")
            
            # Find matching role key in ROLE_SKILLS
            matched_role_key = None
            for key in ROLE_SKILLS:
                if role_title.lower() == key.lower():
                    matched_role_key = key
                    break
            
            # If not exact, try partial match
            if not matched_role_key:
                for key in ROLE_SKILLS:
                    if key.lower() in role_title.lower() or role_title.lower() in key.lower():
                        matched_role_key = key
                        break
            
            if matched_role_key:
                required_skills = ROLE_SKILLS[matched_role_key]
                missing = []
                
                for req in required_skills:
                    req_norm = _normalize_skill(req)
                    if req_norm in suggested_skills:
                        continue
                    if not _skill_matches(student_skill_set, student_keyword_set, req):
                        missing.append(req)
                
                # Suggest up to 2 missing skills per role
                for m in missing[:2]:
                    title = f"Learn {m}"
                    reason = f"'{m}' is commonly required for {role_title} roles."
                    _add_rec(recommendations, title, reason, "Low", seen_titles)
                    suggested_skills.add(_normalize_skill(m))

    # 8. Low placement probability
    if placement_probability < PLACEMENT_MODERATE_THRESHOLD:
        priority = "High" if placement_probability < PLACEMENT_LOW_THRESHOLD else "Medium"
        if not any(r["title"] in ["Improve coding skills", "Enhance coding skills",
                                   "Improve aptitude skills", "Enhance aptitude skills"]
                   for r in recommendations):
            _add_rec(recommendations, "Create a comprehensive improvement plan",
                     f"Your placement probability is {placement_probability:.0f}%. Focus on improving your weakest areas: coding, aptitude, internships, and projects.",
                     priority, seen_titles)

    # 9. Positive note if no recommendations
    if not recommendations:
        _add_rec(recommendations, "Maintain current momentum",
                 "You're on the right track. Keep practicing and consider advanced topics.",
                 "Low", seen_titles)

    # ---------- FINALISE ----------
    final_recs = []
    seen = set()
    for rec in recommendations:
        if rec["title"] not in seen:
            seen.add(rec["title"])
            final_recs.append(rec)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    final_recs.sort(key=lambda x: (priority_order.get(x["priority"], 2), x["title"]))

    overall_priority = "High" if any(r["priority"] == "High" for r in final_recs) else \
                       "Medium" if any(r["priority"] == "Medium" for r in final_recs) else "Low"

    if not strengths:
        strengths = ["No specific strengths identified – keep working!"]

    return {
        "priority": overall_priority,
        "recommendations": final_recs[:MAX_RECOMMENDATIONS],
        "strengths": strengths
    }


# -------------------------------------------------------------------
# Test Section
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = {
        "placement_probability": 98.8,
        "skills": ["Python", "SQL", "Deep Learning", "Machine Learning",
                   "Data Visualization", "NumPy", "Scikit-learn",
                   "GitHub", "C++", "Data Analysis", "data science", "llm"],
        "job_role_matches": [
            {"role": "software developer", "match_score": 100.0, "matched_skills": ["c++", "github", "python"]},
            {"role": "artificial intelligence engineer", "match_score": 97.0, "matched_skills": ["deep learning", "c++", "python"]},
            {"role": "computer vision engineer", "match_score": 97.0, "matched_skills": ["deep learning", "machine learning", "python"]},
            {"role": "ICT application developer", "match_score": 85.0, "matched_skills": ["c++", "github", "python"]},
            {"role": "data warehouse designer", "match_score": 85.0, "matched_skills": ["c++", "python", "data analysis"]}
        ],
        "project_score": 86.0,
        "internships": 2,
        "coding_score": 78.0,
        "aptitude_score": 78.0,
        "cgpa": 8.45,
        "attendance": 85.0
    }

    result = generate_recommendations(
        placement_probability=sample_data["placement_probability"],
        skills=sample_data["skills"],
        job_role_matches=sample_data["job_role_matches"],
        project_score=sample_data["project_score"],
        internships=sample_data["internships"],
        coding_score=sample_data["coding_score"],
        aptitude_score=sample_data["aptitude_score"],
        cgpa=sample_data["cgpa"],
        attendance=sample_data["attendance"]
    )

    print("Recommendation Engine Test")
    print("=" * 50)
    print(f"Overall Priority: {result['priority']}")
    print("\nStrengths:")
    for s in result["strengths"]:
        print(f"  ✅ {s}")

    print("\nRecommendations:")
    for i, rec in enumerate(result["recommendations"], 1):
        print(f"{i}. {rec['title']} (Priority: {rec['priority']})")
        print(f"   Reason: {rec['reason']}")