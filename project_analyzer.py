"""
project_analyzer.py
-------------------
Analyzes a student's project name and description to provide a quality score,
strengths, and improvement suggestions using HYBRID approach:
  - Zero-shot classification (transformers) for semantic understanding
  - Keyword matching for precision on technical terms

Rule: AI score is capped when there's no keyword evidence in the text.
Score capped at 92/100 for realism.
"""

import re
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =================================================================
# CONFIGURABLE SCORING PARAMETERS
# =================================================================
BASE_SCORE_PERCENTAGE = 0.7
STRENGTH_THRESHOLD = 0.5
IMPROVEMENT_THRESHOLD = 0.3
USE_ZERO_SHOT = True

AI_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

ZERO_SHOT_SCALE_FACTOR = 1.5
MIN_CONFIDENCE = 0.2

# =================================================================
# REALISM CAPS
# =================================================================
CATEGORY_CAP_RATIO = 0.95
TOTAL_SCORE_CAP = 92
NO_EVIDENCE_CAP_RATIO = 0.30   # NEW: Cap AI score at 30% if no keyword found
# =================================================================

# =================================================================
# DIMENSION DEFINITIONS
# =================================================================
DIMENSIONS = {
    "real_world_relevance": {
        "max_score": 15,
        "keywords": [
            "real-world", "real world", "production", "industry", "business",
            "deployment", "practical", "enterprise", "impact",
            "user", "customer", "market", "commercial", "real data",
            "actual data", "field", "live", "production-ready"
        ],
        "zero_shot_label": "real world relevance"
    },
    "ai_ml": {
        "max_score": 15,
        "keywords": [
            "machine learning", "ml", "ai", "deep learning", "neural",
            "nlp", "computer vision", "llm", "generative ai", "rag",
            "predict", "classification", "regression", "clustering",
            "recommend", "forecast", "predictive", "model", "training",
            "tuning", "ensemble", "xgboost", "tensorflow", "pytorch",
            "forecasting", "prediction", "artificial intelligence",
            "cnn", "rnn", "transformer", "embedding"
        ],
        "zero_shot_label": "artificial intelligence and machine learning"
    },
    "data_analytics": {
        "max_score": 10,
        "keywords": [
            "data", "analytics", "sql", "database", "etl", "visualization",
            "dashboard", "bi", "pandas", "data science", "data pipeline",
            "data warehouse", "big data", "analysis", "exploratory",
            "statistics", "reporting", "insights", "visual", "report",
            "insight", "streamlit", "plotly"
        ],
        "zero_shot_label": "data analysis and visualization"
    },
    "technical_complexity": {
        "max_score": 15,
        "keywords": [
            "api", "microservices", "distributed", "cloud", "docker",
            "kubernetes", "scalability", "optimization", "architecture",
            "algorithm", "model tuning", "parallel", "concurrency",
            "security", "authentication", "performance", "efficient",
            "complex", "system design", "integration", "optimize",
            "pipeline", "backend", "frontend", "full stack", "rest"
        ],
        "zero_shot_label": "technical complexity and system architecture"
    },
    "deployment_potential": {
        "max_score": 10,
        "keywords": [
            "deploy", "streamlit", "flask", "fastapi", "web app",
            "web application", "cloud", "aws", "azure", "gcp",
            "heroku", "docker", "ci/cd", "container", "hosted",
            "online", "server", "api endpoint", "web", "app",
            "deployment", "live", "production", "vercel"
        ],
        "zero_shot_label": "deployment and web application"
    },
    "industry_relevance": {
        "max_score": 10,
        "keywords": [
            "finance", "healthcare", "retail", "e-commerce", "logistics",
            "manufacturing", "energy", "telecom", "education", "recruitment",
            "human resources", "supply chain", "banking", "insurance",
            "social media", "transport", "agriculture",
            "student", "placement", "career", "job", "shopping", "payment"
        ],
        "zero_shot_label": "industry and business application"
    },
    "innovation": {
        "max_score": 10,
        "keywords": [
            "novel", "innovative", "cutting-edge", "state-of-the-art",
            "breakthrough", "research", "new approach", "unique",
            "pioneering", "first", "experimental", "advanced",
            "creative", "unconventional", "original"
        ],
        "zero_shot_label": "innovation and novelty"
    },
    "scalability": {
        "max_score": 5,
        "keywords": [
            "scalable", "big data", "real-time", "high-performance",
            "large-scale", "distributed", "elastic", "horizontal scaling",
            "load balancing", "high throughput"
        ],
        "zero_shot_label": "scalability and high performance"
    },
    "explainability": {
        "max_score": 5,
        "keywords": [
            "explain", "interpret", "lime", "shap", "transparent",
            "white-box", "feature importance", "model insights",
            "explainability", "interpretability", "understandable",
            "clear", "reasoning"
        ],
        "zero_shot_label": "explainability and interpretability"
    },
    "documentation": {
        "max_score": 5,
        "keywords": [
            "documentation", "readme", "api docs", "swagger",
            "tutorial", "guidelines", "user guide", "technical documentation",
            "docstrings", "comments", "examples"
        ],
        "zero_shot_label": "documentation and user guides"
    }
}

# ---------- Zero-shot classifier (cached) ----------
_ZERO_SHOT_PIPELINE = None

def get_zero_shot_classifier():
    global _ZERO_SHOT_PIPELINE
    if _ZERO_SHOT_PIPELINE is not None:
        return _ZERO_SHOT_PIPELINE
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        _ZERO_SHOT_PIPELINE = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1
        )
        return _ZERO_SHOT_PIPELINE
    except Exception:
        return None

# ---------- Helper functions ----------
def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

def _keyword_match_count(text: str, keywords: List[str]) -> int:
    norm_text = _normalize_text(text)
    matched = set()
    for kw in keywords:
        kw_lower = kw.lower()
        if len(kw_lower) <= 3:
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            if re.search(pattern, norm_text):
                matched.add(kw)
        else:
            if kw_lower in norm_text:
                matched.add(kw)
    return len(matched)

def _keyword_score(text: str, keywords: List[str], max_score: float) -> float:
    if not keywords:
        return 0.0
    matched = _keyword_match_count(text, keywords)
    if matched == 0:
        return 0.0
    ratio = matched / len(keywords)
    score = (0.5 + ratio * 0.5) * max_score
    return min(score, max_score)

# ---------- Main analysis function ----------
def analyze_project(project_name: str, project_description: str) -> Dict:
    combined_text = project_name + " " + project_description
    category_scores = {}
    strengths = []
    improvements = []

    # ----- STEP 1: Zero-shot AI -----
    ai_scores = {}
    if USE_ZERO_SHOT and TRANSFORMERS_AVAILABLE:
        classifier = get_zero_shot_classifier()
        if classifier is not None:
            candidate_labels = [DIMENSIONS[dim]["zero_shot_label"] for dim in DIMENSIONS]
            try:
                result = classifier(combined_text, candidate_labels, multi_label=True)
                for label, score in zip(result['labels'], result['scores']):
                    for k, v in DIMENSIONS.items():
                        if v["zero_shot_label"] == label:
                            max_score = v["max_score"]
                            adjusted = score * ZERO_SHOT_SCALE_FACTOR
                            adjusted = max(adjusted, MIN_CONFIDENCE)
                            adjusted = min(adjusted, 1.0)
                            ai_scores[k] = adjusted * max_score
                            break
            except Exception as e:
                print(f"Zero-shot failed: {e}")

    # ----- STEP 2: Keyword scores -----
    keyword_scores = {}
    keyword_counts = {}
    for dim, config in DIMENSIONS.items():
        keyword_counts[dim] = _keyword_match_count(combined_text, config["keywords"])
        keyword_scores[dim] = _keyword_score(
            combined_text,
            config["keywords"],
            config["max_score"]
        )

    # ----- STEP 3: Combine AI + Keywords with EVIDENCE CHECK -----
    for dim, config in DIMENSIONS.items():
        max_score = config["max_score"]
        ai_val = ai_scores.get(dim, 0.0)
        kw_val = keyword_scores.get(dim, 0.0)
        kw_count = keyword_counts[dim]

        if ai_val > 0 and kw_val > 0:
            final = max(ai_val, kw_val)
        elif ai_val > 0:
            final = ai_val
        else:
            final = kw_val

        final = min(final, max_score)

        # ✅ KEY FIX: If no keyword evidence, cap the score
        if kw_count == 0:
            evidence_cap = max_score * NO_EVIDENCE_CAP_RATIO
            if final > evidence_cap:
                final = evidence_cap

        # Cap individual category at 95% of max
        category_cap = max_score * CATEGORY_CAP_RATIO
        if final > category_cap:
            final = category_cap

        category_scores[dim] = round(final, 1)

        # Determine strengths/improvements
        if final >= STRENGTH_THRESHOLD * max_score:
            strengths.append(dim.replace('_', ' ').title())
        elif final < IMPROVEMENT_THRESHOLD * max_score:
            improvements.append(dim.replace('_', ' ').title())
        elif kw_count == 0:
            # Medium score but no evidence → improvement
            improvements.append(dim.replace('_', ' ').title())

    # ----- STEP 4: Total quality score -----
    total_score = sum(category_scores.values())
    quality_score = min(round(total_score), TOTAL_SCORE_CAP)

    strengths = list(dict.fromkeys(strengths))
    improvements = list(dict.fromkeys(improvements))

    if not strengths:
        strengths = ["No clear strengths identified – consider adding more detail."]
    if not improvements:
        improvements = ["Already well-rounded – keep up the good work!"]

    return {
        "project_name": project_name,
        "quality_score": quality_score,
        "strengths": strengths,
        "improvements": improvements,
        "category_scores": category_scores
    }


# -------------------------------------------------------------------
# Test Section
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: AI Project
    print("Test 1: AI Placement Project")
    print("=" * 50)
    result1 = analyze_project(
        "AI Placement Analytics System",
        "Analyze student performance and predict placement using machine learning, "
        "data analytics and interactive dashboard. Uses real-world data and LIME for explainability. "
        "Built with Python, Streamlit, ready for cloud deployment."
    )
    print(f"Quality Score: {result1['quality_score']}/100")
    print(f"Strengths: {result1['strengths']}")
    print(f"Improvements: {result1['improvements']}")

    print("\n" + "=" * 50)
    print("Test 2: E-Commerce Website (should NOT have AI/ML as strength)")
    print("=" * 50)
    result2 = analyze_project(
        "E-Commerce Shopping Website",
        "Developed a full-stack e-commerce website for buying electronics online. "
        "Built with React for frontend, Node.js and Express for backend, and MongoDB "
        "for database. Features include user authentication, product catalog, shopping "
        "cart, order management, and payment integration with Razorpay. Implemented "
        "JWT-based security and input validation. Designed a responsive UI with "
        "Tailwind CSS. Deployed on Vercel and Heroku."
    )
    print(f"Quality Score: {result2['quality_score']}/100")
    print(f"Strengths: {result2['strengths']}")
    print(f"Improvements: {result2['improvements']}")
    print("\nCategory Scores:")
    for dim, score in result2['category_scores'].items():
        max_score = DIMENSIONS[dim]["max_score"]
        print(f"  {dim.replace('_', ' ').title():20} {score:>5.1f} / {max_score}")