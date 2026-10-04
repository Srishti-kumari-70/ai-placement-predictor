"""
job_role_matcher.py
-------------------
Loads ESCO dataset from esco/ and suggests job roles based on student skills.
Uses sentence-transformers for semantic matching between student skills and ESCO skill labels.

Performance notes (same results as before, much faster):
  * Label embeddings are computed ONCE and saved to esco/_cache/*.npy, so later
    server starts load them from disk instead of re-encoding every ESCO label.
  * The embedding matrix is built and L2-normalised once; each student skill is
    matched with a single matrix-vector product (cosine similarity).
  * Each student skill is matched once per request (previously twice) and the
    result is memoised.
  * CSV -> dict loops use zip() instead of DataFrame.iterrows().
"""

import pandas as pd
import os
import re
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

# ---------- File paths ----------
ESCO_DIR = "esco"

OCC_FILE = os.path.join(ESCO_DIR, "occupations_en.csv")
SKILL_FILE = os.path.join(ESCO_DIR, "skills_en.csv")
RELATION_FILE = os.path.join(ESCO_DIR, "occupationSkillRelations_en.csv")

# ---------- Settings ----------
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(ESCO_DIR, "_cache")
# True  = embed preferred + alternative labels (best recall, slower one-time build)
# False = embed preferred labels only (about 5-10x faster one-time build; exact
#         matching still uses every label)
EMBED_ALT_LABELS = True

# Score soft cap (see suggest_roles): raw scores above SOFT_CAP_START are compressed toward 100
SOFT_CAP_START = 85.0
SOFT_CAP_SPREAD = 25.0

# ---------- Embedding model (cached) ----------
_EMBEDDING_MODEL = None

@st.cache_resource
def get_embedding_model():
    """Load the SentenceTransformer model once and cache it."""
    return SentenceTransformer(MODEL_NAME)

# ---------- Data loading (cached) ----------
_ESCO_DATA = None

def load_esco_data() -> Dict[str, pd.DataFrame]:
    global _ESCO_DATA
    if _ESCO_DATA is not None:
        return _ESCO_DATA

    encodings = ["utf-8", "latin-1"]
    for enc in encodings:
        try:
            occ = pd.read_csv(OCC_FILE, encoding=enc)
            skill = pd.read_csv(SKILL_FILE, encoding=enc)
            rel = pd.read_csv(RELATION_FILE, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode ESCO files.")

    print("=== ESCO Column Names ===")
    print("occupations_en.csv columns:", occ.columns.tolist())
    print("skills_en.csv columns:", skill.columns.tolist())
    print("occupationSkillRelations_en.csv columns:", rel.columns.tolist())
    print("==========================")

    if occ.empty or skill.empty or rel.empty:
        raise ValueError("One or more ESCO files are empty.")

    _ESCO_DATA = {"occupations": occ, "skills": skill, "relations": rel}
    return _ESCO_DATA


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise KeyError(f"None of {candidates} found.")


def _normalise(text: str) -> str:
    """Lowercase, strip, collapse multiple spaces."""
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', ' ', str(text).strip()).lower()


def _labels_signature(labels: List[str]) -> str:
    """Short fingerprint of the label list + model, used to validate the disk cache."""
    h = hashlib.md5()
    h.update(MODEL_NAME.encode("utf-8"))
    for lbl in labels:
        h.update(lbl.encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _load_or_build_embeddings(labels: List[str]) -> np.ndarray:
    """Return a normalised (n_labels x dim) matrix, from disk if available."""
    sig = _labels_signature(labels)
    path = os.path.join(CACHE_DIR, f"label_embeddings_{sig}.npy")

    if os.path.exists(path):
        try:
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[0] == len(labels):
                print(f"[job_role_matcher] Loaded {len(labels)} label embeddings from disk cache.", flush=True)
                return arr
        except Exception:
            pass  # corrupted cache -> rebuild below

    print(f"[job_role_matcher] Building embeddings for {len(labels)} ESCO labels "
          f"(one-time, saved to {CACHE_DIR})...", flush=True)
    t0 = time.time()
    model = get_embedding_model()
    arr = _l2_normalise(model.encode(labels, batch_size=256, show_progress_bar=False))
    print(f"[job_role_matcher] Embeddings built in {time.time() - t0:.1f}s.", flush=True)

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, arr)
    except Exception as e:
        print(f"[job_role_matcher] Could not save embedding cache: {e}", flush=True)
    return arr


@st.cache_resource
def build_role_matcher():
    """
    Builds mappings:
      - normalised_skill_name → skill_uri
      - label list + normalised embedding matrix
      - skill_uri → set of occupation_uris
      - occupation_uri → occupation_title
      - occupation_uri → set of skill_uris
    """
    t_start = time.time()
    data = load_esco_data()
    occ = data["occupations"]
    skill = data["skills"]
    rel = data["relations"]

    # Detect columns
    occ_id_col = _find_col(occ, ["concepturi", "uri", "id", "uuid"])
    occ_title_col = _find_col(occ, ["preferredlabel", "title", "label", "name"])
    skill_id_col = _find_col(skill, ["concepturi", "uri", "id", "uuid"])
    skill_pref_col = _find_col(skill, ["preferredlabel", "title", "label", "name"])
    skill_alt_col = _find_col(skill, ["altlabels", "altlabel", "alt", "synonyms"])
    rel_occ_col = _find_col(rel, ["occupationuri", "occupationid", "occupation_uuid", "occupation"])
    rel_skill_col = _find_col(rel, ["skilluri", "skillid", "skill_uuid", "skill"])

    # ---- 1. Skill name → URI (using preferred + alt) ----
    skill_name_to_uri = {}
    all_skill_labels = []
    preferred_labels = []

    alt_series = skill[skill_alt_col] if skill_alt_col in skill.columns else [""] * len(skill)

    for uri, pref_raw, alt_val in zip(skill[skill_id_col], skill[skill_pref_col], alt_series):
        if pd.isna(uri):
            continue

        pref = _normalise(pref_raw)
        if pref:
            skill_name_to_uri[pref] = uri
            all_skill_labels.append((pref, uri))
            preferred_labels.append(pref)

        if pd.notna(alt_val) and str(alt_val).strip():
            alt_str = str(alt_val)
            for sep in ['|', ',', ';']:
                if sep in alt_str:
                    parts = [p.strip() for p in alt_str.split(sep) if p.strip()]
                    break
            else:
                parts = [alt_str.strip()]
            for alt in parts:
                norm_alt = _normalise(alt)
                if norm_alt:
                    skill_name_to_uri[norm_alt] = uri
                    all_skill_labels.append((norm_alt, uri))

    # ---- 2. Embeddings for labels (disk-cached, normalised matrix) ----
    if EMBED_ALT_LABELS:
        embed_labels = list(skill_name_to_uri.keys())
    else:
        embed_labels = [l for l in dict.fromkeys(preferred_labels) if l in skill_name_to_uri]

    if embed_labels:
        emb_matrix = _load_or_build_embeddings(embed_labels)
    else:
        emb_matrix = np.zeros((0, 0), dtype="float32")

    # ---- 3. Occupation URI → Title ----
    occ_uri_to_title = {}
    for uri, title_raw in zip(occ[occ_id_col], occ[occ_title_col]):
        title = str(title_raw).strip()
        if pd.notna(uri) and title:
            occ_uri_to_title[uri] = title

    # ---- 4. Skill URI ↔ Occupation URI mappings ----
    skill_to_occupations = {}
    occupation_to_skills = {}
    for skill_uri, occ_uri in zip(rel[rel_skill_col], rel[rel_occ_col]):
        if pd.notna(skill_uri) and pd.notna(occ_uri):
            skill_to_occupations.setdefault(skill_uri, set()).add(occ_uri)
            occupation_to_skills.setdefault(occ_uri, set()).add(skill_uri)

    print(f"[job_role_matcher] Matcher ready in {time.time() - t_start:.1f}s.", flush=True)

    return {
        "skill_name_to_uri": skill_name_to_uri,
        "embed_labels": embed_labels,
        "emb_matrix": emb_matrix,
        "occ_uri_to_title": occ_uri_to_title,
        "skill_to_occupations": skill_to_occupations,
        "occupation_to_skills": occupation_to_skills,
        "all_skill_labels": all_skill_labels,
        "match_cache": {},
    }


# ---------- Semantic matching ----------
def _match_skill_with_embedding(student_skill: str, matcher, similarity_threshold: float = 0.4) -> Tuple[str, bool]:
    """
    Try to match a student skill using:
      1. Exact match on any label.
      2. Embedding-based cosine similarity (single matrix-vector product).
    Results are memoised per (skill, threshold).
    """
    norm = _normalise(student_skill)
    if not norm:
        return None, False

    skill_name_to_uri = matcher["skill_name_to_uri"]

    # 1. Exact match
    if norm in skill_name_to_uri:
        return skill_name_to_uri[norm], True

    # 2. Embedding-based match
    emb_matrix = matcher["emb_matrix"]
    if emb_matrix.shape[0] == 0:
        return None, False

    cache = matcher["match_cache"]
    key = (norm, similarity_threshold)
    if key in cache:
        return cache[key], False

    model = get_embedding_model()
    student_emb = _l2_normalise(model.encode([norm], show_progress_bar=False))[0]

    similarities = emb_matrix @ student_emb  # cosine similarity (both sides normalised)
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    uri = None
    if best_score >= similarity_threshold:
        uri = skill_name_to_uri[matcher["embed_labels"][best_idx]]

    cache[key] = uri
    return uri, False


def suggest_roles(student_skills: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
    if not student_skills:
        return []

    t0 = time.time()
    matcher = build_role_matcher()
    occ_uri_to_title = matcher["occ_uri_to_title"]
    skill_to_occupations = matcher["skill_to_occupations"]
    occupation_to_skills = matcher["occupation_to_skills"]

    # Normalise and deduplicate student skills
    cleaned = list(set([_normalise(s) for s in student_skills if _normalise(s)]))

    # Match every student skill once
    matched_uri = {}
    for s in cleaned:
        uri, _ = _match_skill_with_embedding(s, matcher)
        if uri:
            matched_uri[s] = uri

    matched_esco_skills = list(matched_uri.keys())

    if not matched_esco_skills:
        return []

    # Count how many student skills are covered by each occupation
    occupation_skill_count = {}
    for skill_name in matched_esco_skills:
        uri = matched_uri[skill_name]
        occ_uris = skill_to_occupations.get(uri, set())
        for occ_uri in occ_uris:
            occupation_skill_count.setdefault(occ_uri, set()).add(skill_name)

    # ---- Compute scores ----
    results = []
    total_skills = len(cleaned)
    for occ_uri, matched_set in occupation_skill_count.items():
        occ_total_skills = len(occupation_to_skills.get(occ_uri, set()))

        if occ_total_skills == 0:
            continue

        # Coverage of student skills
        student_coverage = len(matched_set) / total_skills

        # Coverage of occupation skills (capped at 10)
        effective_total = min(occ_total_skills, 10)
        occ_coverage = len(matched_set) / effective_total
        occ_coverage = min(occ_coverage, 1.0)

        # Base score
        score = (student_coverage * 60) + (occ_coverage * 40)

        # Bonus for matching more skills
        score += len(matched_set) * 3

        # ---- Domain relevance bonus / penalty ----
        role_lower = occ_uri_to_title.get(occ_uri, "").lower()

        relevant_keywords = [
            'data', 'software', 'analyst', 'computer',
            'information', 'ict', 'developer', 'programmer',
            'database', 'cloud', 'machine learning', 'ai',
            'application', 'systems', 'engineer'
        ]

        irrelevant_keywords = [
            'military', 'environmental', 'battery',
            'microelectronics', 'materials', 'chemical',
            'civil', 'mechanical', 'electrical power'
        ]

        if any(kw in role_lower for kw in relevant_keywords):
            score += 25

        if any(kw in role_lower for kw in irrelevant_keywords):
            score -= 20

        # ---- Priority role bonus (well-known target roles) ----
        priority_roles = {
            'data scientist': 20,
            'data analyst': 18,
            'machine learning engineer': 18,
            'software developer': 15,
            'software engineer': 15,
            'data engineer': 15,
            'business analyst': 12,
            'database developer': 10,
            'cloud engineer': 10,
            'full stack developer': 10,
            'frontend developer': 10,
            'backend developer': 10,
            'ai engineer': 15,
            'research scientist': 8,
        }

        for priority_role, bonus in priority_roles.items():
            if priority_role in role_lower:
                score += bonus
                break

        # Soft cap: scores up to SOFT_CAP_START are unchanged; above that they bend toward
        # 100 and never reach it, so 100% is not given away to every strong match and
        # the ranking order is preserved.
        score = max(score, 0.0)
        if score > SOFT_CAP_START:
            score = SOFT_CAP_START + (100.0 - SOFT_CAP_START) * (
                1.0 - float(np.exp(-(score - SOFT_CAP_START) / SOFT_CAP_SPREAD)))

        role = occ_uri_to_title.get(occ_uri, "Unknown")
        results.append({
            "role": role,
            "match_score": round(score, 1),
            "matched_skills": list(matched_set)
        })

    # Sort and filter
    results.sort(key=lambda x: x["match_score"], reverse=True)
    results = [r for r in results if r["match_score"] >= 10.0]

    print(f"[job_role_matcher] suggest_roles core took {time.time() - t0:.2f}s", flush=True)
    return results[:top_n]


# ---------- Test ----------
if __name__ == "__main__":
    print("ESCO Job Role Matcher - Test Run (sentence-transformers enabled)")
    print("Loading ESCO data...")
    try:
        test_skills = ["Python", "SQL", "Machine Learning", "Data analysis", "Pandas"]
        print(f"\nStudent skills: {test_skills}")
        suggestions = suggest_roles(test_skills, top_n=10)
        if suggestions:
            print("\nTop job role suggestions:")
            for i, job in enumerate(suggestions, 1):
                print(f"{i}. {job['role']} — {job['match_score']:.1f}% skill match")
                print(f"   Matched skills: {', '.join(job['matched_skills'])}")
        else:
            print("\nNo matching occupations found. Check your ESCO files or skill names.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()