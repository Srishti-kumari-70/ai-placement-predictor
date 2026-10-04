"""
Integrated Database Loader – uses pre‑cleaned data.
"""

import os
import logging
import sqlite3
import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------
INPUT_CSV = 'datadb/student_placement_data_clean.csv'
DB_PATH = 'datadb/college_placement.db'
LOG_LEVEL = logging.INFO

# ------------------------------
# LOGGING
# ------------------------------
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# DATABASE LOADER
# ------------------------------
def load_to_database(df, db_path):
    """Load a cleaned DataFrame into a normalized SQLite database."""
    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")   # enforce FK

    try:
        # Drop tables in **reverse** dependency order (children first)
        tables_to_drop = ['placement', 'skills', 'academics', 'students']
        for table in tables_to_drop:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Dropped table (if existed): {table}")

        # 1. Students table
        students = df[['student_id', 'gender', 'age']]
        students.to_sql('students', conn, if_exists='replace', index=False)
        logger.info(f"Inserted {len(students)} rows into 'students'.")

        # 2. Academics
        academics = df[['student_id', 'ssc_percentage', 'hsc_percentage', 'degree_cgpa']]
        academics.to_sql('academics', conn, if_exists='replace', index=False)
        logger.info(f"Inserted {len(academics)} rows into 'academics'.")

        # 3. Skills (include outlier flags if present)
        skill_cols = ['student_id', 'coding_score', 'aptitude_score', 'attendance', 'internships', 'projects']
        extra_cols = [c for c in df.columns if c.endswith('_outlier')]
        if extra_cols:
            skill_cols.extend(extra_cols)
        skills = df[skill_cols]
        skills.to_sql('skills', conn, if_exists='replace', index=False)
        logger.info(f"Inserted {len(skills)} rows into 'skills'.")

        # 4. Placement
        placement = df[['student_id', 'placement_status']]
        placement.to_sql('placement', conn, if_exists='replace', index=False)
        logger.info(f"Inserted {len(placement)} rows into 'placement'.")

        conn.commit()
        logger.info("✅ Database loaded successfully.")

    except Exception as e:
        logger.error(f"❌ Failed to load database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

# ------------------------------
# MAIN
# ------------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        logger.error(f"Input CSV not found: {INPUT_CSV}")
        return

    logger.info(f"Loading cleaned data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    load_to_database(df, DB_PATH)

if __name__ == "__main__":
    main()