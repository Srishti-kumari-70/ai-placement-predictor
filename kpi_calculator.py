import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# ===============================================
#  CONFIG – Edit these
# ===============================================
DB_PATH = 'datadb/college_placement.db'
OUTPUT_CSV = 'reports/college_kpi_summary.csv'
OUTPUT_DIR = 'reports'

# ===============================================
#  LOGGING
# ===============================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================================
#  LOAD DATA FROM DATABASE
# ===============================================
def load_data():
    """Load data from database tables."""
    logger.info(f"Loading data from {DB_PATH}...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Load from feature-engineered table if exists, else from joined tables
        try:
            df = pd.read_sql_query("SELECT * FROM student_features", conn)
            logger.info("Loaded feature-engineered data from 'student_features'")
        except:
            # Fallback to joined tables
            query = """
            SELECT s.student_id, s.gender, s.age,
                   a.ssc_percentage, a.hsc_percentage, a.degree_cgpa,
                   sk.coding_score, sk.aptitude_score, sk.attendance, sk.internships, sk.projects,
                   p.placement_status
            FROM students s
            JOIN academics a ON s.student_id = a.student_id
            JOIN skills sk ON s.student_id = sk.student_id
            JOIN placement p ON s.student_id = p.student_id
            """
            df = pd.read_sql_query(query, conn)
            logger.info("Loaded clean data from joined tables")
        
        conn.close()
        logger.info(f"✅ Loaded {len(df)} students")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

# ===============================================
#  CALCULATE KPIs
# ===============================================
def calculate_kpis(df):
    """Calculate all college KPIs."""
    logger.info("Calculating KPIs...")
    
    # Overall KPIs
    total_students = len(df)
    placed_count = len(df[df['placement_status'] == 'Placed'])
    not_placed_count = total_students - placed_count
    placement_rate = (placed_count / total_students) * 100
    
    # Academic KPIs
    avg_cgpa = df['degree_cgpa'].mean()
    avg_coding = df['coding_score'].mean()
    avg_aptitude = df['aptitude_score'].mean()
    avg_attendance = df['attendance'].mean()
    
    # ✅ NEW: Experience KPIs
    avg_internships = df['internships'].mean()
    avg_projects = df['projects'].mean()
    
    # Overall summary
    overall_kpis = {
        'Metric': [
            'Total Students', 'Placed Students', 'Not Placed', 
            'Placement Rate %', 'Avg CGPA', 'Avg Coding Score', 
            'Avg Aptitude Score', 'Avg Attendance %',
            'Avg Internships', 'Avg Projects'
        ],
        'Value': [
            total_students, placed_count, not_placed_count,
            round(placement_rate, 2), round(avg_cgpa, 2),
            round(avg_coding, 2), round(avg_aptitude, 2),
            round(avg_attendance, 1),
            round(avg_internships, 2), round(avg_projects, 2)
        ]
    }
    kpi_df = pd.DataFrame(overall_kpis)
    
    # Gender-wise
    gender_stats = df.groupby('gender').agg(
        total=('student_id', 'count'),
        placed=('placement_status', lambda x: (x == 'Placed').sum()),
        placement_rate=('placement_status', lambda x: (x == 'Placed').mean() * 100)
    ).round(2)
    
    # Risk-wise (if risk_category exists)
    if 'risk_category' in df.columns:
        risk_stats = df.groupby('risk_category').agg(
            total=('student_id', 'count'),
            placed=('placement_status', lambda x: (x == 'Placed').sum()),
            placement_rate=('placement_status', lambda x: (x == 'Placed').mean() * 100)
        ).round(2)
    else:
        risk_stats = None
    
    # Department-wise (if department exists)
    if 'department' in df.columns:
        dept_stats = df.groupby('department').agg(
            total=('student_id', 'count'),
            placed=('placement_status', lambda x: (x == 'Placed').sum()),
            placement_rate=('placement_status', lambda x: (x == 'Placed').mean() * 100)
        ).round(2).sort_values('placement_rate', ascending=False)
    else:
        dept_stats = None
    
    # Add overall_strength stats if it exists
    if 'overall_strength' in df.columns:
        strength_stats = df.groupby('placement_status')['overall_strength'].agg(['mean', 'median']).round(2)
    else:
        strength_stats = None
    
    results = {
        'overall': kpi_df,
        'gender': gender_stats,
        'risk': risk_stats,
        'department': dept_stats,
        'strength': strength_stats
    }
    
    logger.info("✅ KPI calculation complete")
    return results

# ===============================================
#  SAVE RESULTS
# ===============================================
def save_results(results, output_dir):
    """Save all KPI results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    results['overall'].to_csv(os.path.join(output_dir, 'kpi_overall.csv'), index=False)
    results['gender'].to_csv(os.path.join(output_dir, 'kpi_gender.csv'))
    
    if results['risk'] is not None:
        results['risk'].to_csv(os.path.join(output_dir, 'kpi_risk.csv'))
    
    if results['department'] is not None:
        results['department'].to_csv(os.path.join(output_dir, 'kpi_department.csv'))
    
    if results['strength'] is not None:
        results['strength'].to_csv(os.path.join(output_dir, 'kpi_strength.csv'))
    
    logger.info(f"✅ All KPI reports saved to {output_dir}/")

# ===============================================
#  DISPLAY DASHBOARD
# ===============================================
def display_dashboard(results):
    """Print the KPI dashboard to console."""
    print("\n" + "="*60)
    print("🏆 COLLEGE PERFORMANCE DASHBOARD")
    print("="*60)
    
    # Overall
    overall = results['overall']
    print(f"👨‍🎓 Total Students          : {overall[overall['Metric']=='Total Students']['Value'].values[0]}")
    print(f"✅ Placed Students         : {overall[overall['Metric']=='Placed Students']['Value'].values[0]}")
    print(f"❌ Not Placed Students     : {overall[overall['Metric']=='Not Placed']['Value'].values[0]}")
    print(f"📈 Placement Rate          : {overall[overall['Metric']=='Placement Rate %']['Value'].values[0]:.2f}%")
    print(f"📚 Average CGPA            : {overall[overall['Metric']=='Avg CGPA']['Value'].values[0]:.2f}")
    print(f"💻 Average Coding Score    : {overall[overall['Metric']=='Avg Coding Score']['Value'].values[0]:.2f}")
    print(f"🧠 Average Aptitude Score  : {overall[overall['Metric']=='Avg Aptitude Score']['Value'].values[0]:.2f}")
    print(f"🏫 Average Attendance      : {overall[overall['Metric']=='Avg Attendance %']['Value'].values[0]:.1f}%")
    print(f"🏢 Average Internships     : {overall[overall['Metric']=='Avg Internships']['Value'].values[0]:.2f}")
    print(f"📂 Average Projects        : {overall[overall['Metric']=='Avg Projects']['Value'].values[0]:.2f}")
    
    print("\n" + "-"*60)
    print("👦👧 GENDER-WISE:")
    print(results['gender'].to_string())
    
    if results['risk'] is not None:
        print("\n" + "-"*60)
        print("⚠️ RISK CATEGORY BREAKDOWN:")
        print(results['risk'].to_string())
    
    if results['department'] is not None:
        print("\n" + "-"*60)
        print("🏛️ DEPARTMENT-WISE (Top to Bottom):")
        print(results['department'].to_string())
    
    if results['strength'] is not None:
        print("\n" + "-"*60)
        print("💪 OVERALL STRENGTH BY PLACEMENT:")
        print(results['strength'].to_string())
    
    print("\n" + "="*60)

# ===============================================
#  MAIN
# ===============================================
def main():
    logger.info("="*50)
    logger.info("🚀 KPI CALCULATOR")
    logger.info("="*50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate KPIs
    results = calculate_kpis(df)
    
    # Display dashboard
    display_dashboard(results)
    
    # Save results
    save_results(results, OUTPUT_DIR)
    
    logger.info("="*50)
    logger.info("🎉 KPI Calculator Complete!")
    logger.info(f"📁 Reports saved in: {OUTPUT_DIR}/")
    logger.info("="*50)

if __name__ == '__main__':
    main()