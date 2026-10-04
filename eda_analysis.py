"""
EDA Pipeline – Complete, Simple, No Warnings, No Errors
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# ===============================================
#  CONFIG – Bas yahan change karo
# ===============================================
DB_PATH = 'datadb/college_placement.db'
OUTPUT_DIR = 'reports/eda'

# ===============================================
#  1. LOAD DATA FROM DATABASE
# ===============================================
def load_data():
    print("Loading data from database...")
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
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df)} students")
    return df

# ===============================================
#  2. CREATE OUTPUT FOLDER WITH TIMESTAMP
# ===============================================
def create_output_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(path, exist_ok=True)
    print(f"Output folder: {path}")
    return path

# ===============================================
#  3. SUMMARY STATISTICS
# ===============================================
def save_summary_stats(df, out_dir):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats_df = df.groupby('placement_status')[numeric_cols].agg(['mean', 'median', 'std']).round(2)
    stats_df.to_csv(os.path.join(out_dir, 'summary_stats.csv'))
    print("Summary stats saved")

# ===============================================
#  4. T‑TEST (Placed vs Not Placed)
# ===============================================
def perform_ttest(df, out_dir):
    """T-test using numpy only (no scipy needed)."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    placed = df[df['placement_status'] == 'Placed']
    not_placed = df[df['placement_status'] == 'Not_Placed']
    
    results = []
    for col in numeric_cols:
        g1 = placed[col].dropna().values
        g2 = not_placed[col].dropna().values
        
        n1, n2 = len(g1), len(g2)
        m1, m2 = np.mean(g1), np.mean(g2)
        v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        
        # T-value (Welch's t-test)
        t = (m1 - m2) / np.sqrt(v1/n1 + v2/n2)
        
        # Degrees of freedom (Welch)
        df_welch = ((v1/n1 + v2/n2)**2) / (
            ((v1/n1)**2)/(n1-1) + ((v2/n2)**2)/(n2-1)
        )
        
        # Approximate p-value using normal distribution
        # (works well for large samples; n>30)
        # Two-tailed p-value approximation
        from math import erf, sqrt
        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        
        results.append({
            'feature': col,
            't_stat': round(t, 3),
            'p_value': round(p, 4),
            'significant': p < 0.05
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(out_dir, 'ttest_results.csv'), index=False)
    print("T-test results saved")
    return result_df

# ===============================================
#  5. PLOT: Pie Chart – Placement Ratio
# ===============================================
def plot_placement_pie(df, out_dir):
    counts = df['placement_status'].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FF5733'])
    plt.title('Placement Ratio')
    plt.savefig(os.path.join(out_dir, 'placement_pie.png'), dpi=150)
    plt.close()
    print("Pie chart saved")

# ===============================================
#  6. PLOT: Histogram – CGPA Distribution
# ===============================================
def plot_cgpa_histogram(df, out_dir):
    plt.figure(figsize=(10,5))
    sns.histplot(data=df, x='degree_cgpa', hue='placement_status', kde=True, bins=20)
    plt.title('CGPA Distribution by Placement Status')
    plt.xlabel('CGPA')
    plt.savefig(os.path.join(out_dir, 'cgpa_histogram.png'), dpi=150)
    plt.close()
    print("CGPA histogram saved")

# ===============================================
#  7. PLOT: Boxplots – All Key Features
# ===============================================
def plot_boxplots(df, out_dir):
    features = ['degree_cgpa', 'coding_score', 'aptitude_score', 'attendance']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(x='placement_status', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} by Placement')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boxplots.png'), dpi=150)
    plt.close()
    print("Boxplots saved")

# ===============================================
#  8. PLOT: Correlation Heatmap
# ===============================================
def plot_correlation_heatmap(df, out_dir):
    corr = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("Correlation heatmap saved")

# ===============================================
#  9. PLOT: Student vs Class Average
# ===============================================
def plot_student_comparison(df, out_dir):
    student = df.iloc[0]
    student_id = student['student_id']
    class_avg = df[['degree_cgpa', 'coding_score', 'aptitude_score', 'attendance']].mean()
    student_cgpa = student['degree_cgpa'] / 10 * 100
    class_cgpa = class_avg['degree_cgpa'] / 10 * 100
    categories = ['CGPA (scaled)', 'Coding', 'Aptitude', 'Attendance']
    student_vals = [student_cgpa, student['coding_score'], student['aptitude_score'], student['attendance']]
    class_vals = [class_cgpa, class_avg['coding_score'], class_avg['aptitude_score'], class_avg['attendance']]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, student_vals, width, label=f'Student {student_id}', color='indigo')
    ax.bar(x + width/2, class_vals, width, label='Class Average', color='gray')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Student vs Class Average')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'student_comparison.png'), dpi=150)
    plt.close()
    print(f"Student comparison saved (Student: {student_id})")

# ===============================================
#  10. PLOT: Gender-wise Placement
# ===============================================
def plot_gender_placement(df, out_dir):
    gender_place = df.groupby(['gender', 'placement_status']).size().unstack()
    gender_place.plot(kind='bar', stacked=True, color=['#FF5733', '#4CAF50'], figsize=(8,6))
    plt.title('Gender-wise Placement')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Status')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gender_placement.png'), dpi=150)
    plt.close()
    print("Gender-wise plot saved")

# ===============================================
#  11. MAIN PIPELINE
# ===============================================
def main():
    print("="*50)
    print("STARTING EDA PIPELINE")
    print("="*50)

    df = load_data()
    out_dir = create_output_dir()

    save_summary_stats(df, out_dir)
    perform_ttest(df, out_dir)
    plot_placement_pie(df, out_dir)
    plot_cgpa_histogram(df, out_dir)
    plot_boxplots(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_student_comparison(df, out_dir)
    plot_gender_placement(df, out_dir)

    with open(os.path.join(out_dir, 'README.txt'), 'w') as f:
        f.write("EDA Report\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total Students: {len(df)}\n")
        f.write(f"Placed: {len(df[df['placement_status']=='Placed'])}\n")
        f.write(f"Not Placed: {len(df[df['placement_status']=='Not_Placed'])}\n")
        f.write(f"Placement Rate: {len(df[df['placement_status']=='Placed'])/len(df)*100:.1f}%\n")
        f.write("\nFiles generated:\n")
        f.write("- summary_stats.csv\n")
        f.write("- ttest_results.csv\n")
        f.write("- placement_pie.png\n")
        f.write("- cgpa_histogram.png\n")
        f.write("- boxplots.png\n")
        f.write("- correlation_heatmap.png\n")
        f.write("- student_comparison.png\n")
        f.write("- gender_placement.png\n")

    print("="*50)
    print("EDA COMPLETE!")
    print(f"All outputs saved in: {out_dir}")
    print("="*50)

if __name__ == '__main__':
    main()