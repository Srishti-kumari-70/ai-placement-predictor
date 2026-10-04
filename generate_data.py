import numpy as np
import pandas as pd
import os

# ------------------------------
# CONFIGURATION (change these as you like)
# ------------------------------
NUM_STUDENTS = 1000
RANDOM_SEED = 42
MISSING_RATE = 0.05          # 5% values will be missing
OUTLIER_FRACTION = 0.02      # 2% extreme outliers
OUTPUT_PATH = 'datadb/student_placement_data.csv'   # your original folder

np.random.seed(RANDOM_SEED)

# ------------------------------
# 1. Generate correlated numeric features
# ------------------------------
# We'll use a multivariate normal with a correlation matrix.
# Means and standard deviations for each feature.
means = {
    'ssc': 70, 'hsc': 70, 'cgpa': 7.5,
    'coding': 65, 'aptitude': 65, 'attendance': 80
}
stds = {
    'ssc': 10, 'hsc': 12, 'cgpa': 1.0,
    'coding': 15, 'aptitude': 15, 'attendance': 10
}

# Correlation matrix (realistic relationships)
corr = np.array([
    [1.0,  0.7,  0.6,  0.3,  0.3,  0.1],
    [0.7,  1.0,  0.6,  0.3,  0.3,  0.1],
    [0.6,  0.6,  1.0,  0.4,  0.4,  0.2],
    [0.3,  0.3,  0.4,  1.0,  0.5,  0.1],
    [0.3,  0.3,  0.4,  0.5,  1.0,  0.1],
    [0.1,  0.1,  0.2,  0.1,  0.1,  1.0]
])

# Generate multivariate normal
L = np.linalg.cholesky(corr)
raw = np.random.normal(size=(NUM_STUDENTS, 6))
data = raw @ L.T   # shape (1000, 6)

# Scale to desired means & stds, then clip to realistic ranges
features = ['ssc', 'hsc', 'cgpa', 'coding', 'aptitude', 'attendance']
df_dict = {}
for i, name in enumerate(features):
    values = means[name] + stds[name] * data[:, i]
    # Clip
    if name in ['ssc', 'hsc']:
        values = np.clip(values, 30, 100)
    elif name == 'cgpa':
        values = np.clip(values, 4.0, 10.0)
    elif name in ['coding', 'aptitude']:
        values = np.clip(values, 0, 100)
    elif name == 'attendance':
        values = np.clip(values, 40, 100)
    # Keep as float (so we can add missing values later)
    df_dict[name] = values.astype(float)

# Round some columns
df_dict['ssc'] = np.round(df_dict['ssc'], 2)
df_dict['hsc'] = np.round(df_dict['hsc'], 2)
df_dict['cgpa'] = np.round(df_dict['cgpa'], 2)

# ------------------------------
# 2. Categorical features
# ------------------------------

genders = np.random.choice(['M', 'F'], size=NUM_STUDENTS, p=[0.55, 0.45])
ages = np.random.normal(loc=21, scale=1.5, size=NUM_STUDENTS).clip(19, 25).astype(int)

# Compute quality score from academic features (for correlating internships/projects)
quality = (df_dict['cgpa'] * 3 + df_dict['coding'] * 0.05 + 
           df_dict['aptitude'] * 0.05 + df_dict['ssc'] * 0.02)

# Better students get more internships (0-5)
internships = np.clip(
    np.random.poisson(lam=np.clip(quality - 20, 0, 5)), 
    0, 5
).astype(int)

# Better students get more projects (0-6)
projects = np.clip(
    np.round((quality - 12) / 5),
    0, 6
).astype(int)

# ------------------------------
# 3. Introduce missing values (randomly)
# ------------------------------
for col in ['ssc', 'hsc', 'cgpa', 'coding', 'aptitude', 'attendance']:
    mask = np.random.random(NUM_STUDENTS) < MISSING_RATE
    df_dict[col][mask] = np.nan

# ------------------------------
# 4. Introduce outliers (extreme values)
# ------------------------------
for col in ['coding', 'aptitude', 'attendance']:
    idx = np.random.choice(NUM_STUDENTS, size=int(NUM_STUDENTS * OUTLIER_FRACTION), replace=False)
    if col in ['coding', 'aptitude']:
        df_dict[col][idx] = np.random.choice([0, 100], size=len(idx))
    else:  # attendance
        df_dict[col][idx] = np.random.choice([40, 100], size=len(idx))

# ------------------------------
# 5. Determine placement status
# ------------------------------
# Temporarily fill missing values with column mean to compute a score
temp_df = pd.DataFrame(df_dict)
temp_imputed = temp_df.fillna(temp_df.mean())

# Weighted score (higher = more likely to be placed)
score = (temp_imputed['ssc'] * 0.10 +
         temp_imputed['hsc'] * 0.10 +
         (temp_imputed['cgpa'] / 10 * 100) * 0.20 +   
         temp_imputed['coding'] * 0.20 +
         temp_imputed['aptitude'] * 0.12 +
         temp_imputed['attendance'] * 0.08 +
         internships * 1.5 +
         projects * 1.5)


noise = np.random.normal(0, 3, NUM_STUDENTS)
adjusted_score = score + noise
placement_status = np.where(adjusted_score > 65, 'Placed', 'Not_Placed')

# ------------------------------
# 6. Assemble final DataFrame
# ------------------------------
df = pd.DataFrame({
    'student_id': [f'S{i:04d}' for i in range(1, NUM_STUDENTS + 1)],
    'gender': genders,
    'age': ages,
    'ssc_percentage': df_dict['ssc'],
    'hsc_percentage': df_dict['hsc'],
    'degree_cgpa': df_dict['cgpa'],
    'coding_score': df_dict['coding'],
    'aptitude_score': df_dict['aptitude'],
    'internships': internships,
    'projects': projects,
    'attendance': df_dict['attendance'],
    'placement_status': placement_status
})

# ------------------------------
# 7. Save
# ------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset saved to {OUTPUT_PATH}")
print(f"   Students: {NUM_STUDENTS}")
print(f"   Missing rate: {MISSING_RATE*100:.0f}%")
print(f"   Outlier fraction: {OUTLIER_FRACTION*100:.0f}%")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

