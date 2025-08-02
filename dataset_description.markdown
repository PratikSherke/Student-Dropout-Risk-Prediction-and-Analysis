# Student Dropout Dataset Description

## Overview
The dataset used in the **Student Dropout Risk Analysis Dashboard** contains 4424 student records with 37 features, capturing demographic, academic, and socio-economic data. The target variable indicates whether a student dropped out, graduated, or is enrolled, converted to a binary variable (Dropout vs. Non-Dropout) for modeling.

### Files
- **student's dropout dataset.csv**: Original dataset with numerical encodings for categorical variables and the provided raw data.
- **mapped_student_dropout_dataset.csv**: Processed dataset with categorical variables mapped to human-readable values for dashboard display (not provided but assumed to exist based on context).

### Features
- **Demographic**:
  - `Marital status`: 1 (Single), 2 (Married), 3 (Widower), 4 (Divorced), 5 (Facto union), 6 (Legally separated)
  - `Application mode`: Numeric codes (e.g., 1, 6, 8, etc.) representing application methods
  - `Application order`: Numeric order of application (e.g., 1, 2, 3, etc.)
  - `Course`: Numeric codes (e.g., 1, 2, 3, etc.) representing different courses
  - `Daytime/evening attendance`: 0 (Evening), 1 (Daytime)
  - `Previous qualification`: Numeric codes (e.g., 1, 3, 12, 14, 15, 16) for prior education levels
  - `Nationality`: Numeric codes (e.g., 1, 3, 7, 9, 12, 14, 15, 19) representing countries
  - `Mother's qualification`: Numeric codes (e.g., 1, 2, 3, 4, 10, 13, 14, 19, 22, 23, 27, 28, 29) for mother's education
  - `Father's qualification`: Numeric codes (e.g., 1, 2, 3, 5, 10, 11, 14, 24, 27, 28, 29) for father's education
  - `Mother's occupation`: Numeric codes (e.g., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 21, 22, 31, 39, 45) for mother's job
  - `Father's occupation`: Numeric codes (e.g., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) for father's job
  - `Displaced`: 0 (No), 1 (Yes) indicating displacement status
  - `Educational special needs`: 0 (No), 1 (Yes) indicating special needs
  - `Debtor`: 0 (No), 1 (Yes) indicating financial debt
  - `Tuition fees up to date`: 0 (No), 1 (Yes) indicating payment status
  - `Gender`: 0 (Female), 1 (Male)
  - `Scholarship holder`: 0 (No), 1 (Yes) indicating scholarship status
  - `Age at enrollment`: Age in years (e.g., 18, 19, 20, etc.)
  - `International`: 0 (No), 1 (Yes) indicating international student status
- **Academic**:
  - `Curricular units 1st sem (credited)`: Number of credited units in the first semester
  - `Curricular units 1st sem (enrolled)`: Number of enrolled units in the first semester
  - `Curricular units 1st sem (evaluations)`: Number of evaluations in the first semester
  - `Curricular units 1st sem (approved)`: Number of approved units in the first semester
  - `Curricular units 1st sem (grade)`: Average grade in the first semester
  - `Curricular units 1st sem (without evaluations)`: Number of units without evaluations in the first semester
  - `Curricular units 2nd sem (credited)`: Number of credited units in the second semester
  - `Curricular units 2nd sem (enrolled)`: Number of enrolled units in the second semester
  - `Curricular units 2nd sem (evaluations)`: Number of evaluations in the second semester
  - `Curricular units 2nd sem (approved)`: Number of approved units in the second semester
  - `Curricular units 2nd sem (grade)`: Average grade in the second semester
  - `Curricular units 2nd sem (without evaluations)`: Number of units without evaluations in the second semester
- **Socio-economic**:
  - `Unemployment rate`: Percentage rate (e.g., 7.6, 8.9, 9.4, etc.)
  - `Inflation rate`: Percentage rate (e.g., -0.3, 0.3, 0.6, etc.)
  - `GDP`: Percentage rate (e.g., -4.06, -1.7, 0.32, etc.)
- **Target**: 
  - Original: "Dropout", "Graduate", "Enrolled"
  - Processed: Binary (1 = Dropout, 0 = Non-Dropout)

### Preprocessing
- **Cleaning**: Corrected column names (e.g., 'Nacionality' to 'Nationality' is not applicable here as the provided data uses the correct spelling).
- **Feature Selection**: Dropped correlated features (e.g., Nationality, Mother’s occupation, Father’s qualification) may be considered based on further analysis.
- **PCA**: Reduced 12 curricular unit features into one `Curricular 1st and 2nd sem PCA` feature (planned for modeling).
- **Normalization**: Applied StandardScaler to numerical features (planned for modeling).
- **Mapping**: Created `mapped_student_dropout_dataset.csv` with readable categorical values for dashboard use (assumed based on context).

### Notes
- The dataset contains 4424 rows with no missing values observed in the provided sample.
- The BOM (Byte Order Mark) `﻿` is present at the start of the CSV, which should be handled during parsing.
- Categorical variables (e.g., Marital status, Course) are encoded as numeric values, requiring mapping for interpretability.
- Refer to `student's_dropout_prediction_projejct.ipynb` for detailed EDA and preprocessing steps.

### Sample Data Insights
- Age at enrollment ranges from 18 to 55 years.
- Target distribution includes "Dropout" (e.g., rows 1, 3), "Graduate" (e.g., rows 2, 4), and "Enrolled" (e.g., row 17).
- Economic indicators vary widely (e.g., Unemployment rate from 7.6 to 16.2, GDP from -4.06 to 3.51).