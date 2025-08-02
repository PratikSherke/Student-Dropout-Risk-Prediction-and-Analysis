# Predicting and Analyzing Student Dropout Risks using Supervised ML Algorithms
*An interactive machine learning dashboard to predict and analyze student dropout risk*

## 📝 Project Overview
The **Student Dropout Risk Analysis Dashboard** is a comprehensive web application built with Streamlit, designed to predict student dropout risk using a variety of supervised machine learning classifiers. The project evaluates Logistic Regression, Decision Trees, Support Vector Machines (SVM), Random Forest, K-Nearest Neighbors (KNN), and XGBoost, with XGBoost selected for the dashboard due to its superior performance. It includes extensive exploratory data analysis (EDA), feature engineering with PCA, and SHAP-based explainability, enhanced by GridSearchCV for hyperparameter tuning, learning curves, ROC curves, and confusion matrices. The dashboard provides interactive visualizations to help educators identify at-risk students and understand key influencing factors.

### Key Features
- **Multi-Model Prediction**: Compares Logistic Regression, Decision Trees, SVM, Random Forest, KNN, and XGBoost.
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize models like Random Forest and KNN.
- **Feature Engineering**: Applies PCA to reduce dimensionality of curricular unit features, addressing multicollinearity.
- **Explainability**: Employs SHAP for transparent predictions with bar, beeswarm, waterfall, and force plots.
- **Interactive Dashboard**: Built with Streamlit, allowing users to select students and explore risk scores and feature impacts.
- **Exploratory Data Analysis (EDA)**: Includes pie charts, histograms, bar plots, and correlation heatmaps.
- **Performance Metrics**: Features learning curves, ROC curves (AUC 0.87 for XGBoost), and confusion matrices.
- **Risk Categorization**: Classifies students into Low (<0.3), Medium (0.3–0.7), and High (>0.7) risk.
- **Customizable**: Adaptable to other datasets with similar features.

## 🛠️ Technologies Used
- **Programming Language**: Python 3.10+
- **Machine Learning**: scikit-learn (Logistic Regression, Decision Trees, SVM, Random Forest, KNN, PCA, StandardScaler, GridSearchCV), XGBoost
- **Explainability**: SHAP
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, seaborn
- **Web Framework**: Streamlit
- **Dependencies**: kaleido (for Plotly export), tqdm (progress bars)
- **Environment**: Jupyter Notebook, Google Colab (optional)

## 📊 Dataset
The project uses the "Student Dropout Dataset" (`student's_dropout_dataset.csv`), with 4,424 records and 35 features:
- **Demographic**: Marital status, gender, age at enrollment, nationality
- **Academic**: Course, previous qualifications, curricular units (credited, enrolled, evaluations, approved, grades)
- **Socio-economic**: Scholarship holder, debtor status, tuition fees status, economic indicators (unemployment rate, inflation rate, GDP)
- **Target**: Dropout, Graduate, Enrolled (converted to binary: Dropout = 1, Non-Dropout = 0)

A mapped dataset (`mapped_student_dropout_dataset.csv`) provides readable categorical values. See [Dataset Description](dataset_description.md) for details.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (optional, for cloning)
- Datasets: `student's dropout dataset.csv` and `mapped_student_dropout_dataset.csv`

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/student-dropout-prediction.git
   cd student-dropout-prediction
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Place `student's dropout dataset.csv` and `mapped_student_dropout_dataset.csv` in the project root.
   - Update file paths in `app.py` if datasets are elsewhere.

5. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

6. **Access the Dashboard**:
   - Open `http://localhost:8501` in your browser.
   - For remote access, use `localtunnel`:
     ```bash
     npx localtunnel --port 8501
     ```

### Project Structure
```
student-dropout-prediction/
├── app.py                                     # Main Streamlit application
├── student's_dropout_dataset.csv                 # Original dataset
├── mapped_student_dropout_dataset.csv            # Mapped dataset
├── requirements.txt                              # Dependencies
├── README.md                                     # Documentation
├── dataset_description.md                        # Dataset details
├── student's_dropout_prediction_projejct.ipynb   #Jupyter notebooks

```

## 📈 Usage
1. **Launch the Dashboard**: Run `streamlit run app.py`.
2. **Select a Student**: Enter a student index (0 to N-1) in the sidebar.
3. **Explore Results**:
   - **Student Information**: View categorical features (e.g., Gender, Course).
   - **Risk Assessment**: Check risk score and category.
   - **SHAP Visualizations**: Analyze global (bar, beeswarm) and individual (waterfall, force) feature impacts.
   - **Model Performance**: Review accuracy (84% for XGBoost), AUC (0.87), and confusion matrices.
4. **Interpret Insights**: Use SHAP tables to identify key risk factors.

## 🧪 Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Visualized target distribution (49% Graduate, 32.1% Dropout, 17% Enrolled).
   - Analyzed age, gender, course, marital status, and financial status with pie charts, histograms, and bar plots.
   - Used correlation heatmaps to detect multicollinearity.
2. **Data Preprocessing**:
   - Corrected column names (e.g., 'Nacionality' to 'Nationality').
   - Converted target to binary (Dropout = 1, Others = 0).
   - Applied PCA to reduce 12 curricular unit features into one component.
   - Dropped correlated features (e.g., Nationality, Mother’s occupation).
   - Normalized features with StandardScaler.
3. **Model Training**:
   - Trained Logistic Regression, Decision Trees, SVM, Random Forest, KNN, and XGBoost.
   - Used GridSearchCV to tune hyperparameters (e.g., max_depth for Decision Trees, n_estimators for Random Forest).
   - Generated learning curves to select optimal hyperparameters.
   - Selected XGBoost for the dashboard (78% accuracy).
4. **Evaluation**:
   - Calculated accuracies: Logistic Regression (81%), Decision Trees (83.2%), SVM (80.9%), Random Forest (82.8%), KNN (79.5%), XGBoost (84%).
   - Plotted ROC curves (AUC 0.87 for XGBoost) and confusion matrices for all models.
5. **Explainability**:
   - Used SHAP for global and individual feature importance.
   - Visualized with bar, beeswarm, waterfall, and force plots.
6. **Dashboard**:
   - Built with Streamlit for interactive exploration.
   - Displays student details, risk scores, categories, and SHAP visualizations.

## 📊 Results
- **Model Performance**:
  - **Baseline Accuracy**: ~68% (majority class).
  - **Actual Accuracies**: Logistic Regression (81%), Decision Trees (83.2%), SVM (80.9%), Random Forest (82.8%), KNN (79.5%), XGBoost (84%).
  - **AUC Score**: 0.87 (XGBoost).
- **Key Predictors**: Curricular performance (PCA feature), age at enrollment, course, and financial status (e.g., tuition fees, scholarships).
- **Risk Categories**: Low (<0.3), Medium (0.3–0.7), High (>0.7).
- **EDA Insights**: Nursing has a low dropout rate (15.4%), while Biofuel Production Technologies (66.7%) and legally separated students (66.7%) show high dropout rates.

## 💡 Future Improvements
- Add real-time data input or API integration.
- Experiment with ensemble methods or neural networks.
- Include additional features (e.g., attendance, mental health).
- Enhance dashboard with filters for risk categories or features.
- Deploy on Streamlit Cloud or Heroku.

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments
- **Dataset**: Publicly available student dropout dataset.
- **Libraries**: Streamlit, SHAP, XGBoost, scikit-learn, Plotly, Matplotlib, seaborn.
- **Inspiration**: Supporting education with data-driven insights.

## 📬 Contact
For questions or contributions, open an issue or contact [your-email@example.com](mailto:your-email@example.com).

---

*Developed with 💡 SHAP, Streamlit, and XGBoost · Updated on August 02, 2025.*
