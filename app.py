# Importing relevant libraries

# Data wrangling
import numpy as np
import pandas as pd
from tqdm import tqdm



# Importing Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import sys
# Data Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import subprocess
import sys


# Data pre-processing
from sklearn.preprocessing import StandardScaler

# Data splitting
from sklearn.model_selection import train_test_split

# Machine learning Models
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

# #Installing dependencies
# !pip install -U kaleido
# import seaborn as sns

# -----------------------------
# 🛠️ Utility Function to Install Kaleido
# -----------------------------
def install_kaleido():
    try:
        import kaleido
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "kaleido"])

install_kaleido()

# -----------------------------
# 🧪 Load Dataset
# -----------------------------
st.set_page_config(page_title="Student Dropout Risk Analysis", layout="wide")
st.title("\U0001F393 Student Dropout Risk Analysis Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv(r"student's_dropout_dataset.csv")

raw_data = load_data()

# Convert Target into Binary (1 = Dropout, 0 = Others)
data = raw_data.copy()
dummies = pd.get_dummies(data['Target'])
dummies.drop(['Enrolled', 'Graduate'], axis=1, inplace=True)
data['Target'] = dummies

# PCA analysis

# Extract columns for PCA
data_forPCA = data[['Curricular units 1st sem (credited)',
          'Curricular units 1st sem (enrolled)',
          'Curricular units 1st sem (evaluations)',
          'Curricular units 1st sem (without evaluations)',
          'Curricular units 1st sem (approved)',
          'Curricular units 1st sem (grade)',
          'Curricular units 2nd sem (credited)',
          'Curricular units 2nd sem (enrolled)',
          'Curricular units 2nd sem (evaluations)',
          'Curricular units 2nd sem (without evaluations)',
          'Curricular units 2nd sem (approved)',
          'Curricular units 2nd sem (grade)']]


# PCA with one component
pca = PCA(n_components=1)

# Fit PCA to data and transform it
pca_result = pca.fit_transform(data_forPCA)

# Create a new DataFrame with the reduced feature
df_pca = pd.DataFrame(data=pca_result, columns=['PCA Feature']).squeeze()
data['Curricular 1st and 2nd sem PCA'] = df_pca

# Dropping features
data.drop(['Nationality', 'Mother\'s occupation', 'Father\'s qualification',
          'Curricular units 1st sem (credited)',
          'Curricular units 1st sem (enrolled)',
          'Curricular units 1st sem (evaluations)',
          'Curricular units 1st sem (without evaluations)',
          'Curricular units 1st sem (approved)',
          'Curricular units 1st sem (grade)',
          'Curricular units 2nd sem (credited)',
          'Curricular units 2nd sem (enrolled)',
          'Curricular units 2nd sem (evaluations)',
          'Curricular units 2nd sem (without evaluations)',
          'Curricular units 2nd sem (approved)',
          'Curricular units 2nd sem (grade)', 'Inflation rate', 'GDP',
           'Unemployment rate'], axis = 1, inplace = True)
# -----------------------------
# ✂️ Feature Preparation & Splitting
# -----------------------------
# Assigning x and y features

X_features = data.drop('Target', axis = 1)
X_features.head()
y = data['Target']


# Normalizing data
scaler =  StandardScaler()
X = scaler.fit_transform(X_features)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load Mapped Data for Streamlit Display Purposes
df_mapped = pd.read_csv('mapped_student_dropout_dataset.csv')

# Dropping features
df_mapped.drop(['Nationality', 'Mother\'s occupation', 'Father\'s qualification',
          'Curricular units 1st sem (credited)',
          'Curricular units 1st sem (enrolled)',
          'Curricular units 1st sem (evaluations)',
          'Curricular units 1st sem (without evaluations)',
          'Curricular units 1st sem (approved)',
          'Curricular units 1st sem (grade)',
          'Curricular units 2nd sem (credited)',
          'Curricular units 2nd sem (enrolled)',
          'Curricular units 2nd sem (evaluations)',
          'Curricular units 2nd sem (without evaluations)',
          'Curricular units 2nd sem (approved)',
          'Curricular units 2nd sem (grade)', 'Inflation rate', 'GDP',
           'Unemployment rate'], axis = 1, inplace = True)

df_mapped['Curricular 1st and 2nd semester PCA'] = df_pca
X_test_stm = df_mapped.drop(columns=['Target']).iloc[df_mapped.index]

# -----------------------------
# ⚙️ Model Training: XGBoost
# -----------------------------
model_XGB = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model_XGB.fit(X_train, y_train)

# -----------------------------
# 💡 Risk Prediction and Categorization
# -----------------------------
y_pred_proba = model_XGB.predict_proba(X_test)[:, 1]

X_test = pd.DataFrame(X_test, columns=X_features.columns)

risk_scores = pd.Series(y_pred_proba, index=X_test.index)

def get_risk_category(prob):
    if prob <= 0.3:
        return 'Low Risk'
    elif prob <= 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'

risk_labels = risk_scores.apply(get_risk_category)

# -----------------------------
# 🎯 SHAP Analysis
# -----------------------------
shap.initjs()
explainer = shap.Explainer(model_XGB, X_train)
shap_values = explainer(X_test)

# -----------------------------
# 🧑 Student Selection and Display
# -----------------------------
st.sidebar.header("\U0001F50D Select Student (by index)")
student_idx = st.sidebar.number_input(
    "Enter student index (0 to {}):".format(len(X_test) - 1),
    min_value=0, max_value=len(X_test) - 1, value=0
)

student_data = X_test_stm.iloc[student_idx]
student_risk_score = risk_scores.iloc[student_idx]
student_risk_label = risk_labels.iloc[student_idx]
student_shap = shap_values[student_idx]

# -----------------------------
# 📋 Student Information
# -----------------------------
categorical_features = [
     'Course',
    'Previous qualification', 'Debtor',
    'Tuition fees up to date', 'Gender',
    'Scholarship holder', 'Age at enrollment'
]

st.subheader("\U0001F4CB Student Information")
cat_info = student_data[categorical_features].to_frame().reset_index()
cat_info.columns = ['Feature', 'Value']
st.table(cat_info)

# -----------------------------
# 📊 Risk Score & Category
# -----------------------------
st.subheader("\U0001F4CA Risk Assessment")
st.metric("Risk Score", f"{student_risk_score:.3f}")
st.metric("Risk Category", student_risk_label)

# -----------------------------
# 📈 SHAP Visualizations
# -----------------------------
st.subheader("\U0001F4CC Global Feature Importance (Bar Plot)")
shap.plots.bar(shap_values, max_display=15, show=False)
st.pyplot(plt.gcf())

st.markdown("**\U0001F50E Top 5 Global Features:**")
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
global_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP|': mean_abs_shap
}).sort_values(by='Mean |SHAP|', ascending=False)
st.dataframe(global_importance_df.head(5), use_container_width=True)

st.subheader("\U0001F41D SHAP Beeswarm Plot")
shap.plots.beeswarm(shap_values, max_display=15, show=False)
st.pyplot(plt.gcf())

st.subheader("\U0001F9CD Individual Student Risk Breakdown (Waterfall Plot)")
shap.plots.waterfall(student_shap, show=False)
st.pyplot(plt.gcf())

# -----------------------------
# 💬 Individual Feature Impact
# -----------------------------
st.markdown("**\U0001F50E Top Factors Influencing This Student's Risk:**")
indiv_explanation_df = pd.DataFrame({
    'Feature': student_data.index,
    'Value': student_data.values,
    'SHAP Value': student_shap.values
}).sort_values(by='SHAP Value', key=np.abs, ascending=False)
st.dataframe(indiv_explanation_df.head(5), use_container_width=True)

# -----------------------------
# 🔄 SHAP Force Plot
# -----------------------------
st.subheader("\U0001F501 SHAP Force Plot (Model View)")
st.markdown("_(Use Jupyter or save as HTML for full interactivity)_")

try:
    force_plot_html = shap.plots.force(
        explainer.expected_value, shap_values.values[student_idx], student_data, matplotlib=False
    )
    st.components.v1.html(shap.getjs() + force_plot_html.html(), height=300)
except:
    st.warning("Force plot rendering may not work in Streamlit. Run in Jupyter or export as HTML.")

# -----------------------------
# ✅ Footer
# -----------------------------
st.markdown("---")
st.caption("Developed with \U0001F4A1 SHAP & Streamlit · Customize further for your student database.")
