from ensurepip import bootstrap
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# <--------------------------Global Variables-------------------------->

class_names = ["edible", "poisonous"]

# <--------------------------Functions-------------------------------->

@st.cache(persist=True)
def load_data():
    """
    Load data from csv file
    
    Returns: LabelEncoded dataframe
    """
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder() # LabelEncoder is used to convert categorical data to numerical data
    for col in data.columns:
        data[col] = label.fit_transform(data[col]) 
    return data

@st.cache(persist=True)
def split_data(df):
    """
    Split data into training and test set
    
    Returns: training and test dataframes
    """
    y = df["type"]
    x = df.drop("type", axis=1) # drop type column from x
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def plot_metrics(metrics_list):
    """
    Plot metrics
    
    Args:
        metrics_list: list of metrics
    """
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        disp_con = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp_con.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')
        st.pyplot(disp_con.figure_)
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        disp_roc.plot()
        st.pyplot(disp_roc.figure_)
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(x_test)[:, 1])
        disp_prc = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp_prc.plot()
        st.pyplot(disp_prc.figure_)


# <--------------------------Load Data-------------------------------->

df = load_data()
x_train, x_test, y_train, y_test = split_data(df)

original_data = pd.read_csv("mushrooms.csv")

# <-------------------------Streamlit App------------------------------>

st.title("Machine Learning Web App")
st.sidebar.title("Machine Learning Web App")
st.markdown("""Classification of mushrooms into edible and poisonous""")

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression",
                                 "Random Forest"))

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 0.1)
    
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", 
                                    "Precision-Recall Curve"))
                                    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

if classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 0.1)

    """
    C is regularization parameter. It controls the strength of regularization. 
    The smaller the value of C, the less regularization is used.
    """
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", 
                                    "Precision-Recall Curve"))
                                    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

if classifier == "Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Estimators", 100, 5000, key="n_estimators")
    # max_depth = st.sidebar.number_input("Maximum Depth", 1, 20, step=1, key="max_depth")
    max_depth = st.sidebar.slider("Maximum Depth", 1, 20, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap", ("True", "False"), key="bootstrap")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", 
                                    "Precision-Recall Curve"))
                                    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                    bootstrap=bootstrap, n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)