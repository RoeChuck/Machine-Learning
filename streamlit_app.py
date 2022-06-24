import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

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
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()


# <--------------------------Load Data-------------------------------->

df = load_data()
x_train, x_test, y_train, y_test = split_data(df)

original_data = pd.read_csv("mushrooms.csv")

# <-------------------------Streamlit App------------------------------>

st.title("Machine Learning Streamlit Web App")
st.sidebar.title("Machine Learning Streamlit Web App")
st.markdown("""Using binary classification to differentiate between edible and poisonous mushrooms""")

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression",
                                 "Random Forest"))

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")




















if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)
