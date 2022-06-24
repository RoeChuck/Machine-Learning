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

# <--------------------------Functions-------------------------------->

@st.cache(persist=True)
def load_data():
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder() # LabelEncoder is used to convert categorical data to numerical data
    for col in data.columns:
        data[col] = label.fit_transform(data[col]) 
    return data

# <--------------------------Load Data-------------------------------->

data = load_data()
original_data = pd.read_csv("mushrooms.csv")

# <-------------------------Streamlit App------------------------------>
st.title("Machine Learning Streamlit Web App")
st.sidebar.title("Machine Learning Streamlit Web App")
st.markdown("""Using binary classification to differentiate between edible and poisonous mushrooms""")
