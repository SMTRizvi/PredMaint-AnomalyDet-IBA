import streamlit as st
import pandas as pd
from check_1 import processing
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report
import shap
from check_1 import *

# Main Streamlit app code
def main():
    st.title("Predictive Analytics on Cloud")

    # File upload section
    st.subheader("Upload CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Model path input
        # model_path = st.text_input("Enter model path")
        # model_path = os.getcwd() 
        model_path = os.path.join(os.getcwd(), "model_lstm.h5")
        # print(model_path)

        # Preprocess the data and execute the model
        prob, Y_val, ypred, report = processing(df, model_path)

        # Display the results
        st.subheader("Predictions")
        count_of_1 = np.sum(ypred == 1)
        count_of_0 = np.sum(ypred == 0)

        st.write("The count of non-failures predicted ", count_of_0)
        st.write("The count of failures predicted ", count_of_1)
        st.write('ROC AUC: ',round(roc_auc_score(Y_val,prob),4))

        fig, ax = plt.subplots()
        ax.bar(["Non-Failures", "Failures"], [count_of_0, count_of_1], label='Predictions distributions')
        # ax.bar([0, 1], [0, 1], 'k--')  # Diagonal line
        
        ax.set_xlabel('State')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution')
        # ax.legend(loc='lower right')
        st.pyplot(fig)

        st.subheader("ROC Curve")
        
        fpr, tpr, thresholds = roc_curve(Y_val, prob)
        auc = roc_auc_score(Y_val, prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
        ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        # plt.show()

        df['sector']     = df['device'].str[:4]
        df['equipment']  = df['device'].str[4:]
         
        # count of devices
        st.subheader("Count of Unique Devices in Analysis")
        sector_counts = df['sector'].value_counts()
        equipment_counts = df['equipment'].value_counts()
        figure, ax = plt.subplots()
        ax.bar(["equipment"], [len(equipment_counts)], label='')
        ax.set_xlabel('Devices')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution')
        st.pyplot(figure)
if __name__ == '__main__':
    main()