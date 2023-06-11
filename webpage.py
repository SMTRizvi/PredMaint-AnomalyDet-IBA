import streamlit as st
import pandas as pd
import pickle
from tensorflow import keras
from check_1 import processing

st.set_page_config(layout="wide")

# Load the trained model
model = keras.models.load_model('model_lstm.h5')

st.title('Predictive Maintenance Platform on Cloud')

st.write('This is a predictive maintenance platform. '
         'The system predicts failure of machines based on trained data '
         'and provides other relevant machine data.')

# Main Streamlit app code
def main():
    st.title("Equipment Data Stream")

    # Sidebar options
    st.sidebar.header("Options")
    generate_work_order = st.sidebar.button("Generate Work Order")

    if generate_work_order:
        st.sidebar.write("Create Work Order for Equipment", unsafe_allow_html=True)
        st.sidebar.button("Create Work Order", key="create_work_order_button")

    if st.sidebar.button("View Maintenance Schedule"):
        # Add your code for viewing maintenance schedule here
        st.write("Retrieving maintenance schedule from ERP.")

    if st.sidebar.button("Generate Notification"):
        # Add your code for generating notifications here
        st.write("Generate Notification for affected equipment.")

    # View Predictive Data
    if st.sidebar.button("View Predictive Data"):
        # Load the CSV file
        csv_file = st.file_uploader("Upload CSV file", type="csv")
        if csv_file is not None:
            df = pd.read_csv(csv_file)

            st.header("Choose your ML model")
            model_path = st.text_input("Enter the model path")

            
            st.subheader("Loaded Data:")
            st.write(df)

            form_submitted = st.form_submit_button("Process")

            if form_submitted:
                # Call the processing function
                result = processing(df, model_path)

                # Display the result
                st.subheader("Processing Result")
                st.write(result)

            # Make predictions on the loaded data
            #predictions = model.predict(data)

                st.subheader("Predictions:")
                st.write(result)
            #print (predictions)



if __name__ == '__main__':
    main()
