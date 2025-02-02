import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from PIL import Image


def streamlit_menu(example=1):
        with st.sidebar:
            selected = option_menu(
            menu_title=None,  
            options=["Home", "Upload Data", "Track Expense", "Advice"],  
            icons=["house", "archive", "pie-chart-fill","card-checklist"],  
            menu_icon="cast",  
            default_index=0,  
            orientation="horizontal",
        )
        return selected

selected = streamlit_menu()

if selected == "Home":
    st.title("Welcome To Expense Tracker")

file_uploaded = False

if selected == "Upload Data":

    st.title(f"Please upload your bank statement for expense tracking")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the contents of the CSV
        st.write("Preview of Uploaded CSV:")
        st.dataframe(df)
        
        # File saving
        save_button = st.button("Save CSV File")
        if save_button:
            save_path = "input\\uploaded_file.csv"
            df.to_csv(save_path, index=False)
            st.success(f"File has been saved as {save_path}")
            file_uploaded = True
    os.system("python visualize.py")        



if selected == "Track Expense":
    st.title("Click on the Button To Know More")

    image_directory = "images"  
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('jpg', 'jpeg', 'png'))]
    custom_labels = image_files[:11]  
        
    # Display buttons dynamically with the custom labels
    custom_labels = [os.path.splitext(image_name)[0] for image_name in image_files[:10]]  

    # Display buttons dynamically with the custom labels
    for image_name, label in zip(image_files[:10], custom_labels):
        if st.button(f"{label}"):
            image_path = os.path.join(image_directory, image_name)
            image = Image.open(image_path)
            st.image(image, caption=f"{label}", use_container_width=True)
                                    
if selected == "Advice":
    st.title(f"Following are the things you can do")
    print("Exit Status:")
    with open("output.txt", "r") as file:
        content = file.read()  # Reads the entire file
        st.write(content)
   
