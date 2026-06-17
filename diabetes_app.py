#Importing the dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
import base64
import pickle as pk
from streamlit_option_menu import option_menu





#configuring the page setup
st.set_page_config(page_title='Diabetes prediction system',layout='centered')

st.image('diabot.jpeg', width=120,caption='DIABOT')
selection=option_menu(menu_title=None,options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],default_index=0,orientation="horizontal")


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href

@st.cache_resource
def load_model():
    return pk.load(open("The_Latest_Diabetes_Model.sav", "rb"))


@st.cache_resource
def load_scaler():
    return pk.load(open("my_saved_std_scaler.pkl", "rb"))

#single prediction function
def Diabetes(givendata):

    loaded_model = load_model()
    std_scaler_loaded = load_scaler()

    input_data_as_numpy_array = np.asarray(givendata)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    std_X_resample = std_scaler_loaded.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_X_resample)

    if prediction[0] == 1:
        return "Diabetes is Detected"
    else:
        return "No Diabetes Detected"
    
 
#main function handling the input

def main():

    st.header("Diabetes Detection and Prediction System")

    # ======================
    # AGE (mapped from dataset definition)
    # ======================
    age = st.number_input(
        "Age (13-level category: 1=18-24 ... 13=80+)",
        min_value=0,
        max_value=13,
        value=0,
        step=1
    )

    # ======================
    # CATEGORICAL INPUTS WITH FULL LABELS
    # ======================
    option1 = st.selectbox(
        "Sex (0 = Female, 1 = Male)",
        ("", "Male", "Female")
    )

    option2 = st.selectbox(
        "High Blood Pressure ",
        ("", "Yes", "No")
    )

    option3 = st.selectbox(
        "High Cholesterol",
        ("", "Yes", "No")
    )

    option4 = st.selectbox(
        "Heart Disease or Attack (CHD or MI)",
        ("", "Yes", "No")
    )

    option5 = st.selectbox(
        "Stroke (Ever told you had a stroke) (0 = No, 1 = Yes)",
        ("", "Yes", "No")
    )

    option6 = st.selectbox(
        "Physical Activity (past 30 days, not including job)",
        ("", "Yes", "No")
    )

    option7 = st.selectbox(
        "General Health",
        ("", "Poor", "Fair", "Good", "Very Good", "Excellent")
    )

    option8 = st.selectbox(
        "Difficulty Walking (serious difficulty walking/climbing stairs)",
        ("", "Yes", "No")
    )

    option9 = st.selectbox(
        "Fruit Consumption (1+ times per day)",
        ("", "Yes", "No")
    )

    option10 = st.selectbox(
        "Vegetable Consumption (1+ times per day)",
        ("", "Yes", "No")
    )

    option11 = st.selectbox(
        "Education Level (1–6 scale from no school to doctorate)",
        (
            "",
            "Less than high school education",
            "High school education",
            "College or Associate",
            "Bachelor's degree",
            "Master's degree",
            "Doctoral degree"
        )
    )

    # ======================
    # NUMERIC INPUTS
    # ======================
    BMI = st.slider("BMI (Body Mass Index)", 10, 60, 25)

    PhysHlth = st.number_input(
        "Physical Health (Days of illness/injury in past 30 days)",
        min_value=0,
        max_value=30,
        value=0
    )

    Income = st.number_input(
        "Income (1–8 scale: 1 < $10k ... 8 ≥ $75k)",
        min_value=0,
        max_value=8,
        value=0
    )

    # ======================
    # VALIDATION
    # ======================
    all_filled = (
        age > 0 and
        option1 != "" and
        option2 != "" and
        option3 != "" and
        option4 != "" and
        option5 != "" and
        option6 != "" and
        option7 != "" and
        option8 != "" and
        option9 != "" and
        option10 != "" and
        option11 != ""
    )

    # ======================
    # ENCODING
    # ======================
    Sex = 1 if option1 == "Male" else 0
    HighBP = 1 if option2 == "Yes" else 0
    HighChol = 1 if option3 == "Yes" else 0
    HeartDiseaseorAttack = 1 if option4 == "Yes" else 0
    Stroke = 1 if option5 == "Yes" else 0
    PhysActivity = 1 if option6 == "Yes" else 0
    Fruits = 1 if option9 == "Yes" else 0
    Veggies = 1 if option10 == "Yes" else 0
    DiffWalk = 1 if option8 == "Yes" else 0

    health_mapping = {
        "Poor": 1,
        "Fair": 2,
        "Good": 3,
        "Very Good": 4,
        "Excellent": 5
    }
    GenHlth = health_mapping.get(option7, 0)

    education_mapping = {
        "Less than high school education": 1,
        "High school education": 2,
        "College or Associate": 3,
        "Bachelor's degree": 4,
        "Master's degree": 5,
        "Doctoral degree": 6
    }
    Education = education_mapping.get(option11, 0)

    # ======================
    # PREDICTION
    # ======================
    if st.button("Predict"):

        if not all_filled:
            st.error("Please complete all fields and ensure Age > 0 before predicting.")
            return

        input_data = [
            HighBP,
            HighChol,
            BMI,
            Stroke,
            HeartDiseaseorAttack,
            PhysActivity,
            Fruits,
            Veggies,
            GenHlth,
            PhysHlth,
            DiffWalk,
            Sex,
            age,
            Education,
            Income
        ]

        detectionResult = Diabetes(input_data)

        if detectionResult == "Diabetes is Detected":
            st.error(detectionResult)
        else:
            st.success(detectionResult)


def multi(input_data):
    loaded_model=pk.load(open("The_Latest_Diabetes_Model.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    # dfinput=dfinput.iloc[1:].reset_index(drop=True)

    st.header('A view of the uploaded dataset')
    st.markdown('')
    st.dataframe(dfinput)

    dfinput=dfinput.values
    std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
    std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
    predict=st.button("predict")


    if predict:
        prediction = loaded_model.predict(std_dfinput)
        interchange=[]
        for i in prediction:
            if i==1:
                newi="Diabetes Detected"
                interchange.append(newi)
            elif i==0:
                newi="No Diabetes"
                interchange.append(newi)
            
        st.subheader('Here is your prediction')
        prediction_output = pd.Series(interchange, name='Diabetics results')
        prediction_id = pd.Series(np.arange(len(interchange)),name="Patient_ID")
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

if selection =="Single Prediction":
    main()

if selection == "Multi Prediction":
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    
    
