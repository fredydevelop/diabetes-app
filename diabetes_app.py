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

selection=option_menu(menu_title=None,options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],default_index=0,orientation="horizontal")


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def LungDetector(givendata):
    
    loaded_model=pk.load(open("The_New_Latest_Diabetes_Model.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("my_new_saved_std_scaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    if prediction==1:
      return "Diabetes is Detected"
    else:
      return "No Diabetes Detected"
    
 
#main function handling the input
def main():
    st.header("Diabetes Detection and prediction System")
    
    AGE = st.slider('Patient age', 0, 200, key="age")
    st.write("Patient is", AGE, 'years old')

    option1 = st.selectbox('Sex',("",'Male' ,'Female'),key="sex")
    if (option1=='Male'):
        GENDER=1
    else:
        GENDER=0

    option2 = st.selectbox('SMOKING',("",'Yes' ,'No'),key="smok")
    if (option2=='Yes'):
        SMOKING=1
    else:
        SMOKING=0

    option3 = st.selectbox('YELLOW_FINGERS',("",'Yes' ,'No'),key="yellow")
    if (option3=='Yes'):
        YELLOW_FINGERS=1
    else:
        YELLOW_FINGERS=0

    option4 = st.selectbox('ANXIETY',("",'Yes' ,'No'),key="anxiet")
    if (option4=='Yes'):
        ANXIETY=1
    else:
        ANXIETY=0

    option5 = st.selectbox('PEER_PRESSURE',("",'Yes' ,'No'),key="peer")
    if (option5=='Yes'):
        PEER_PRESSURE=1
    else:
        PEER_PRESSURE=0


    option6 = st.selectbox('CHRONIC DISEASE',("",'Yes' ,'No'),key="chronic")
    if (option6=='Yes'):
        CHRONIC_DISEASE=1
    else:
        CHRONIC_DISEASE=0


    option7 = st.selectbox('FATIGUE',("",'Yes' ,'No'),key="fatge")
    if (option7=='Yes'):
        FATIGUE=1
    else:
        FATIGUE=2


    option8 = st.selectbox('ALLERGY',("",'Yes' ,'No'),key="allerg")
    if (option8=='Yes'):
        ALLERGY=1
    else:
        ALLERGY=0

    option9 = st.selectbox('WHEEZING',("",'Yes' ,'No'),key="wheez")
    if (option9=='Yes'):
        WHEEZING=1
    else:
        WHEEZING=0


    option10 = st.selectbox('ALCOHOL CONSUMING',("",'Yes' ,'No'),key="alcohol")
    if (option10=='Yes'):
        ALCOHOL_CONSUMING=1
    else:
        ALCOHOL_CONSUMING=0

    
    option11 = st.selectbox('COUGHING',("",'Yes' ,'No'),key="cough")
    if (option11=='Yes'):
        COUGHING=1
    else:
        COUGHING=0

    option12 = st.selectbox('SHORTNESS OF BREATH',("",'Yes' ,'No'),key="shortbreath")
    if (option12=='Yes'):
        SHORTNESS_OF_BREATH=1
    else:
        SHORTNESS_OF_BREATH=0

    option13 = st.selectbox('SWALLOWING DIFFICULTY',("",'Yes' ,'No'),key="swallow")
    if (option13=='Yes'):
        SWALLOWING_DIFFICULTY=1
    else:
        SWALLOWING_DIFFICULTY=0

    option14 = st.selectbox('CHEST PAIN',("",'Yes' ,'No'),key="chestpain")
    if (option14=='Yes'):
        CHEST_PAIN=1
    else:
        CHEST_PAIN=0

    st.write("\n")
    st.write("\n")





    detectionResult = ''#for displaying result
    
    # creating a button for Prediction
    if age!=0 and option1!=""  and option2!=""  and option3!=""  and option4!="" and option5!="" and option6!="" and option7 !=""and  option8 !="" and option9!="" and option10 !="" and option11 !="" and option12 !="" and option13 !="" and option14 !=""  and st.button('Predict'):
        detectionResult = LungDetector(["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"])
        st.success(detectionResult)


def multi(input_data):
    loaded_model=pk.load(open("The_Latest_Diabetes_Model", "rb"))
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
    st.set_option('deprecation.showPyplotGlobalUse', False)
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
    
    
