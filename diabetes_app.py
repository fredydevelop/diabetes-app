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

st.image('logo.jpg', width=120,caption='SMART ABETES')
selection=option_menu(menu_title=None,options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],default_index=0,orientation="horizontal")


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def Diabetes(givendata):
    
    loaded_model=pk.load(open("The_Latest_Diabetes_Model.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    if prediction==1:
      return "Diabetes is Detected"
    else:
      return "No Diabetes Detected"
    
 
#main function handling the input
def main():
    st.header("Diabetes Detection and prediction System")
    
    #getting user input
    
    age = st.slider('Patient age', 0, 200, key="age")
    st.write("Patient is", age, 'years old')

    option1 = st.selectbox('Sex',("",'Male' ,'Female'),key="sex")
    if (option1=='Male'):
        Sex=1
    else:
        Sex=0

    option2 = st.selectbox('HighBP',("",'Yes' ,'No'),key="highbp")
    if (option2=='Yes'):
        HighBP=1
    else:
        HighBP=0

    option3 = st.selectbox('High Cholesterol',("",'Yes' ,'No'),key="high_chol")
    if (option3=='Yes'):
        HighChol=1
    else:
        HighChol=0

    option4 = st.selectbox('Heart Disease or Attack',("",'Yes' ,'No'),key="heart_disease")
    if (option4=='Yes'):
        HeartDiseaseorAttack=1
    else:
        HeartDiseaseorAttack=0


    BMI = st.slider('Patient BMI', 0, 200, key="bmi")
    st.write("Patient BMI is", BMI)

    
#
    option5 = st.selectbox('Stroke',("",'Yes' ,'No'),key="stroke")
    if (option5=='Yes'):
        Stroke=1
    else:
        Stroke=0


    

    option6 = st.selectbox('PhysActivity',("",'Yes' ,'No'),key="physical_activity")
    if (option6=='Yes'):
        PhysActivity=1
    else:
        PhysActivity=0


    option7 = st.selectbox('GenHlth',("",'Poor' ,'Fair',"Good","Very Good","Excellent"),key="GenHlth")
    if (option7=='Poor'):
        GenHlth=1

    elif (option7=='Fair'):
        GenHlth=2
    
    elif (option7=='=Good'):
        GenHlth=3

    elif (option7=='Very Good'):
        GenHlth=4

    else:
        GenHlth=5


    option8 = st.selectbox('Difficult in walking',("",'Yes' ,'No'),key="DiffWalk")
    if (option8=='Yes'):
        DiffWalk=1
    else:
        DiffWalk=0

    option9 = st.selectbox('Fruit Consumption',("",'Yes' ,'No'),key="Fruits")
    if (option9=='Yes'):
        Fruits=1
    else:
        Fruits=0


    option10 = st.selectbox('Veggies',("",'Yes' ,'No'),key="Veggies")
    if (option10=='Yes'):
        Veggies=1
    else:
        Veggies=0

    
    option11 = st.selectbox('Education',("",'Less than high school education' ,'High school education','college or associate', "Bachelor's degree","Master's degree","Doctoral degree"),key="Education")
    if (option11=='Less than high school education'):
        Education=1

    elif (option11=='High school education'):
        Education=2
    
    elif (option11=='=college or associate'):
        Education=3

    elif (option11== "Bachelor's degree"):
        Education=4

    elif (option11=="Master's degree"):
        Education=5

    else:
        Education=5


    Income = st.slider('Patient Income * 100', 0, 10000, key="income")
    st.write("Patient income is", Income)


    PhysHlth = st.slider('What is the level of Patient Health', 0, 60, key="PhysHlth")
    


    st.write("\n")
    st.write("\n")




    detectionResult = ''#for displaying result
    
    # creating a button for Prediction
    if age!="" and option1!=""  and option2!=""  and option3!=""  and option4!="" and option5!="" and option6!="" and option7 !=""and  option8 !="" and option9!="" and option10 !="" and option11 !=""  and st.button('Predict'):
        detectionResult = Diabetes([HighBP,HighChol,BMI,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,GenHlth,PhysHlth,DiffWalk,Sex,age,Education,Income,])
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
    
    
