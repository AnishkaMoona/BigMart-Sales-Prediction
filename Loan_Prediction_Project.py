# Loan Prediction Analysis - Classification


# Commented out IPython magic to ensure Python compatibility.
#importing libraries

#to do operations with arrays and matrices
import numpy as np 

#store the dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# %matplotlib inline
warnings.filterwarnings('ignore')

#for the webapp
import streamlit as st
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def load_data():
    return pd.read_csv('Loan Prediction Dataset.csv')

def data_eda(df):
    df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean())

    df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
    df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Married']=df['Married'].fillna(df['Married'].mode()[0])

    return df

def log(df):
    #total income
    df['Total_Income']=df['ApplicantIncome']+df['CoapplicantIncome']

    df['ApplicantIncome_Log']=np.log(df['ApplicantIncome'])
    df['LoanAmount_Log']=np.log(df['LoanAmount'])
    df['LoanTerm_Log']=np.log(df['Loan_Amount_Term'])
    df['TotalIncome_Log']=np.log(df['Total_Income'])

    return df

def classify(str):
    if str == 'Y':
        return 'Congratulations! you are eligible to apply for a loan'
    else:
        return 'Sorry, you cannot apply for a loan.'

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Prediction: WebApp</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    df = load_data()

    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Preprocessing and Exploration', 'Prediction','About'])

    if page == 'Homepage':
        """About the dataset:
            1. Loan_ID: Unique Loan ID
            2. Gender: Male/Female
            3. Married Applicant (Y/N)
            4. Dependents: Number of dependents
            5. Applicant education: Graduate/Under Graduate
            6. Self employed: (Y/N)
            7. Applicant Income
            8. Co-applicant Income
            9. Loan amount in thousands
            10. Loan amount term in months
            11. Credit History - meets guidelines
            12. Property area: Urban/Semi Urban/Rural
            13. Loan Status: Approved yes/no"""

        st.title('Loan Prediction & Analysis')
        st.write('Select a page in the sidebar')
        st.dataframe(df)
        if st.checkbox('Show Column Descriptions'):
            st.dataframe(df.describe())
    elif page == 'Preprocessing and Exploration':
        st.title('Explore the Loan Prediction Data-set')
        st.write('There were null values in the dataset, this was handled. Numerical data handled by replacing with mean. Categorical data handled by replacing with mode.')
        
        option = st.selectbox("Choose an option", ['Check if null values', 'EDA', 'Log Transformation'])
        if option == 'Check if null values':
            df = data_eda(df)
            st.dataframe(df.isnull().sum())
        elif option == 'EDA':
            sns.set_theme(style="darkgrid")
            st.write('Countplots for categorical data')
            fig, ax = plt.subplots(3,2,figsize=(20,20))
            
            sns.countplot(df['Gender'], ax=ax[0][0])
            sns.countplot(df['Married'], ax=ax[0][1])
            sns.countplot(df['Education'], ax=ax[1][0])
            sns.countplot(df['Self_Employed'], ax=ax[1][1])
            sns.countplot(df['Property_Area'], ax=ax[2][0])
            sns.countplot(df['Loan_Status'], ax=ax[2][1])
            fig.show()
            st.pyplot()

            st.write('Distplot for numerical data')
            fig, ax = plt.subplots(2,2,figsize=(20,20))

            sns.distplot(df['ApplicantIncome'], ax=ax[0][0], color='r')
            sns.distplot(df['CoapplicantIncome'], ax=ax[0][1], color='g')
            sns.distplot(df['LoanAmount'], ax=ax[1][0], color='r')
            sns.distplot(df['Loan_Amount_Term'], ax=ax[1][1], color='g')

            fig.show()
            st.pyplot()
        elif option == 'Log Transformation':
            st.write('Added a new attribute, total income for analysis. Applied log transformation to transform skewed data to approximately normality. Data transformation technique to replace x with log(x)')

            df = log(df)

            st.write('Log Transformed Distplots')

            fig, ax = plt.subplots(2,2,figsize=(20,20))

            sns.distplot(df['ApplicantIncome_Log'], ax=ax[0][0], color='r')
            sns.distplot(df['LoanAmount_Log'], ax=ax[1][0], color='r')
            sns.distplot(df['LoanTerm_Log'], ax=ax[0][1], color='g')
            sns.distplot(df['TotalIncome_Log'],ax=ax[1][1],color='g')

            fig.show()
            st.pyplot()

            st.write('Co-relation Analysis, Matrix')

            corr = df.corr()
            plt.figure(figsize=(15,10))
            sns.heatmap(corr, annot = True, cmap="BuPu")
            st.pyplot()
    elif page == 'Prediction' :
        st.title('Data Models & Prediction')
        st.write('Data has been split into train and test and label encoded, as we have a mix of categorical as well as numerical data')

        df = data_eda(df)
        dfnew = log(df)

        # drop unnecessary columns
        cols = ['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income','TotalIncome_Log']
        dfnew = dfnew.drop(columns=cols, axis=1)

        #input and output attributes
        x = dfnew.drop(columns=['Loan_Status'],axis=1)
        y = dfnew['Loan_Status']

        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

        le1 = LabelEncoder()
        x_train['Gender'] = le1.fit_transform(x_train['Gender']) 
        x_test['Gender']=le1.transform(x_test['Gender'])

        le2 = LabelEncoder()
        x_train['Married'] = le2.fit_transform(x_train['Married']) 
        x_test['Married']=le2.transform(x_test['Married'])

        le3 = LabelEncoder()
        x_train['Education'] = le3.fit_transform(x_train['Education']) 
        x_test['Education']=le3.transform(x_test['Education'])

        le4 = LabelEncoder()
        x_train['Self_Employed'] = le4.fit_transform(x_train['Self_Employed']) 
        x_test['Self_Employed']=le4.transform(x_test['Self_Employed'])

        le5 = LabelEncoder()
        x_train['Property_Area'] = le5.fit_transform(x_train['Property_Area']) 
        x_test['Property_Area']=le5.transform(x_test['Property_Area'])

        le6 = LabelEncoder()
        x_train['Dependents'] = le6.fit_transform(x_train['Dependents']) 
        x_test['Dependents']=le6.transform(x_test['Dependents'])

        model_svm = SVC()
        model_svm.fit(x_train,y_train)
        svm_score = model_svm.score(x_test,y_test)*100

        model_dt = DecisionTreeClassifier()
        model_dt.fit(x_train,y_train)
        dt_score = model_dt.score(x_test,y_test)*100

        model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, verbose=1, max_features=1)
        model_rf.fit(x_train,y_train)
        rf_score = model_rf.score(x_test,y_test)*100

        gender_choice = ['Male','Female']
        gender = st.selectbox('Please select your gender', gender_choice)
        
        choice_yn = ['Yes','No']
        married = st.selectbox('Are you married?', choice_yn)
        
        dependents_choice = ['0','1','2','3+']
        dependents = st.selectbox('Select number of dependents', dependents_choice)
        
        edu_choice = ['Not Graduate','Graduate']
        education = st.selectbox('Select your education level', edu_choice)
        
        self_employ = st.selectbox('Select YES if you are self employed', choice_yn)
        
        credit = ['0','1']
        credit_history = st.selectbox('Does your credit history meet guidelines? Select 1 if yes', credit)
        
        prop = ['Urban','Semiurban', 'Rural']
        property_area = st.selectbox('Select the type of your location area', prop)
        
        ai = st.number_input('Input the income of applicant applying for loan', 1, 100000)
        
        la = st.number_input('Enter the amount of loan :', min_value=1, max_value=50000, step=1)
        
        lt = st.number_input('Enter the term of loan, in months :', min_value=12, max_value=1200, step=1)
        
        inputs = [gender,married,dependents,education,self_employ,credit_history,property_area,ai,la,lt]

        models = st.selectbox("Select a model for prediction", ['Support Vector Machine', 'Decision Tree Classifier', 'Random Forest'])

        inputs[0] = int(le1.transform([inputs[0]]))
        inputs[1] = int(le2.transform([inputs[1]]))
        inputs[2] = int(le6.transform([inputs[2]]))
        inputs[3] = int(le3.transform([inputs[3]]))
        inputs[4] = int(le4.transform([inputs[4]]))
        inputs[6] = int(le5.transform([inputs[6]]))
        inputs[7] = np.log(inputs[7])
        inputs[8] = np.log(inputs[8])
        inputs[9] = np.log(inputs[9])
            
        output_svm = model_svm.predict([inputs])
        output_dt = model_dt.predict([inputs])
        output_rf = model_rf.predict([inputs])

        if st.button('Submit and Predict'):
            if models == 'Support Vector Machine':
                st.success(classify(output_svm[0]))
                st.write('Accuracy score with Support Vector Machine is ' + str(svm_score))
            elif models == 'Decision Tree Classifier':
                st.success(classify(output_dt[0]))
                st.write('Accuracy score with Decision Tree is ' + str(dt_score))
            else:
                st.success(classify(output_rf[0]))
                st.write('Accuracy score with Random Forest is ' + str(rf_score))
                st.write('A single tree in Random forest with max depth 10 and n_estimators = 100: ')

                fig = plt.figure(figsize=(50,50))
                _ = tree.plot_tree(model_rf.estimators_[1], class_names= x_train.columns,filled=True)
        
                plt.axis('off')
                plt.show()
                
                st.pyplot()

                """Performance Measures: Test Data Accuracy and Confusion Matrix"""

                y_pred_test = model_rf.predict(x_test)
                confusion_matrix = pd.crosstab(y_test, y_pred_test, rownames=['Actual'],colnames=['Predicted'])
                sns.heatmap(confusion_matrix,annot=True)
                st.pyplot()
    else:
        """
        Problem Statement: 
        
        Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. 
        
        To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.
        
        This is a standard supervised classification task. A classification problem where we have to predict whether a loan would be approved or not.

        The Homepage consits the dataset and information about it.
        The steps applied to preprocess the data can be viewed in the second page, along with exploratory data analysis. Finally, 3 different machine learning algorithms have been use to classify the customers and predictions are made."""

if __name__ == '__main__':
    main()