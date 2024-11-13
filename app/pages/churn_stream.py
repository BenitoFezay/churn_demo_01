# -------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# STREMLIT APP
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

# PICKEL
import pickle

# -------------------------------------
# 1- Encoding the data using labelencoder
def make_encoding_labelencoder(df, columns):
       label_encoder = LabelEncoder()
       for col in columns:
              df[col] = label_encoder.fit_transform(df[col])
       return df

# -------------------------------
# 2- Scalling data with StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns = ['MontantTrans', 'ScoreCSAT', 'ScoreNPS', 'AgeCompte (j)', 'AgeClient', 'MontantPret', 'TauxInteret']
def making_scaler_standardscaler(df):
       df[columns] = scaler.fit_transform(df[columns])
       return df

# --------------------------------
# # 3- SPLITTING THE DATA
# # 3.1- RESAMPLING 
# from imblearn.over_sampling import SMOTE
# smote = SMOTE()

# # 3.2- Using train and test split
# from sklearn.model_selection import train_test_split

# # -------------------------------
# # 4- CREATE MODEL
# from sklearn.linear_model import LogisticRegression

# # -------------------------------
# # 5- ACCURACY
# import sklearn.metrics as sm

# # ------------------------------
# # 6- CONFUSION-MATRIX
# from sklearn.metrics import confusion_matrix as cm

# # ------------------------------
# # 7- CLASSIFICATION_REPORT
# from sklearn.metrics import classification_report as cr

# # ------------------------------
# # 8- CROSS VALIDATION
# from sklearn.model_selection import cross_val_score, StratifiedKFold

# # ------------------------------
# # 9- BOOSTING MODEL
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score

# -------------------------------
# 10- SAVING MODEL
model_path = './model/churn_model.pkl'

# To load the model later:
churn_model = pickle.load(open(model_path, 'rb'))


# -------------------------------
# 11- TESTING THE MODEL
def testing_model_by_ilocation(data):
    # new_data = X_test.iloc[data]  # Example: Use the first row of X_test for testing
    prediction = churn_model.predict(data)
    print(f"Prediction for the new data point: {prediction}")

    if f"{prediction}" == "[1]":
        # st.write(new_data)
        return st.error("This Client is a churner")
    else:
        # st.write(new_data)
        return st.success("This Client is a loyal")


# ------------------------------
st.title("🤖 Machine Learning")

# --------------------------------
# 12- Churn prediction BY FILL FIELDS
def make_data_encoded(data):
    # ['TypeCompte', 'MontantTrans', 'TypeTransaction', 'ScoreCSAT',
    #    'ScoreNPS', 'AgeCompte (j)', 'AgeClient', 'Ville', 'MontantPret',
    #    'TauxInteret', 'TypeEngagement']

    # Type account ------------------------
    if data["TypeCompte"] == "Compte Courant":
        data["TypeCompte"] = 1 
    else:
        data["TypeCompte"] = 0 

    # Type engagement ---------------------
    if data["TypeEngagement"] == "Utilisation d'application mobile":
        data["TypeEngagement"] = 0
    else:
        data["TypeEngagement"] = 1

    # Type transaction --------------------
    if data["TypeTransaction"] == "Paiement":
        data["TypeTransaction"] = 0
    elif data["TypeTransaction"] == "Retrait":
        data["TypeTransaction"] = 1
    elif data["TypeTransaction"] == "Virement":
        data["TypeTransaction"] = 2
    else:
        data["TypeTransaction"] = 3

    # Ville --------------------------------
    villes = ['Antsohihy', 'Ambanja', 'Antananarivo', 'Ihosy', 'Toliara',
       'Antsiranana', 'Fianarantsoa', 'Toamasina', 'Fort Dauphin',
       'Mahajanga', 'Andapa', 'Ambovombe', 'Morondava', 'Nosy Be'] 

    for i, ville in enumerate(villes):
        if data["Ville"] == ville:
            data["Ville"] = i

# 13- Fields for churn prediction
with st.expander("CHURN PREDICTION - BY FILLING FIELDS"):
       
       st.write("#### Input Data")
       with st.form(key="Churn_form"):
        
              col1, col2, col3 = st.columns(3)
              # montant_transaction
              transaction_amount = col1.number_input("Montant Transaction (ar)", min_value=2_000, value=450_000)
              # Score_csat
              score_csat = col1.slider('Score CSAT client', 0, 10, 3)
              # Montant pret
              loan_amount = col1.number_input("Montant pret", min_value=0, value=0)
              # taux d'interet
              taux_interet = col2.slider("Taux d'interet", 0.00, 5.00, 3.12)
              # age compte 
              age_compte = col2.number_input("Age compte (j)", min_value=3, value=2300)
              # scorre nps
              score_nps = col2.slider('Score NPS client', 0, 10, 3)
              # type compte
              type_account = col3.selectbox('Type de Compte', ('Compte Courant', 'Compte Épargne'))
              # type engagement
              type_engagement = col3.selectbox("Type d'engagement", ('Utilisation application mobile', 'Participation de programme de fidelité'))
              # type transaction
              type_transaction = col3.selectbox("Type de Transaction", ('Virement', 'Paiement' ,'Retrait', 'Depôt'))
              # age client
              age_client = col3.slider('Age client', 18, 100, 47)
              # ville
              ville = col2.selectbox("Adresse", ('Antsohihy', 'Ambanja', 'Antananarivo', 'Ihosy', 'Toliara',
               'Antsiranana', 'Fianarantsoa', 'Toamasina', 'Fort Dauphin',
               'Mahajanga', 'Andapa', 'Ambovombe', 'Morondava', 'Nosy Be'))
              # button submit
              submit_button = st.form_submit_button(label="Valide & predict") 
        

              data = {
               "TypeCompte": type_account,
               'MontantTrans': transaction_amount, 
               'TypeTransaction': type_transaction, 
               'ScoreCSAT': score_csat,
               'ScoreNPS': score_nps, 
               'AgeCompte (j)': age_compte, 
               'AgeClient': age_client, 
               'Ville': ville, 
               'MontantPret': loan_amount,
               'TauxInteret': taux_interet, 
               'TypeEngagement': type_engagement
              }
              
              make_data_encoded(data=data)
              
              if submit_button:
               df = pd.DataFrame(data, index=[0])
               # Scalling input data
               from sklearn.preprocessing import StandardScaler
               columns = {'MontantTrans': {"mean":0.015519 , "std": 0.554258}, 
                       'ScoreCSAT': {"mean":-0.101133 ,"std":0.572905}, 
                       'ScoreNPS': {"mean":0.054224, "std":0.567020}, 
                       'AgeCompte (j)': {"mean":-0.002555 , "std":0.584303}, 
                       'AgeClient':{"mean":4.693210, "std":3.337875}, 
                       'MontantPret': {"mean":0.502908, "std":0.572779}, 
                       'TauxInteret': {"mean": 0.495658, "std":0.499983}}
              
               # with st.expander('Input features'):
               st.write('**Input data before scalling**')
               input_df = df.copy()
               input_df
               
              
              
               # scaler = StandardScaler()
               data = np.array(input_df)
               # input_df[columns] = scaler.fit_transform(input_df[columns])
              
               def manual_standardize(df, columns):
                   for col, stats in columns.items():
                       mean = stats["mean"]
                       std = stats["std"]
                       # Standardiser la colonne en utilisant la formule
                       df[col] = (data - mean) / std
                   return df
              
               manual_standardize(input_df, columns=columns)
              
               # input_penguins = pd.concat([input_df, X_test], axis=0)
               st.write('**Input data after scalling**')
               input_df
              
               testing_model_by_ilocation(data=input_df)


# --------------------------------
# 14- Churn prediction by uploading csv file
import os

def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path.name)
    return f"{extension}"

# Visualization
def make_visualization(data):
    making_scaler_standardscaler(data)
    p = figure(title="Montant de transaction par ville", x_axis_label="Ville", y_axis_label="Montant Transaction")

    X_data = data["Ville"]
    Y_data = data["MontantTrans"]
    p.line(X_data, Y_data, legend_label="Trend", line_width=2)
    chart_data = data["MontantTrans"]
    # st.bokeh_chart(p, use_container_width=True)
    st.bar_chart(chart_data)

       
# Function: churn prediction 
def churn_prediction_by_uploading_file(df_uploaded):
       # columns to be scaling
       columns_to_scaled = ['MontantTrans', 'ScoreCSAT', 'ScoreNPS', 'AgeCompte (j)', 'AgeClient', 'MontantPret', 'TauxInteret']
       # columns to be encoded
       columns_to_encoded = ['TypeCompte', 'TypeTransaction', 'Ville', 'TypeEngagement']
       # all columns
       columns = columns_to_scaled + columns_to_encoded
       # Virify whether all columns in column_list are present in the dataset
       missing_columns = [col for col in columns if col not in df_uploaded.columns]
       if missing_columns:
              st.error(f"The following columns are missing in the dataset: {missing_columns}")
       else:
              st.write(df_uploaded.head(3))
              # shape of date
              st.write(f"Data size: {df_uploaded.shape[0]}")
              # accept the prediction whether the dataset'size is more than 1
              if df_uploaded.shape[0] > 1 :
                     # Remove columns not in the list
                     df_churn = df_uploaded[[col for col in columns if col in df_uploaded.columns]]
                     # Encoding the data using labelencoder
                     df_churn = make_encoding_labelencoder(df_churn, columns_to_encoded)
                     # Scaling the data using standardscaler
                     df_churn = making_scaler_standardscaler(df_churn)
                     df_churn = df_churn[100:151]
                     # make prediction
                     try:
                            prediction_results = []
                            for i in range(0, int(df_churn.shape[0])):
                                   new_data = df_churn.iloc[i]
                                   prediction = churn_model.predict([new_data])
                                   prediction_results.append(prediction)
                                   
                            df_pred = pd.DataFrame({'Prediction': np.array(prediction_results).flatten()})
                            # Create a new column 'Classification' based on the 'Prediction' column
                            df_pred['Classification'] = df_pred['Prediction'].map({1: 'Churner', 0: 'Loyal'})
                            # Count the occurrences of each classification
                            classification_counts = df_pred['Classification'].value_counts()
                            classification_counts = pd.DataFrame(classification_counts)
                            st.write(classification_counts["count"])
                            st.bar_chart(classification_counts)
                     
                     except ValueError as e:
                            st.error(f"Prediction error: {e}")
                     except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
    

with st.expander("CHURN PREDICTION - BY UPLOADIN FILE"):
       st.write("#### File uploader")
       df_uploaded = st.file_uploader(label="Upload the dataset here.")
       if df_uploaded:
               if get_file_extension(df_uploaded) == ".csv":
                     df_uploaded = pd.read_csv(df_uploaded)
                     churn_prediction_by_uploading_file(df_uploaded)
               elif get_file_extension(df_uploaded) == ".xlsx":
                     df_uploaded = pd.read_excel(df_uploaded)
                     churn_prediction_by_uploading_file(df_uploaded)
                     
               else:
                     st.error("#### Make sure you had uploaded csv or excel file")




   
