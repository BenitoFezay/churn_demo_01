# -------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------
# 1- ENOCDING "Churn"
label_encoder = LabelEncoder()
# df_churn["Churn"] = label_encoder.fit_transform(df_churn[["Churn"]])


# -------------------------------
# 6- Scalling data with StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns = ['MontantTrans', 'ScoreCSAT',
       'ScoreNPS', 'AgeCompte (j)', 'AgeClient',
       'MontantPret', 'TauxInteret'
       ]

def making_scaler_standardscaler(df):
    df[columns] = scaler.fit_transform(df[columns])

# --------------------------------
# 7- SPLITTING THE DATA
# 7.1- RESAMPLING 
from imblearn.over_sampling import SMOTE
smote = SMOTE()

# 7.2- Using train and test split
from sklearn.model_selection import train_test_split

# -------------------------------
# 8- CREATE MODEL
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 10- ACCURACY
import sklearn.metrics as sm

# ------------------------------
# 11- CONFUSION-MATRIX
from sklearn.metrics import confusion_matrix as cm

# ------------------------------
# 12- CLASSIFICATION_REPORT
from sklearn.metrics import classification_report as cr

# ------------------------------
# 13- CROSS VALIDATION
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ------------------------------
# 14- BOOSTING MODEL
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# -------------------------------
# 17- SAVING MODEL
import pickle

model_path = './model/churn_model.pkl'
# pickle.dump(model_xgb, open(model_path, 'wb'))

# To load the model later:
# churn_model = joblib.load('./model/churn_model.pkl')
churn_model = pickle.load(open(model_path, 'rb'))


# -------------------------------
# 18- TESTING THE MODEL
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
# 19- STREMLIT APP
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

st.title("🤖 Machine Learning")

# --------------------------------
# 21- Churn prediction BY FILL FIELDS
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



with st.expander("CHURN PREDICTION - BY FILLING FIELDS"):
    st.write("#### Input Data")
    with st.form(key="Churn_form"):
        
        col1, col2, col3 = st.columns(3)
        # home_value = col1.number_input("Data location", min_value=0, max_value=len(X_test), value=0)
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
# 22- Churn prediction by uploading csv file
import os

def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path.name)
    return f"{extension}"

# Visualization
# from bokeh.plotting import figure
def make_visualization(data):
    making_scaler_standardscaler(data)
    p = figure(title="Montant de transaction par ville", x_axis_label="Ville", y_axis_label="Montant Transaction")

    X_data = data["Ville"]
    Y_data = data["MontantTrans"]
    p.line(X_data, Y_data, legend_label="Trend", line_width=2)
    chart_data = data["MontantTrans"]
    # st.bokeh_chart(p, use_container_width=True)
    st.bar_chart(chart_data)

with st.expander("CHURN PREDICTION - BY UPLOADIN FILE"):
    st.write("#### File uploader")
    df_uploaded = st.file_uploader(label="Upload the dataset here.")
    if df_uploaded:
        if get_file_extension(df_uploaded) == ".csv":
            df_uploaded = pd.read_csv(df_uploaded)
            df_uploaded
            make_visualization(df_uploaded)

        elif get_file_extension(df_uploaded) == ".xlsx":
            df_uploaded = pd.read_excel(df_uploaded)
            df_uploaded
            make_visualization(df_uploaded)
        else:
            st.error("#### Make sure that you had uploaded csv or excel file")



# ------------------------------
# **- Display prediction
   
