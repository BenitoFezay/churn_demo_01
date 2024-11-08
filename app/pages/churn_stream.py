# -------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



# ---------------------------------
# 1- INCORPORATE DATA 
df_churn = pd.read_csv('./data/data_churn.csv')

# ---------------------------------
# 2- CREATE CHURN COLUMN
df_churn["Churn"] = df_churn["StatusCompte"].map({0:"Yes", 1: "No"})

# ---------------------------------
# 3- ENOCDING "Churn"
label_encoder = LabelEncoder()
df_churn["Churn"] = label_encoder.fit_transform(df_churn[["Churn"]])

# ---------------------------------
# 4- REMOVING THE COLUMN "StatutCompte"
df_churn.drop("StatusCompte", axis=1, inplace=True)

print(df_churn.columns)

# ---------------------------------
# 5- Fillna to 0 for some variables ["MontantPret", "TauxInteret"]
df_churn["MontantPret"].fillna(0, inplace=True)
df_churn["TauxInteret"].fillna(0, inplace=True)

# -------------------------------
# 6- Scalling data with StandardScaler
from sklearn.preprocessing import StandardScaler


# df_churn = df_churn.copy()
sacler = StandardScaler()
columns = ['MontantTrans', 'ScoreCSAT',
       'ScoreNPS', 'AgeCompte (j)', 'AgeClient',
       'MontantPret', 'TauxInteret'
       ]
df_churn[columns] = sacler.fit_transform(df_churn[columns])


# --------------------------------
# 7- SPLITTING THE DATA
X = df_churn.drop("Churn", axis=1)
y = df_churn["Churn"]

# 7.1- RESAMPLING 
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 7.2- Using train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# -------------------------------
# 8- CREATE MODEL
from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train, y_train)


# -------------------------------
# 9- MAKE PREDICTION
# pred = model.predict(X_test)
# print("prediction: ", pred)


# -------------------------------
# 10- ACCURACY
import sklearn.metrics as sm
# accuracy = sm.accuracy_score(y_test, pred) * 100
# print("accuracy: ", accuracy) 

# ------------------------------
# 11- CONFUSION-MATRIX
from sklearn.metrics import confusion_matrix as cm
# conf_m = cm(y_test, pred)
# print("confusion_matrix: ", conf_m)


# ------------------------------
# 12- CLASSIFICATION_REPORT
from sklearn.metrics import classification_report as cr
# rerpors = cr(y_test, pred)
# print("\nClassification-report: \n", rerpors);



# ------------------------------
# 13- CROSS VALIDATION
from sklearn.model_selection import cross_val_score, StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# kfscore = cross_val_score(LogisticRegression(), X_train, y_train, cv=skf)
# print("kfscore: ", np.average(kfscore))



# ------------------------------***********************************
# 14- BOOSTING MODEL
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model_xgb = XGBClassifier(max_depth=7, subsample=0.9, n_estimators=140)
model_xgb.fit(X_train, y_train)
# y_predict = model_xgb.predict(X_test)
# y_train_pred = model_xgb.predict(X_train)
# print("Boosting_Test: ", accuracy_score(y_test, y_predict))
# print("Boosting_Train: ", accuracy_score(y_train, y_train_pred))
# print("\nBoosting_classification_report: \n", cr(y_test, y_predict))

# ------------------------------
# 15- GRID SEARCH
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [50, 70, 100],
#     "max_depth": [5, 7, 8],
#     "subsample": [0.5, 0.7, 0.9],
#     "learning_rate": [0.1, 0.01, 0.001],
# }
# grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring="accuracy", verbose=4, refit="f1")
# grid_search.fit(X_train, y_train)


# # ------------------------------
# # 16- BEST MODEL
# best_model = grid_search.best_estimator_
# y_predict = best_model.predict(X_test)
# print("\nGreadSearch_Classification_report: \n", cr(y_test, y_predict))


# -------------------------------
# 17- SAVING MODEL
import pickle
import joblib

model_path = './model/churn_model.pkl'
pickle.dump(model_xgb, open(model_path, 'wb'))

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
        return st.info("This Client is a :red[churner]")
    else:
        # st.write(new_data)
        return st.success("This Client is a loyal")

# y_predict_loaded = churn_model.predict(X_test)
# accuracy_loaded = accuracy_score(y_test, y_predict_loaded)
# print(f"Accuracy of the loaded model: {accuracy_loaded}") 

# ------------------------------
# 19- STREMLIT APP
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

st.title("🤖 Machine Learning")

# -------------------------------
# 20- Data Visualization
# with st.expander("VISUALIZATION"):
#     st.write("#### Churn visualization")
#     st.bar_chart(df_churn, x="AgeCompte (j)", y="Churn")



# --------------------------------
# 21- Churn prediction
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



with st.expander("CHURN PREDICTION"):
    st.write("#### Input Data")
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
# 22- Side bar
# with st.sidebar:
#     st.header("Input Features")
#     type_account = st.selectbox('Type de Compte', ('Compte Courant', 'Compte Épargne'))
#     col1, col2 = st.columns(2)
#     transaction_amount = col1.number_input("Montant Transaction (Ariary)", min_value=2_000, value=500_000) #st.slider('Montant de Transaction (Ariary)', 2_000, 800_000_000, 100_000) 
#     age_client = st.slider('Age client', 18, 100, 47) 


# st.write("### Repayments")
# col1, col2, col3 = st.columns(3)
# col1.metric(label="Monthly Repayments", value=f"${monthly_payment:,.2f}")
# col2.metric(label="Total Repayments", value=f"${total_payments:,.0f}")
# col3.metric(label="Total Interest", value=f"${total_interest:,.0f}")



# Display the data-frame as a chart.
# st.write("### Payment Schedule")
# payments_df = df[["Year", "Remaining Balance"]].groupby("Year").min()
# st.line_chart(payments_df)


# ------------------------------
# **- Display prediction
   
