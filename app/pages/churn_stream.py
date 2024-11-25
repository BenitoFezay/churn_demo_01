# -------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from datetime import date

# SKLEARN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# STREMLIT APP
import streamlit as st
import matplotlib.pyplot as plt

# PICKEL
import pickle



# ------------------------------------- 
#         V A R I A B L E S
# -------------------------------------
# Scalling input data
columns_params = {'MontantTrans': {"mean":2.298365e+06 , "std": 1.502631e+06}, 
         'ScoreCSAT': {"mean":5.381457 ,"std": 2.834845}, 
         'ScoreNPS': {"mean":5.437009, "std":2.795327}, 
         'AgeCompte (j)': {"mean":1342.523675, "std":1051.974388}, 
         'AgeClient':{"mean":53.844637, "std":20.453082}, 
         'MontantPret': {"mean":5.792726e+10, "std":7.356084e+10}, 
         'TauxInteret': {"mean": 1.475264, "std":1.681651}}

# -------------------------------------
# filtered columns
filtered_columns = ['TypeCompte', 'MontantTrans', 'TypeTransaction', 'ScoreCSAT', 'ScoreNPS', 'AgeCompte (j)', 'AgeClient', 'Ville', 'MontantPret', 'TauxInteret', 'TypeEngagement']
# columns to be scaling
columns_to_scaled = ['MontantTrans', 'ScoreCSAT', 'ScoreNPS', 'AgeCompte (j)', 'AgeClient', 'MontantPret', 'TauxInteret']
# columns to be encoded
columns_to_encoded = ['TypeCompte', 'TypeTransaction', 'Ville', 'TypeEngagement']
# all columns
all_columns = columns_to_scaled + columns_to_encoded
# dataset exapmle
data_exampler = pd.read_csv('./data/df_test.csv')


# ------------------------------------- 
#          II- F U N C T I O N S
# -------------------------------------
# 1.1- Encoding the data using labelencoder
def make_encoding_labelencoder(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

# -------------------------------
# 1.2- Scalling data with StandardScaler
columns = ['MontantTrans', 'ScoreCSAT', 'ScoreNPS', 'AgeCompte (j)', 'AgeClient', 'MontantPret', 'TauxInteret']

# -------------------------------
# 1.3- TESTING THE MODEL
def prediction_one_line_data(data):
    prediction = churn_model.predict(data)

    if f"{prediction}" == "[1]":
        return "Churner"
    else:
        return "Loyal"

# -----------------------------
# 1.4- Manual standardscaler
def manual_standardscaler(df, columns):
    # verify the size of data
    if df.shape[0] == 1: 
        for col, stats in columns.items():
            mean = stats["mean"]
            std = stats["std"]
            try:
                # Standardiser la colonne en utilisant la formule
                df[col] = (np.array(df[col].values)[0] - mean) / std
            except:
                df[col] = (df[col].values - mean) / std
        return df

    elif df.shape[0] > 1:
        df_0 = df.loc[0]
        for col, stats in columns.items():
            mean = stats["mean"]
            std = stats["std"]
            # Standardiser la colonne en utilisant la formule
            df_0[col] = (df.loc[0, col] - mean) / std
        df_0 = pd.DataFrame(df_0).transpose()

        for i in range(1, df.shape[0]):
            df_1 = df.loc[i]
            for col, stats in columns.items():
                mean = stats["mean"]
                std = stats["std"]
                # Standardiser la colonne en utilisant la formule
                df_1[col] = (df.loc[i, col] - mean) / std
            df_1 = pd.DataFrame(df_1).transpose()
            df_0 = pd.concat([df_0, df_1])
        return df_0

# --------------------------------
# 1.5- Manual encoding data
def make_data_encoded(data, is_one_line=True):
    if is_one_line:
        data = pd.DataFrame(data, index=[0])
    # Type account ------------------------
    data["TypeCompte"] = data["TypeCompte"].map({"Compte Courant": 0, "Compte Épargne": 1})  

    # Type engagement ---------------------
    data["TypeEngagement"] = data["TypeEngagement"].map({"Utilisation d'application mobile": 0, "Participation Programme Fidélité": 1})

    # Type transaction --------------------
    data["TypeTransaction"] = data["TypeTransaction"].map({'Depot': 0,'Paiement': 1, 'Retrait': 2, 'Virement': 3})

    # Ville --------------------------------
    data["Ville"] = data["Ville"].map({'Antsohihy':0, 'Ambanja':1, 'Antananarivo':2, 'Ihosy':3, 'Toliara':4,
       'Antsiranana':5, 'Fianarantsoa':6, 'Toamasina':7, 'Fort Dauphin':8,
       'Mahajanga':9, 'Andapa':10, 'Ambovombe':11, 'Morondava':12, 'Nosy Be':13})
    
    return data

# --------------------------------
# 1.6- Churn prediction by uploading csv file
def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path.name)
    return f"{extension}"

# --------------------------------
# 1.7- Making churn prediction
def prediction_multi_lines(df_churn):
    try:
        prediction_results = churn_model.predict(df_churn[filtered_columns])
        df_pred = pd.DataFrame({'Prediction': np.array(prediction_results).flatten()})
        # Create a new column 'Classification' based on the 'Prediction' column
        df_pred['Classification'] = df_pred['Prediction'].map({1: 'Churner', 0: 'Loyal'})
        # Count the occurrences of each classification
        classification_counts = df_pred['Classification'].value_counts()
        # classification_counts = df_pred['Prediction'].value_counts()
        classification_counts = pd.DataFrame(classification_counts)

        # ------------------------
        # Result
        col1, col2 = st.columns(2)
        with col1:
            df_pred
            # st.write(classification_counts["count"])
        with col2:
            st.bar_chart(classification_counts)
    
    except ValueError as e:
        st.error(f"Prediction error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --------------------------------------------
# 1.8- Preparing data to churn prediction 
def prepare_data_and_predict(df_uploaded):
    # Virify whether all columns in column_list are present in the dataset
    missing_columns = [col for col in all_columns if col not in df_uploaded.columns]
    if missing_columns:
            st.error(f"The following columns are missing in the dataset: {missing_columns}")
    else:
        st.write(df_uploaded.head(3))
        # shape of date
        st.write(f"**Data size**: {df_uploaded.shape[0]}")
        st.header("Result of Churn prediction", divider=True)
        # accept the prediction whether the dataset'size is more than 1
        if df_uploaded.shape[0] > 0:
            # Remove columns not in the list
            df_churn = df_uploaded[[col for col in all_columns if col in df_uploaded.columns]]
            if df_churn.shape[0] == 1:
                # encoding the data manualy
                try:
                    df_churn = make_data_encoded(data=df_churn)
                except:
                    pass
                # scaling the data using standardscaler
                df_churn = manual_standardscaler(df_churn, columns=columns_params)
                # making prediction
                result_prediction = prediction_one_line_data(df_churn[filtered_columns])
                if result_prediction == "Churner":
                    st.error("This Client is churner")
                else:
                    st.success("This Client is loyal")
                    
            else:
                try:
                    # manual encoding fata
                    df_churn = make_data_encoded(data=df_churn, is_one_line=False)
                    # making standard scaler
                    df_churn = manual_standardscaler(df=df_churn, columns=columns_params)
                    # making prediction
                    prediction_multi_lines(df_churn)
                except:
                    st.info("###### Can't support the this data.")
    

# ---------------------------------
#   II- M O D E L
# ---------------------------------
# 2.1- SAVING MODEL
model_path = './model/churn_model.pkl'

# 2.2- To load the model later:
churn_model = pickle.load(open(model_path, 'rb'))


# ---------------------------------
#   A P P L I C A T I O N
# ---------------------------------
st.title("🤖 Machine Learning")

# ------------------------------
#  PREDICTION BY FILLING FIELDS
# ------------------------------
with st.expander("CHURN PREDICTION - BY FILLING FIELDS"):
    st.write("#### Input Data")
    with st.form(key="Churn_form"):
        col1, col2, col3 = st.columns(3)
        # montant_transaction
        transaction_amount = col1.number_input("Montant Transaction (ar)", min_value=100, value=450_000)
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
        type_engagement = col3.selectbox("Type d'engagement", ("Utilisation d'application mobile", "Participation Programme Fidélité"))
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
        
        data = make_data_encoded(data=data)
        
        if submit_button:
            df = pd.DataFrame(data, index=[0])
            input_df = df.copy()
            # input_df
            input_df = manual_standardscaler(input_df, columns=columns_params)
            result_prediction = prediction_one_line_data(data=input_df)
            if result_prediction == "Churner":
                st.error("This Client is churner")
            else:
                st.success("This Client is loyal")
                

# ------------------------------
#  PREDICTION BY UPLOADIND FILE
# ------------------------------
with st.expander("CHURN PREDICTION - BY UPLOADIN FILE"):
    st.write("***Your data must content some columns(11) like this :***")
    st.dataframe(data_exampler)
    df_uploaded = st.file_uploader(label="Upload the dataset here")
    if df_uploaded:
        if get_file_extension(df_uploaded) == ".csv":
            df_uploaded = pd.read_csv(df_uploaded)
            prepare_data_and_predict(df_uploaded)
        elif get_file_extension(df_uploaded) == ".xlsx":
            df_uploaded = pd.read_excel(df_uploaded)
            prepare_data_and_predict(df_uploaded)
                
        else:
            st.error("#### Make sure you had uploaded csv or excel file")


# ------------------------------
#   R F M   FUNCTIONS
# ------------------------------
# Get date min from date ouverture column
def get_min__date_ouverture(df):
    df["DateOuverture"] = pd.to_datetime(df["DateOuverture"])
    min_date_ouverture = df.groupby('ClientID')['DateOuverture'].min()
    return pd.DataFrame(min_date_ouverture)

# ------------------------------
# Get date max from date transaction
def get_max_date_transaction(df):
    df["DateTransaction"] = pd.to_datetime(df["DateTransaction"])
    max_date_transactions = df.groupby('ClientID')['DateTransaction'].max()
    return pd.DataFrame(max_date_transactions)

# ------------------------------
# Builde data RMF
def prepare_rfm_data(df, min_date, max_date):
    max_date_transactions = max_date.merge(min_date, on='ClientID', how='left')

    rfm_df = max_date_transactions.copy()
    rfm_df.rename(columns={'max': 'Transaction', 'DateOuverture': 'Ouverture'}, inplace=True)

    return rfm_df

# ------------------------------
# Calculate Recency 
def calculate_recency(rfm_df, date_transaction="2024-10-01"):
    rfm_df['Transaction'] = pd.to_datetime(rfm_df['Transaction'])
    rfm_df['Ouverture'] = pd.to_datetime(rfm_df['Ouverture'])

    # Assuming last_date_transaction's date for calculating recency
    last_date_transaction = pd.to_datetime(date_transaction)

    # Calculate recency for each customer based on their most recent transaction
    rfm_df['Recency'] = np.abs((last_date_transaction - rfm_df['Transaction']).dt.days / 30)

    return rfm_df


# ------------------------------
# Calculate Frequency
def calculate_frequency(df, rfm_df, date_transaction="2024-10-01"):
    date_transaction = pd.to_datetime(date_transaction)
    # Calculate frequency for each customer (number of transactions)
    K_times = df.groupby('ClientID')['DateTransaction'].count()

    rfm_df["AgeCompte"] = (date_transaction - rfm_df['Ouverture']).dt.days / 30

    # Add frequency to the rfm_df DataFrame
    rfm_df = rfm_df.merge(K_times, on='ClientID', how='left')

    # Rename the frequency column
    rfm_df = rfm_df.rename(columns={'DateTransaction': 'K_times'})
    rfm_df["Frequency"] = rfm_df["K_times"] / rfm_df["AgeCompte"]

    rfm_df.drop(columns=['Transaction', 'Ouverture', "K_times", "AgeCompte"], axis=1, inplace=True)
    return rfm_df


# ------------------------------
#  SEGMENTATION CLIENT
# ------------------------------
# with st.expander("SEGMENTATION CLIENT"):
    # with st.form(key="segementation_client"):
        # upload file
        # segementation_data = st.file_uploader(label="Upload the dataset here", type=["csv"])
        # last date transaction
        # today = date.today()
        # last_date_transaction = st.date_input("Last date transaction", today, format="DD/MM/YYYY")
        # submit button
        # submit_button = st.form_submit_button(label="Sbumit")

    # Verify whether the data existe 
    # if segementation_data and submit_button:
        # Transforming data uploaded to data frame
        # segementation_data = pd.read_csv(segementation_data)        
        # df_client = pd.DataFrame(segementation_data)

        # scaler = StandardScaler()
        # rfm_df = calculate_frequency(df_client, last_date_transaction)
        # rfm_df
        # rfm_df[["Recency", "Frequency"]] = scaler.fit_transform(rfm_df[["Recency", "Frequency"]])

        # def optimization_of_k(data, max_k):
        #     mean_distortions = []
        #     inertias = []

        #     for k in range(1, max_k):
        #         kmeans = KMeans(n_clusters=k)
        #         kmeans.fit(data)

        #         mean_distortions.append(kmeans.inertia_)
        #         inertias.append(kmeans.inertia_)

        #     # make plot
        #     plt.plot(range(1, max_k), mean_distortions, marker='o')
        #     plt.xlabel('Number of clusters')
        #     plt.ylabel('Mean distortion')
        #     plt.grid(True)
        #     plt.show()

        # optimization_of_k(rfm_df[["Recency", "Frequency"]], 10)

        # kmeans = KMeans(n_clusters=4)
        # kmeans.fit(rfm_df[["Recency", "Frequency"]])
        # rfm_df['cluster_k'] = kmeans.labels_

        # plt.scatter(x=rfm_df['Recency'], y=rfm_df['Frequency'], c=rfm_df['cluster_k'])
        # plt.xlabel('Recency')
        # plt.ylabel('Frequency')
        # plt.title('K-means Clustering')
        # plt.show()

        # for k in range(2, 9):
        #     kmeans = KMeans(n_clusters=k)
        #     kmeans.fit(rfm_df[["Recency", "Frequency"]])
        #     rfm_df[f'cluster_{k}'] = kmeans.labels_

        # for k in range(2, 9):
        #     plt.scatter(x=rfm_df['Recency'], y=rfm_df['Frequency'], c=rfm_df[f'cluster_{k}'])
        #     plt.xlabel('Recency')
        #     plt.ylabel('Frequency')
        #     plt.title(f'K-means Clustering - {k}')
        #     plt.show()

        # for k in range(2, 9):
        #     silhouette_avg = silhouette_score(rfm_df[["Recency", "Frequency"]], rfm_df[f'cluster_{k}'])
        #     print(f"For n_clusters={k}, the silhouette_score is {silhouette_avg}")
