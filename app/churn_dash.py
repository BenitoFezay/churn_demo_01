import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px


# ---------------------------------
# 1- INCORPORATE DATA 
df_churn = pd.read_csv('../../../Data/Data_churn.csv')

# ---------------------------------
# 2- CREATE CHURN COLUMN
df_churn["Churn"] = df_churn["StatutCompte"].map({0:"Yes", 1: "No"})

# ---------------------------------
# 3- ENOCDING "Churn"
label_encoder = LabelEncoder()
df_churn["Churn"] = label_encoder.fit_transform(df_churn[["Churn"]])

# ---------------------------------
# 4- REMOVING THE COLUMN "StatutCompte"
df_churn.drop("StatutCompte", axis=1, inplace=True)

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
model = LogisticRegression()
model.fit(X_train, y_train)


# -------------------------------
# 9- MAKE PREDICTION
pred = model.predict(X_test)
print("prediction: ", pred)


# -------------------------------
# 10- ACCURACY
import sklearn.metrics as sm
accuracy = sm.accuracy_score(y_test, pred) * 100
print("accuracy: ", accuracy) 

# ------------------------------
# 11- CONFUSION-MATRIX
from sklearn.metrics import confusion_matrix as cm
conf_m = cm(y_test, pred)
print("confusion_matrix: ", conf_m)


# ------------------------------
# 12- CLASSIFICATION_REPORT
from sklearn.metrics import classification_report as cr
rerpors = cr(y_test, pred)
print("\nClassification-report: \n", rerpors);



# ------------------------------
# 13- CROSS VALIDATION
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfscore = cross_val_score(LogisticRegression(), X_train, y_train, cv=skf)
print("kfscore: ", np.average(kfscore))



# ------------------------------
# 14- BOOSTING MODEL
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model_xgb = XGBClassifier(max_depth=7, subsample=0.9, n_estimators=140)
model_xgb.fit(X_train, y_train)
y_predict = model_xgb.predict(X_test)
y_train_pred = model_xgb.predict(X_train)
print("Boosting_Test: ", accuracy_score(y_test, y_predict))
print("Boosting_Train: ", accuracy_score(y_train, y_train_pred))
print("\nBoosting_classification_report: \n", cr(y_test, y_predict))


# ------------------------------
# 15- GRID SEARCH
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 70, 100],
    "max_depth": [5, 7, 8],
    "subsample": [0.5, 0.7, 0.9],
    "learning_rate": [0.1, 0.01, 0.001],
}
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring="accuracy", verbose=4, refit="f1")
grid_search.fit(X_train, y_train)


# ------------------------------
# 16- BEST MODEL
best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_test)
print("\nGreadSearch_Classification_report: \n", cr(y_test, y_predict))


# Initialize the app
app = Dash(__name__)

# App layout
app.layout = [
    html.H3(children='CHURN PREDICTION', style={'textAlign':'center'}),
    html.Hr(),
    # dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='my-final-radio-item-example'),
    dash_table.DataTable(data=df_churn.to_dict('records'), page_size=12)
    # dcc.Graph(figure={}, id='my-final-graph-example')
]

# Add controls to build the interaction
# @callback(
#     Output(component_id='my-final-graph-example', component_property='figure'),
#     Input(component_id='my-final-radio-item-example', component_property='value')
# )
# def update_graph(col_chosen):
#     fig = px.histogram(df_churn, x='continent', y=col_chosen, histfunc='avg')
#     return fig


if __name__ == '__main__':
    app.run(debug=True)