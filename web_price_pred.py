import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

data = pd.read_csv("https://raw.githubusercontent.com/jinu043/vehicleprice_prediction/main/ml_cl_dubicars.csv")

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)

st.write("""
## PREDICTION OF VEHICLE PRICE
""")

st.sidebar.header("INPUT VEHICLE DETAILS")

def user_input_featutres():
    vehicle_name = st.sidebar.selectbox("Vehicle Name",options=data["vehicle_name"].unique().tolist(),)
    model = st.sidebar.selectbox("Model", options=data[data["vehicle_name"]==vehicle_name]["model"].unique().tolist())
    vehicle_type = st.sidebar.selectbox("Vehicle Type", options=data[data["model"]==model]["vehicle_type"].unique().tolist())
    model_year = st.sidebar.selectbox("Vehicle Type", options=data[data["vehicle_type"]==vehicle_type]["model_year"].unique().tolist())
    fuel_type = st.sidebar.selectbox("Fuel Type", options=data[(data["vehicle_type"]==vehicle_type)&(data["model_year"]==model_year)]["fuel_type"].unique().tolist())
    spec = st.sidebar.selectbox("Spec", options=data[(data["vehicle_type"]==vehicle_type)&(data["model_year"]==model_year)&(data["fuel_type"]==fuel_type)]["spec"].unique().tolist())
    mileage = st.sidebar.selectbox("Mileage", options=range(0,101))
    input_data = {
        "vehicle_name":vehicle_name,
        "model":model,
        "vehicle_type":vehicle_type,
        "model_year":model_year,
        "fuel_type":fuel_type,
        "spec":spec,
        "mileage":mileage
    }
    features = pd.DataFrame(input_data, index=[0])
    return features
df = user_input_featutres()
st.subheader("Vehicle Input Features")
st.write(df)

data["model_year"] = data["model_year"].astype("category")
cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

encoder = OneHotEncoder(sparse=False)
encoder.fit(data[cat_cols])
enc_cols = list(encoder.get_feature_names_out(cat_cols))

data1 = data.copy()

min_max_sc = MinMaxScaler()
min_max_sc.fit(np.array(data1["mileage"]).reshape(-1,1))
data1["mileage"] = min_max_sc.transform(np.array(data1["mileage"]).reshape(-1,1))

data1[enc_cols] = encoder.transform(data1[cat_cols])

num_cols = data.select_dtypes(include="int").columns.tolist()

train_ = data1[enc_cols+num_cols]
inp_cols = train_.columns.tolist()[:-1]
target_col = "price"
inputs = train_[inp_cols]
targets = train_[target_col]

xgb = XGBRegressor(n_jobs=-1, n_estimators=800, max_depth=30,
                     random_state=10, learning_rate=0.2, subsample=0.8)

xgb.fit(inputs, targets)

def single_input_prediction(df):
    input_sample = pd.DataFrame(df, index=[0])
    input_sample["mileage"] = min_max_sc.transform(np.array(input_sample["mileage"]).reshape(-1,1))
    input_sample[enc_cols] = encoder.transform(input_sample[cat_cols])
    inputs = input_sample[enc_cols+["mileage"]]
    price_predicted = xgb.predict(inputs)
    return np.ceil(price_predicted[0])
st.subheader("Approximate Price of Vehicle")

st.markdown('<p style=“font-size:300px;color:red”>' + "AED " +str(single_input_prediction(df)) + '</p>', unsafe_allow_html=True)


