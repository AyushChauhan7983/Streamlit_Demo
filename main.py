import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("data//Salary_Data.csv")
x = np.array(data["YearsExperience"]).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data["Salary"]))


st.title("Salary predictor")
st.image("data//sal.jpg", width=800)

nav = st.sidebar.radio("",["Home","Prediction","Contribute"])
if nav=="Home":
    if st.checkbox("Show Table"):
        st.table(data)
    graph = st.selectbox("What kind of graph?",["Non-interactive","Interactive"])
    val = st.slider("Filter data using years",0,20)
    data = data.loc[data["YearsExperience"]>=val]

    if graph=="Non-interactive":
        fig = plt.figure(figsize=(10,5))
        plt.scatter(data["YearsExperience"],data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot(fig)
    if graph=="Interactive":
        layout = go.Layout(
            xaxis = dict(range=[0,16]),
            yaxis = dict(range=[0,210000])
        )
        fig = go.Figure(data=go.Bar(x=data["YearsExperience"],y=data["Salary"]),layout=layout)
        st.plotly_chart(fig)

if nav=="Prediction":
    st.header("Know your salary")
    val = st.number_input("Enter your experience",0.00,20.00,step=0.25)
    val = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f'Your predicted salary is {round(pred)}')

if nav=="Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience", 0.0, 20.0)
    sal = st.number_input("Enter your Salary", 0.00, 1000000.00, step=1000.0)
    if st.button("submit"):
        to_add = {"YearsExperience": [ex], "Salary": [sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//Salary_Data.csv", mode='a', header=False, index=False)
        st.success("Submitted")
