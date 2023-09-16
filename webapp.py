import streamlit as st
import torch.nn as nn
import numpy as np
import torch

mean = 1.2050804798491299e-16
std = 1.0

def std_data(x):
    """
    Standardizing data
    """
    x = x-mean
    x = x/std
    return x

class ChurnModel(nn.Module):
    """
    Neural Network for churn prediction
    """
    def __init__(self,no_of_features):
        super(ChurnModel, self).__init__()
        self.linear1 = nn.Linear(no_of_features,no_of_features*2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(no_of_features*2,1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, targets_train ):
        out = self.linear1(targets_train)
        out  = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


model = torch.load("md_1.pth")
def predict_value(inputs):
    if len(inputs)==6:
        inputs = np.array(inputs)
        inputs = std_data(inputs)
        inputs = torch.from_numpy(inputs.astype(np.float32))
        pred = model(inputs)
        print(pred)
        value = pred.item()
        if value < 0.5:
            st.write(f"## Not Churned - {100-round(value*100,3)} %")
        else:
            st.write(f"## Churned :{100-round(value*100,3)} %")
    else:
        print(f"Length not matched")

features = ['Age','Gender','Location','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']
Age = st.number_input("Age",max_value=101,min_value=18)
Gender = st.radio(label="Gender",options={"Male":0,"Female":1})
Gender = 0 if Gender == "Male" else 1

display = ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston']

options = list(range(len(display)))

Location = st.selectbox("Location", options, format_func=lambda x: display[x])

Subscription_Length_Months = st.number_input("Subscription Lenght(Months)",min_value=0,max_value=90)
Monthly_Bill = st.number_input("Monthly Bill",min_value=1,max_value=10000)
Total_Usage_GB = st.number_input("Total Usage (GB)",min_value=1,max_value=10000)

input_ar = [Age,Gender,Location,Subscription_Length_Months,Monthly_Bill,Total_Usage_GB]

btn = st.button("Predict")

if btn:
    predict_value(input_ar)
