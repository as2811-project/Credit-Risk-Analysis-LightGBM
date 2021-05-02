import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def app():
    st.title('Exploratory Data Analysis - UCI Credit Dataset')
    df = pd.read_csv("UCI_Credit_Card.csv")
    pr = ProfileReport(df, explorative=True)
    st.header('**Dataset**')
    st.write(df)
    st.write('---')
    st.header('**Analysis Report**')
    st_profile_report(pr)
