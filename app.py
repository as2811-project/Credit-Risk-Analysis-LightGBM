import streamlit as st
from multiapp import MultiApp
from apps import home, eda, pred, study # import your app modules here

app = MultiApp()

st.markdown("""
# Credit Risk Analysis Web App
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Exploratory Data Analysis", eda.app)
app.add_app("Individual Prediction", pred.app)
app.add_app("Comparative Study Results", study.app)

# The main app
app.run()
