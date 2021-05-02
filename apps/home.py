import streamlit as st

def app():
    st.title('Introduction')

    st.write('Credit risk analysis is quite an important process in the financial sector. Banks and lending agencies make a lot of money through credit. However, it is highly risky. Hence, the need for Credit Risk Analysis exists.')

    st.write('When it comes to large loans, like home loans, car loans etc. banks often get some form of security against the credit. So if the client defaults on his/her payments, the bank will have something to fall back on to avoid losses. This is not really the case with personal loans though. A lot of people look for personal loans for various reasons. These are usually small amounts. Risk analysis becomes crucial here as these loans are sanctioned mostly based on credit scores. Predicting whether or not a client will default cannot be done with credit scores. A person defaulting on his/her loans depends on various factors. This pushed us to learn about how credit risk analysis is performed and how defaults are predicted')

    st.write('One of our findings from our preliminary research was that the algorithm used industry wide to predict defaults was Logistic Regression. There are cases where this algorithm performs really well and vice versa. This inspired us to perform a comparative study of logistic regression with our algorithm of choice, LightGBM. It is an algorithm/framework developed very recently by Microsoft. We have also compared XGboost and Catboost along with these two algorithms.')

    st.write('We will be showcasing our project with this web app.')
