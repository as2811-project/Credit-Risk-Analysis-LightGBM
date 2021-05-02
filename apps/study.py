import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go

def app():
    st.markdown('''Comparative Study Results''')
    st.write('This is how the algorithms compare:')
    study = pd.read_csv('apps/comparisondata.csv')
    algorithms=['LightGBM', 'XGBoost', 'Logistic Regression','CatBoost']
    fig = go.Figure(data=[
    go.Bar(name='Accuracy', x=algorithms, y=[85.5556,82.4222,81.9778,82.066667]),
    go.Bar(name='Precision', x=algorithms, y=[78.798, 70.4, 69.8276, 65.766739]),
    go.Bar(name='Recall', x=algorithms, y=[47.4372, 35.3769, 32.5628, 37.202199])
    ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)
    st.table(study)
    st.write('We conclude that LightGBM outperforms the other three algorithms and is hence, better suited for this use case. The execution time is faster than that of both XGBoost and Catboost. It is slightly slower than Logistic Regression however it makes up for it through higher accuracy, precision and recall. Precision and recall are the two most important factors in determining how good an algorithm is for predicting defaults. This is because in this particular project, false positives mean people who paid being classed as defaults and false negatives mean the opposite. Having a higher number of false negatives is bad, this can be avoided if the algorithm has better recall.')
    st.write('Here are some diagrams: ')
    
    image = Image.open('images/Heatmap.png')
    st.image(image, caption='heatmap')
    st.write('''The heatmap you see above depicts the correlation between all the categorical features available in the dataset. There are 24 features in total.''')
    st.write('● Limit_bal is the feature that depicts the amount given as credit to the client.')
    st.write('''● Sex is the next feature. The values under this category are 1 and 2, 1 being male and 2 being
    female.''')
    st.write('''● Education has 4 sub-categories, namely, highschool, college/university, graduate school and
    others. This category depicts the educational qualification of an individual client.''')
    st.write('''● Marriage depicts the marital status of the clients. There are three subcategories here, namely,
    married, divorced and others.''')
    st.write('''● Pay_0 - Pay_5 depicts the loan repayment status of the client starting from April to August
    (in reverse chronological order)''')
    st.write('''● Bill_amt1 - Bill_amt6 depicts the bill statement amount in reverse chronological order
    across the 6 month period.''')
    st.write('''● Pay_amt1 - Pay_amt6 depicts the previous months repayment status. If the value is -1, it
    means the client had paid duly, if it is 1 it means the client delayed payment by one month
    and so on.''')
    st.write('''● Default is the last category. It shows whether that particular client defaulted in repaying their
    loan. 1 means yes, 0 means no.''')
    st.write('''There are quite a few insights that can be gained from the heatmap we see above. There is an interesting negative correlation between limit_bal and default. This means that the higher the credit limit is, the lower the default rate is. This is interesting because people tend to fail on loan repayments when the amount is large. The “default” feature correlates the most with the first payment feature pay_0, it means that people who delay their very first payment tend to default. This is logically understandable considering these clients will then have to be repaying a larger amount every subsequent month along with interest, making it incredibly difficult to pay off completely. We can observe that the entire region from pay_0 or pay_1 to pay_6 is darker for defaults. This suggests that payment behaviour is a strong indication of a client’s ability to repay their loan. As understood from the previous point, regular payments can help a client be on the safer side.''')

    image1 = Image.open('images/roclgbm.png')
    st.image(image1, caption='AUC Curve - LightGBM')
    st.write('''From out testing and experimentation, LightGBM came out on top. An AUC score of 0.90 is excellent. This is considerably higher than both XGBoost and Logistic Regression.''')
    
    image2 = Image.open('images/rocxgb.png')
    st.image(image2, caption='AUC Curve - XGBoost')
    image3 = Image.open('images/roclr.png')
    st.image(image3, caption='AUC Curve - Logistic Regression')

    