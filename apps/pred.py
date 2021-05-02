import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

def app():
    st.write("""
    This app is a tool that can predict a person's loan repayment capability using LightGBM, one of the most advanced gradient boosting algorithms
    around.
    Data obtained from the UCI Machine Learning Repository.
    """)

    default = pd.read_csv('UCI_Credit_Card.csv', index_col="ID")
    default.rename(columns=lambda x: x.lower(), inplace=True)

    default['grad_school'] = (default['education'] == 1).astype('int')
    default['university'] = (default['education'] == 2).astype('int')
    default['high_school'] = (default['education'] == 3).astype('int')
    default.drop('education', axis=1, inplace=True)

    default['male'] = (default['sex']==1).astype('int')
    default.drop('sex', axis=1, inplace=True)

    default['married'] = (default['marriage'] == 1).astype('int')
    default.drop('marriage', axis=1, inplace=True)

    pay_features = ['pay_0','pay_2','pay_3','pay_4','pay_5','pay_6']
    for p in pay_features:
        default.loc[default[p]<=0, p] = 0

    default.rename(columns={'default payment next month':'default'}, inplace=True) 

    target_name = 'default'
    X = default.drop('default', axis=1)
    robust_scaler = RobustScaler()
    X = robust_scaler.fit_transform(X)
    y = default[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)

    import lightgbm as lgb
    d_train = lgb.Dataset(X_train, label=y_train)

    lgbm_params = {'learning_rate':0.039, 'boosting_type':'gbdt',    
              'objective':'binary',
              'metric':['auc', 'binary_logloss'],
              'num_leaves':150,
              'max_depth':18}

    clf = lgb.train(lgbm_params, d_train, 100) 

    y_pred_lgbm=clf.predict(X_test)


    st.sidebar.header('User Input Features')


    limit_bal=st.sidebar.text_input('Credit Amount',500)
    age=st.sidebar.text_input('Age',25)
    bill_amt1=st.sidebar.text_input('Previous Bill Amount 1',100)
    bill_amt2=st.sidebar.text_input('Previous Bill Amount 2',0)
    bill_amt3=st.sidebar.text_input('Previous Bill Amount 3',0)
    bill_amt4=st.sidebar.text_input('Previous Bill Amount 4',0)
    bill_amt5=st.sidebar.text_input('Previous Bill Amount 5',0)
    bill_amt6=st.sidebar.text_input('Previous Bill Amount 6',100)
    pay_amt1=st.sidebar.text_input('Paid Amount 1',100)
    pay_amt2=st.sidebar.text_input('Paid Amount 2',0)
    pay_amt3=st.sidebar.text_input('Paid Amount 3',0)
    pay_amt4=st.sidebar.text_input('Paid Amount 4',0)
    pay_amt5=st.sidebar.text_input('Paid Amount 5',0)
    pay_amt6=st.sidebar.text_input('Paid Amount 6',100)
    sex=st.sidebar.selectbox('Sex',(1,0))
    grad_school=st.sidebar.selectbox('Grad School',(0,1))
    university=st.sidebar.selectbox('University',(0,1))
    high_school=st.sidebar.selectbox('High School',(0,1))
    married=st.sidebar.selectbox('Marital Status',(1,2,3))
    pay_0=st.sidebar.selectbox('Payment Status 1',(-1,1,2,3,4,5,6,7,8,9))
    pay_2=st.sidebar.selectbox('Payment Status 2',(-1,1,2,3,4,5,6,7,8,9))
    pay_3=st.sidebar.selectbox('Payment Status 3',(-1,1,2,3,4,5,6,7,8,9))
    pay_4=st.sidebar.selectbox('Payment Status 4',(-1,1,2,3,4,5,6,7,8,9))
    pay_5=st.sidebar.selectbox('Payment Status 5',(-1,1,2,3,4,5,6,7,8,9))
    pay_6=st.sidebar.selectbox('Payment Status 6',(-1,1,2,3,4,5,6,7,8,9))
    
    from collections import OrderedDict
    new_customer = OrderedDict([('limit_bal', limit_bal),('age', age ),('bill_amt1', bill_amt1),
                            ('bill_amt2', bill_amt2 ),('bill_amt3',bill_amt3 ),('bill_amt4', bill_amt4 ),
                            ('bill_amt5', bill_amt5 ),('bill_amt6', bill_amt6 ), ('pay_amt1', pay_amt1 ),('pay_amt2', pay_amt2 ),
                            ('pay_amt3', 0 ),('pay_amt4', 0 ),('pay_amt5', 0 ), ('pay_amt6', 0 ),
                            ('sex', sex ),('grad_school', grad_school ),('university', university ), ('high_school', high_school ),
                            ('married', married ),('pay_0', pay_0 ),('pay_2', pay_2 ),('pay_3', pay_3 ),
                            ('pay_4', pay_4),('pay_5', pay_5), ('pay_6', pay_6)])
    
    new_customer = pd.Series(new_customer)
    data = new_customer.values.reshape(1, -1)
    data = robust_scaler.transform(data)
    st.header('Prediction from LightGBM Model')


    prob = clf.predict(data)[0]
    if (prob >= 0.5):
        st.write('Will default')
    else:
        st.write('Will pay')
