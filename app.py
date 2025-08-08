import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time

# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2021/03/chaitali-majumder/house-price-497112-KhCJQICS.jpg')



st.header('Model of housing prices to predict median house values in California ',divider=True)



st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20230804/pngtree-an-upside-graph-showing-prices-and-houses-in-the-market-image_13000262.jpg')


# read_data
temp_df = pd.read_csv('california.csv')

random.seed(52)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)


ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values])


with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)


price = chatgpt.predict(final_value)[0]
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))
progress_bar=st.progress(0)
placeholder=st.empty()
placeholder.subheader('predicting price!!')
place=st.empty()
place.image('https://cdn.dribbble.com/userupload/20153655/file/original-d8d1785aab8bdcd7354c44167ccbc9b5.gif',width=100)

    

if price>0:

    for i in range (100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    # st.subheader(body)
    placeholder.empty()
    place.empty()

    st.success(body)
else:
    body='invalid house feature values'
    st.warning(body)











