#!/usr/bin/env python
# coding: utf-8

# In[16]:


#training code
import pandas as pd
import math
import pandas_datareader as dataread
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense,LSTM 
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = dataread.DataReader('AAPL', data_source='yahoo', start='2012-01-03', end='2020-12-17')

data=df.filter(['Close'])

dataset =data.values

training_data_len = math.ceil( len(dataset)*.8 )

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


train_data = scaled_data[0: training_data_len, :]

x_train = []

y_train = []

for i in range (60, len(train_data)):

    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape [1], 1))

model = Sequential()

model.add(LSTM(50, return_sequences=True,input_shape=(x_train.shape[1], 1))) 
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,
epochs = 100, batch_size=64,verbose=1)


# In[17]:


test_data = scaled_data[training_data_len - 60: , :]

x_test =[]

y_test= dataset[training_data_len:, :]

for i in range (60,len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
    
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt (np.mean(predictions - y_test) **2)

train = data[:training_data_len]

valid = data[training_data_len:]
valid['Predictions'] = predictions

#plotting predictrion of apple stock

plt.figure(figsize=(16, 7))

plt.title('Apple stock prediction Model')

plt.xlabel('Date', fontsize=17)

plt.ylabel('USD $', fontsize=17)

plt.plot(train ['Close'])

plt.plot(valid[['Close', 'Predictions']])

plt.legend (['Trained', 'actual Value', 'Predictions'], loc='lower right') 
plt.show()


# In[20]:


print(valid)


# In[43]:


#evaluation
import math
from sklearn.metrics import mean_squared_error

rmse_score = math.sqrt(mean_squared_error(y_test,predictions))
print(f"Root Mean Squared Error(test) :{rmse_score}")

from sklearn.metrics import r2_score
print(f"prediction accuracy :{r2_score(y_test,predictions)}")


# In[30]:


#testing our model for given date for prediction of stock closing price

user=input("Enter date:")
aaple_stock = dataread.DataReader('AAPL', data_source='yahoo', start='2012-01-03', end=user)

new_df = aaple_stock.filter (['Close'])

prev_60_days = new_df[-60:].values

prev_60_days_scaled = scaler.transform(prev_60_days) 

x_test= []

x_test.append(prev_60_days_scaled)

x_test = np.array(x_test)

x_test = np.reshape (x_test, (x_test.shape [0], x_test.shape[1], 1))

pred_price = model.predict(x_test)

pred_price = scaler.inverse_transform(pred_price)

print ('predicted prcice of stock is :US $',pred_price[0][0])


# In[33]:


#checking todays stock price
apple_stock2= dataread.DataReader('AAPL', data_source='yahoo', start='2020-07-01', end='2020-07-01')
print(apple_stock2['Close'])


# In[27]:





# In[ ]:




