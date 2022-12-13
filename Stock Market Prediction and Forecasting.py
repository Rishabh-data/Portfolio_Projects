#!/usr/bin/env python
# coding: utf-8

# In[31]:


### Data Collection
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
key = "879bf1c196c59761f67b5b89595b11a5093fa2e1" #Tiingo's API key


# In[21]:


df = pdr.get_data_tiingo('AAPL', api_key=key)


# In[22]:


df.head()


# In[6]:


#df.to_csv('D:\Ricky\Data Science\Portfolio Projects\Stock Market Prediction\AAPL.csv')


# ### We are going to predict Close Price i.e. close will be our target variable

# In[23]:


df.head()


# In[24]:


df.head().index


# In[25]:


df = df.reset_index()
df


# In[17]:


#Seperating our target variable
df1 = df.reset_index()['close']
df1.shape


# In[18]:


#Plotting the close prices
plt.plot(df1)


# In[36]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 
#Scaling our Target Variable
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1)) #Scaling values between 0 & 1
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[39]:


df1


# In[40]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[102]:


train_data.shape, test_data.shape


# In[47]:



import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[48]:



# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100 #Taking last 100 values to predict next value
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[49]:


print(X_train.shape), print(y_train.shape)


# In[52]:


print(X_test.shape), print(ytest.shape)


# In[53]:



# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[54]:



### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[55]:



model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[56]:



model.summary()


# In[58]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[59]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[60]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[61]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[62]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[97]:


numpy.empty_like(df1)


# In[100]:


train_predict.shape, test_predict.shape


# In[103]:


len(numpy.empty_like(df1))


# In[110]:


trainPredictPlot[99:]


# In[107]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[96]:


len(trainPredictPlot)


# In[92]:


len(train_predict)


# ### Predicting Stock Prices for Next 30 days

# In[67]:


len(test_data)


# In[112]:


#To predict for new data, we need last 100 records of test data
x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[115]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist() #Last 100 days of test_data


# In[70]:


# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30): # Running loop for 30 times as we want next 30 days prediction
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:]) #Ensuring we take last 100 records as inpput data for prediction
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist()) #Prediction is added to input thus increasing input list size to 101
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output) #Displaying the list of all 30 predictions


# In[71]:


day_new=np.arange(1,101) # 100 indexes for input data
day_pred=np.arange(101,131) # 30 indexes for 30 predictions


# In[119]:


len(df1)


# In[121]:


plt.plot(day_new,scaler.inverse_transform(df1[1159:])) #From 1159 as previous 100 days data of df1 has been taken
plt.plot(day_pred,scaler.inverse_transform(lst_output)) # 30 days prediction plotted in orange color in below plot


# ### View Complete output as a continuous graph

# In[125]:


df3=df1.tolist()
df3.extend(lst_output) #Combining output predictions to existing data
plt.plot(df3[1000:])


# In[76]:


df3=scaler.inverse_transform(df3).tolist()


# In[77]:


plt.plot(df3)


# In[ ]:




