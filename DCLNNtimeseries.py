"""
Created on Thu Nov 29 09:11:20 2018
@author: zzt
"""
'''
# LSTM for failure time prediction with regression framing
'''
import numpy
import time
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
start = time.clock()

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

numpy.random.seed(4)

dataframe = read_csv('E3.csv', usecols=[1], engine='python', skipfooter=1)
dataset = dataframe.values

dataset = dataset.astype('float32') 

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1,trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0],1 ,testX.shape[1]))

model = Sequential()
model.add(Conv1D(filters=2,kernel_size=2, padding='same', strides=2, activation='relu',input_shape=(1,look_back)))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(6, input_shape=(1, look_back),return_sequences=True))
model.add(LSTM(8, input_shape=(1, look_back),return_sequences=True))
model.add(Flatten())
#model.add(Dropout(0.1))
model.add(Dense(6,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
hs=model.fit(trainX, trainY, epochs=100, batch_size=1, validation_data=(testX,testY),verbose=2)

train_loss=hs.history['loss']
test_loss=hs.history['val_loss']
plt.plot(train_loss,label='train')
plt.plot(test_loss,label='test')
plt.legend()
plt.show()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict               

plt.plot(scaler.inverse_transform(dataset),'g')
plt.plot(trainPredictPlot,'b')
plt.plot(testPredictPlot,'r*')
plt.show()

finalX = numpy.reshape(test[-1], (1, 1, testX.shape[1])) 
featruePredict = model.predict(finalX) 
featruePredict = scaler.inverse_transform(featruePredict) 
print('The next failure time of escalator is: ',featruePredict)

elapsed = (time.clock() - start)
print("Time used:",elapsed,"s")
#save 
'''
numpy.savetxt("testPredict.txt",testPredict)
numpy.savetxt("testY.txt",testY)
numpy.savetxt("trainPredict.txt",trainPredict)
numpy.savetxt("trainY.txt",trainY)
'''