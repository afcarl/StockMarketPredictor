import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import pre_processing
import math
import numpy as np
import sentiment_analyser

def predict():
    # getting data
    look_back = 1
    path = 'data/googl.csv'
    dates,inc_prices = pre_processing.parse(path)  # dates is not required but have to use as a later function demands it
    normalized_inc_prices = pre_processing.normalize_data(inc_prices)
    waste1,train_prices,waste2,test_prices = pre_processing.split_data(dates,inc_prices)
    trainX,trainY = pre_processing.change_dataset(train_prices,look_back)
    testX,testY = pre_processing.change_dataset(test_prices,look_back)
    print(len(trainX),len(trainY))

    # create model
    model = Sequential()
    model.add(Dense(8,input_dim=look_back,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(trainX,trainY,epochs=100,batch_size=2,verbose=1)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    predict_train = model.predict(trainX)
    predict_test = model.predict(testX)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(inc_prices)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(predict_train)+look_back] = predict_train.reshape(1,len(trainX))

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(inc_prices)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(predict_train)+(look_back*2)+1:len(inc_prices)-1] = predict_test.reshape(1,len(testX))

    # plot baseline and predictions
    plt.plot(inc_prices)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


stock = 'google'

api = sentiment_analyser.authorize()
if not sentiment_analyser.search_tweets(api,stock) > 0:
    ch = input('%s Stock Has bad reviews recently! Do you still wanna predict (y/n)' %stock)
    if ch == 'y':
        predict()
    else:
        exit()
else:
    predict()
