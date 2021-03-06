import pandas
import numpy as np

months = {  'jan':1,
            'feb':2,
            'mar':3,
            'apr':4,
            'may':5,
            'jun':6,
            'jul':7,
            'aug':8,
            'sep':9,
            'oct':10,
            'nov':11,
            'dec':12
            }

def parse(filepath):
    dates = []
    inc_prices = []
    dataset = pandas.read_csv(filepath)
    for row in dataset.values:
        cur_date = row[0].split('-')
        cur_date[1] = months[cur_date[1].lower()]
        cur_date[0] = int(cur_date[0])
        cur_date[2] = int(cur_date[2])
        dates.append(cur_date)
        inc_prices.append(row[4]-row[1])
    return np.asarray(dates),np.asarray(inc_prices)

def normalize_data(data):
	data = np.asarray(data,dtype=np.float32)
	mean = data.mean(axis=0)
	std = data.std(axis=0)
	data = (data - mean) / std
	return data

def split_data(dates,normalized_inc_prices):
    train_dates = dates[0:len(dates)-len(dates)//10]
    test_dates = dates[len(train_dates):len(dates)]
    assert len(dates) == len(normalized_inc_prices)
    train_prices = normalized_inc_prices[0:len(dates)-len(dates)//10]
    test_prices = normalized_inc_prices[len(train_dates):len(dates)]
    return train_dates,train_prices,test_dates,test_prices

def change_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		dataX.append(dataset[i:(i+look_back)])
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    dates,inc_prices = parse(filepath='data/googl.csv')
    normalized_inc_prices = normalize_data(inc_prices)
    train_dates,train_prices,test_dates,test_prices = split_data(dates,normalized_inc_prices)   # Dates datset is unused but processed
    trainX,trainY = change_dataset(train_prices)
    testX,testY = change_dataset(test_prices)
    plt.plot(trainX)
    plt.show()
