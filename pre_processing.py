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

def parser(filepath):
    dates = []
    inc_prices = []
    dataset = pandas.read_csv(filepath)
    for row in dataset.values:
        cur_date = row[0].split('-')
        cur_date[1] = months[cur_date[1].lower()]
        cur_date[0] = int(cur_date[0])
        cur_date[2] = int(cur_date[2])
        dates.append(cur_date)
        # print(cur_date)
        inc_prices.append(row[4]-row[1])
    return dates,inc_prices

def normalize_data(data):
	data = np.asarray(data,dtype=np.float32)
	mean = data.mean(axis=0)
	std = data.std(axis=0)
	data = (data - mean) / std
	return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dates,inc_prices = parser(filepath='data/googl.csv')
    # print(inc_prices)
    normalized_inc_prices = normalize_data(inc_prices)
    plt.plot(normalized_inc_prices)
    plt.show()
