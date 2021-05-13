from processing import DataProcessing

import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from datetime import date
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def mean_absolute_percentage_error(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 3)

class Baseline_predictor(DataProcessing):
    def __init__(self, df, predict_ahead, fare, predictant):
        DataProcessing.__init__(self, df, fare)
        self.predict_ahead = predict_ahead
        self.predictant = predictant
        self.train, self.test = self.train_test_split()
        self.rmse, self.mape = self.model()
        
    def train_test_split(self):
        sdate = date(2019, 4, 14)
        edate = date(2019, 6, 27)
        dates = pd.date_range(sdate,edate,freq='d')
        dates.freq = None
        if self.predictant == 'mean':
            d = {'date': dates, 'mean': self.price}
        elif self.predictant == 'trains':
            d = {'date': dates, 'trains': self.trains}
        df = pd.DataFrame(d)
        df = df.set_index('date')
        size = int(len(df) * 0.8)
        df_train, df_test = df[:size], df[size:]
        return (df_train, df_test)
    
    def model(self):
        history = [x for x in self.train[self.predictant].values]
        predictions = list()
        for t in range(int(math.trunc(len(self.test)/self.predict_ahead))):
            output = history[-1]
            for i in range(self.predict_ahead):
                predictions.append(output)
                obs = self.test[self.predictant].values[t+i]
                history.append(obs)
        rmse = math.sqrt(mean_squared_error(self.test[self.predictant][:4*int(math.trunc(len(self.test) / self.predict_ahead))], predictions))
        MAPE = mean_absolute_percentage_error(self.test[self.predictant][:4*int(math.trunc(len(self.test) / self.predict_ahead))], predictions)
        return (rmse, MAPE)

class ARIMA_predictor(DataProcessing):
    def __init__(self, df, predict_ahead, fare, predictant, p, d, q):
        DataProcessing.__init__(self, df, fare)
        self.predict_ahead = predict_ahead
        self.predictant = predictant
        self.train, self.test = self.train_test_split()
        self.p = p
        self.q = q
        self.d = d
        self.rmse, self.mape = self.model()
        
    def train_test_split(self):
        sdate = date(2019, 4, 14)
        edate = date(2019, 6, 27)
        dates = pd.date_range(sdate,edate,freq='d')
        dates.freq = None
        if self.predictant == 'mean':
            d = {'date': dates, 'mean': self.price}
        elif self.predictant == 'trains':
            d = {'date': dates, 'trains': self.trains}
        df = pd.DataFrame(d)
        df = df.set_index('date')
        size = int(len(df) * 0.8)
        df_train, df_test = df[:size], df[size:]
        return (df_train, df_test)
    
    def model(self):
        history = [x for x in self.train[self.predictant].values]
        predictions = list()
        for t in range(int(math.trunc(len(self.test) / self.predict_ahead))):
            model = sm.tsa.statespace.SARIMAX(history, order=(self.p,self.d,self.q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            output = model_fit.predict(start=len(self.train)+self.predict_ahead*t, end=len(self.train)+3+self.predict_ahead*t)
            for i in range(len(output)):
                predictions.append(output[i])
                obs = self.test[self.predictant].values[t+i]
                history.append(obs)
        rmse = math.sqrt(mean_squared_error(self.test[self.predictant][:4*int(math.trunc(len(self.test) / self.predict_ahead))], predictions))
        MAPE = mean_absolute_percentage_error(self.test[self.predictant][:4*int(math.trunc(len(self.test) / self.predict_ahead))], predictions)
        return (rmse, MAPE)
    
class ARIMAX_predictor(DataProcessing):
    def __init__(self, df, predict_ahead, fare, predictant, predictor, p, d, q):
        DataProcessing.__init__(self, df, fare)
        self.predict_ahead = predict_ahead
        self.predictant = predictant
        self.train, self.test = self.train_test_split()
        self.p = p
        self.q = q
        self.d = d
        self.exogs = predictor
        self.rmse, self.mape = self.model()
        
    def train_test_split(self):
        sdate = date(2019, 4, 14)
        edate = date(2019, 6, 27)
        dates = pd.date_range(sdate,edate,freq='d')
        dates.freq = None
        if self.predictant == 'mean':
            d = {'date': dates, 'mean': self.price, 'ave': self.ave[16-self.cor_price[0]:-4-self.cor_price[0]], 
                 'tren': self.tren[16-self.cor_price[2]:-4-self.cor_price[2]], 'renfe': self.renfe[16-self.cor_price[1]:-4-self.cor_price[1]], 
                 'ave_bar': self.ave_bar[16-self.cor_price[3]:-4-self.cor_price[3]], 'tren_bar': self.tren_bar[16-self.cor_price[5]:-4-self.cor_price[5]], 
                 'renfe_bar': self.renfe_bar[16-self.cor_price[4]:-4-self.cor_price[4]]}
        elif self.predictant == 'trains':
            d = {'date': dates, 'trains': self.trains, 'ave': self.ave[16-self.cor_trains[0]:-4-self.cor_trains[0]], 
                 'tren': self.tren[16-self.cor_trains[2]:-4-self.cor_trains[2]], 'renfe': self.renfe[16-self.cor_trains[1]:-4-self.cor_trains[1]], 
                 'ave_bar': self.ave_bar[16-self.cor_trains[3]:-4-self.cor_trains[3]], 'tren_bar': self.tren_bar[16-self.cor_trains[5]:-4-self.cor_trains[5]], 
                 'renfe_bar': self.renfe_bar[16-self.cor_trains[4]:-4-self.cor_trains[4]]}
        df = pd.DataFrame(d)
        df = df.set_index('date')
        size = int(len(df) * 0.8)
        df_train, df_test = df[:size], df[size:]
        return (df_train, df_test)
    
    def model(self):
        history = [x for x in self.train[self.predictant].values]
        ex1 = [x for x in self.train[self.exogs].values]
        ex = np.transpose(np.array([ex1]))
        predictions = list()
        for t in range(int(math.trunc(len(self.test)/self.predict_ahead))):
            model = sm.tsa.statespace.SARIMAX(history, exog=ex, order=(self.p,self.d,self.q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            exog1 = []
            for i in range(self.predict_ahead):
                exog1.append(self.test[self.exogs].values[t+i])
            exog = np.transpose(np.array([exog1]))
            output = model_fit.predict(start=len(self.train)+self.predict_ahead*t, end=len(self.train)+3+self.predict_ahead*t, exog=exog)
            for i in range(self.predict_ahead):
                predictions.append(output[i])
                obs = self.test[self.predictant].values[t+i]
                history.append(obs)
            ex = np.vstack((ex, exog))
        rmse = math.sqrt(mean_squared_error(self.test[self.predictant][:4*int(math.trunc(len(self.test) / self.predict_ahead))], predictions))
        MAPE = mean_absolute_percentage_error(self.test[self.predictant][:4*int(math.trunc(len(self.test) / self.predict_ahead))], predictions)
        return (rmse, MAPE)
    
class LSTM_predictor(DataProcessing):
    def __init__(self, df, predict_ahead, fare, predictant, NFILTERS, BATCH_SIZE, NB_EPOCHS):
        DataProcessing.__init__(self, df, fare)
        self.predict_ahead = predict_ahead
        self.predictant = predictant
        self.train, self.test, self.scaler = self.train_test_split()
        self.NFILTERS = NFILTERS
        self.NB_EPOCHS = NB_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.rmse, self.mape = self.model()
        
    def train_test_split(self):
        sdate = date(2019, 4, 14)
        edate = date(2019, 6, 27)
        dates = pd.date_range(sdate,edate,freq='d')
        dates.freq = None
        if self.predictant == 'mean':
            d = {'date': dates, 'mean': self.price}
        elif self.predictant == 'trains':
            d = {'date': dates, 'trains': self.trains}
        df = pd.DataFrame(d)
        df = df.set_index('date')
        size = int(len(df) * 0.8)
        df_val = df.values
        df_val = df_val.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_val = scaler.fit_transform(df_val)
        df_train, df_test = df_val[0:size], df_val[size:len(df_val)]
        np.random.seed(7)
        return (df_train, df_test, scaler)
    
    def create_dataset_test(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in np.arange(0, len(dataset)-look_back-1, 4):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            b = dataset[(i+look_back):(i+look_back+look_back), 0]
            dataY.append(b)
        return np.array(dataX), np.array(dataY)
    
    def create_dataset_train(self, dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            b = dataset[(i+look_back):(i+look_back+look_back), 0]
            dataY.append(b)
        return np.array(dataX), np.array(dataY)
    
    def model(self):
        trainX, trainY = self.create_dataset_train(self.train, look_back=self.predict_ahead)
        testX, testY = self.create_dataset_test(self.test, look_back=self.predict_ahead)
        trainX = trainX[:-2]
        trainY = trainY[:-2]
        testX = testX[:-1]
        testY = testY[:-1]
        trainY = np.stack(trainY)
        testY = np.stack(testY)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        model = Sequential()
        model.add(LSTM(self.NFILTERS, input_shape=(1, self.predict_ahead)))
        model.add(Dense(self.predict_ahead))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs = self.NB_EPOCHS, batch_size=self.BATCH_SIZE, verbose=2)
        predictions = list()
        for t in range(len(testX)):
            X = testX[t]
            yhat = model.predict(np.reshape(X, (1,1,self.predict_ahead)))
            predictions.append(yhat[0])
        testPredict = self.scaler.inverse_transform([np.concatenate(predictions)])
        testY = self.scaler.inverse_transform([np.concatenate(testY)])
        rmse = math.sqrt(mean_squared_error(testY[0], testPredict[0]))
        MAPE = mean_absolute_percentage_error(testY[0], testPredict[0])
        return (rmse, MAPE)
    
class MV_LSTM_predictor(DataProcessing):
    def __init__(self, df, predict_ahead, fare, predictant, predictor, NFILTERS, BATCH_SIZE, NB_EPOCHS):
        DataProcessing.__init__(self, df, fare)
        self.predict_ahead = predict_ahead
        self.predictant = predictant
        self.exogs = predictor
        self.train, self.test, self.scaler, self.size, self.df_val = self.train_test_split()
        self.NFILTERS = NFILTERS
        self.NB_EPOCHS = NB_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.rmse, self.mape = self.model()
        
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
        
    def train_test_split(self): 
        sdate = date(2019, 4, 14)
        edate = date(2019, 6, 27)
        dates = pd.date_range(sdate,edate,freq='d')
        dates.freq = None
        if self.predictant == 'mean':
            d = {'date': dates, 'mean': self.price, 'ave': self.ave[16-self.cor_price[0]:-4-self.cor_price[0]], 
                 'tren': self.tren[16-self.cor_price[2]:-4-self.cor_price[2]], 'renfe': self.renfe[16-self.cor_price[1]:-4-self.cor_price[1]], 
                 'ave_bar': self.ave_bar[16-self.cor_price[3]:-4-self.cor_price[3]], 'tren_bar': self.tren_bar[16-self.cor_price[5]:-4-self.cor_price[5]], 
                 'renfe_bar': self.renfe_bar[16-self.cor_price[4]:-4-self.cor_price[4]]}
        elif self.predictant == 'trains':
            d = {'date': dates, 'trains': self.trains, 'ave': self.ave[16-self.cor_trains[0]:-4-self.cor_trains[0]], 
                 'tren': self.tren[16-self.cor_trains[2]:-4-self.cor_trains[2]], 'renfe': self.renfe[16-self.cor_trains[1]:-4-self.cor_trains[1]], 
                 'ave_bar': self.ave_bar[16-self.cor_trains[3]:-4-self.cor_trains[3]], 'tren_bar': self.tren_bar[16-self.cor_trains[5]:-4-self.cor_trains[5]], 
                 'renfe_bar': self.renfe_bar[16-self.cor_trains[4]:-4-self.cor_trains[4]]}
        df = pd.DataFrame(d)
        df = df.set_index('date')
        df = df[[self.predictant, self.exogs]]
        size = int(len(df) * 0.8)
        df_val = df.values
        df_val = df_val.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_val = scaler.fit_transform(df_val)
        reframed = self.series_to_supervised(df_val, self.predict_ahead, self.predict_ahead)
        values = reframed.values
        train = values[0:size, :]
        test = values[size:len(values), :]
        #for predictions we leave only non-overlapping sequences
        test_small = np.array([test[0], test[4]])
        return (train, test_small, scaler, size, df_val)
    
    def model(self):
        train_X, train_y = self.train[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]], self.train[:, [8, 10, 12, 14]]
        test_X, test_y = self.test[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]], self.test[:, [8, 10, 12, 14]]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        model = Sequential()
        model.add(LSTM(self.NFILTERS, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(self.predict_ahead))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_X, train_y, epochs=self.NB_EPOCHS, batch_size=self.BATCH_SIZE, verbose=2, shuffle=False)
        yhat = model.predict(test_X)
        check_test_X = self.df_val[self.size+self.predict_ahead:self.size+2*self.predict_ahead+self.predict_ahead]
        yhat = yhat.reshape((2*self.predict_ahead, 1))
        inv_yhat = np.concatenate((yhat, check_test_X[:, 1:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((2*self.predict_ahead, 1))
        inv_y = np.concatenate((test_y, check_test_X[:, 1:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        MAPE = mean_absolute_percentage_error(inv_y, inv_yhat)
        return (rmse, MAPE)