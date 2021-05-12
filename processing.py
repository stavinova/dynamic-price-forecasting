import pandas as pd
from scipy.stats import spearmanr

class DataProcessing:
    def __init__(self, df, fare):
        self.df = df
        self.preproc_data = self.preprocessing()
        self.fare = fare
        self.price, self.trains = self.get_predictants()
        self.ave, self.renfe, self.tren, self.ave_bar, self.renfe_bar, self.tren_bar = self.get_predictors()
        self.cor_price, self.cor_trains = self.correlation_analysis()
        
    def preprocessing(self):
        for i in ['insert_date', 'departure', 'arrival']:
            self.df[i] = pd.to_datetime(self.df[i])
        self.df = self.df.dropna(subset = ['price'])
        self.df['origin'] = self.df['origin'].str.replace(u"Ó", "O")
        self.df['destination'] = self.df['destination'].str.replace(u"Ó", "O")
        self.df['origin'] = self.df['origin'].str.replace(u"Á", "A")
        self.df['destination'] = self.df['destination'].str.replace(u"Á", "A")
        self.df['route'] = self.df['origin']+' to '+ self.df['destination']
        self.df = self.df.drop_duplicates(subset=['departure', 'arrival', 'vehicle_type', 'vehicle_class', 'price', 'fare', 'seats', 'route', 'insert_date'])
        self.df = self.df.reset_index(drop = True)
        self.df['delta'] = self.df.apply(lambda row: (row['departure'] - row['insert_date']).days, axis = 1)
        self.df = self.df[self.df['delta'] >= 2].sort_values('insert_date').drop_duplicates(subset = ['departure', 'route', 'vehicle_type', 'vehicle_class',
                                                                          'fare', 'duration'], keep = 'last')
        self.df = self.df.reset_index(drop = True)
        return self.df
    
    def get_predictants(self):
        dp = self.preproc_data.groupby(['route']).groups
        self.preproc_data['departure_date'] = self.preproc_data['departure'].dt.date
        group_price = self.preproc_data.iloc[dp['MADRID to BARCELONA']][(self.preproc_data['price'] != 0)  & 
                                            (self.preproc_data['fare'] == self.fare)].groupby(['departure_date'])['price'].mean()
        y_num = self.preproc_data.iloc[dp['MADRID to BARCELONA']][(self.preproc_data['price'] != 0)  & 
                                      (self.preproc_data['fare'] == self.fare)].drop_duplicates(subset=['departure']).groupby(['departure_date'])['id'].count()
        return (group_price.values[:75], y_num.values[:75])
    
    def get_predictors(self):
        predictors = []
        for name in ['ave.csv', 'renfe.csv', 'tren.csv', 'ave_bar.csv', 'renfe_bar.csv', 'tren_bar.csv']:
            file = pd.read_csv("data/" + name)
            file = file.reset_index()
            file.columns = file.iloc[0]
            file = file[1:]
            predictors.append(file)
        predictors[0] = predictors[0]['ave "madrid barcelona": (По всему миру)'].values.astype(int)
        predictors[1] = predictors[1]['renfe "madrid barcelona": (По всему миру)'].values.astype(int)
        predictors[2] = predictors[2]['tren "madrid barcelona": (По всему миру)'].values.astype(int)
        predictors[3] = predictors[3]['ave barcelona: (Мадрид)'].values.astype(int)
        predictors[4] = predictors[4]['renfe barcelona: (Мадрид)'].values.astype(int)
        predictors[5] = predictors[5]['tren barcelona: (Мадрид)'].values.astype(int)
        return predictors
    
    def correlation_analysis(self):
        cor_p, cor_t = [], []
        for predictor in [self.ave, self.renfe, self.tren, self.ave_bar, self.renfe_bar, self.tren_bar]:
            max_price, max_trains = 0, 0
            ind_price, ind_trains = 0, 0
            for i in range(15):
                if i != 0:
                    coef_p, p_p = spearmanr(predictor[16-i:-4-i], self.price)
                    coef_t, p_t = spearmanr(predictor[16-i:-4-i], self.trains)
                    if coef_p > max_price:
                        max_price = coef_p
                        ind_price = i
                    if coef_t > max_trains:
                        max_trains = coef_t
                        ind_trains = i
            cor_p.append(ind_price)
            cor_t.append(ind_trains)
        return (cor_p, cor_t)