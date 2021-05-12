import pandas as pd

from models import Baseline_predictor, ARIMA_predictor, ARIMAX_predictor, LSTM_predictor, MV_LSTM_predictor

class PricePrediction:
    __baseline_params = {'predict ahead': 4,
                      'fare': 'Flexible',
                      'predictant': 'mean'}
    __arima_params = {'predict ahead': 4,
                      'fare': 'Flexible',
                      'predictant': 'mean',
                      'p': 1,
                      'd': 0,
                      'q': 1}
    __arimax_params = {'predict ahead': 4,
                       'fare': 'Flexible',
                      'predictant': 'mean',
                      'predictor': 'ave',
                      'p': 1,
                      'd': 0,
                      'q': 1}
    __lstm_params = {'predict ahead': 4,
                     'fare': 'Flexible',
                     'predictant': 'mean',
                     'number of filters': 4,
                     'batch size': 1,
                     'number of epochs': 100}
    __mv_lstm_params = {'predict ahead': 4,
                     'fare': 'Flexible',
                     'predictant': 'mean',
                     'predictor': 'ave',
                     'number of filters': 4,
                     'batch size': 1,
                     'number of epochs': 100}
    def set_baseline_model(self, predict_ahead = 4, fare = 'Flexible', predictant = 'mean'):
        self.__baseline_params['predict ahead'] = predict_ahead
        self.__baseline_params['fare'] = fare
        self.__baseline_params['predictant'] = predictant
        
    def set_ARIMA_model(self, predict_ahead = 4, fare = 'Flexible', predictant = 'mean', p = 1, d = 0, q = 1):
        self.__arima_params['predict ahead'] = predict_ahead
        self.__arima_params['fare'] = fare
        self.__arima_params['predictant'] = predictant
        self.__arima_params['p'] = p
        self.__arima_params['d'] = d
        self.__arima_params['q'] = q
        
    def set_ARIMAX_model(self, predict_ahead = 4, fare = 'Flexible', predictant = 'mean', predictor = 'ave', p = 1, d = 0, q = 1):
        self.__arimax_params['predict ahead'] = predict_ahead
        self.__arimax_params['fare'] = fare
        self.__arimax_params['predictant'] = predictant
        self.__arimax_params['predictor'] = predictor
        self.__arimax_params['p'] = p
        self.__arimax_params['d'] = d
        self.__arimax_params['q'] = q
        
    def set_LSTM_model(self, predict_ahead = 4, fare = 'Flexible', predictant = 'mean', NFILTERS = 4, BATCH_SIZE = 1, NB_EPOCHS = 100):
        self.__lstm_params['predict ahead'] = predict_ahead
        self.__lstm_params['fare'] = fare
        self.__lstm_params['predictant'] = predictant
        self.__lstm_params['number of filters'] = NFILTERS
        self.__lstm_params['batch size'] = BATCH_SIZE
        self.__lstm_params['number of epochs'] = NB_EPOCHS
        
    def set_MV_LSTM_model(self, predict_ahead = 4, fare = 'Flexible', predictant = 'mean', predictor = 'ave', NFILTERS = 4, BATCH_SIZE = 1, NB_EPOCHS = 100):
        self.__mv_lstm_params['predict ahead'] = predict_ahead
        self.__mv_lstm_params['fare'] = fare
        self.__mv_lstm_params['predictant'] = predictant
        self.__mv_lstm_params['predictor'] = predictor
        self.__mv_lstm_params['number of filters'] = NFILTERS
        self.__mv_lstm_params['batch size'] = BATCH_SIZE
        self.__mv_lstm_params['number of epochs'] = NB_EPOCHS
        
    def infer(self, df):
        model_b = Baseline_predictor(df, self.__baseline_params['predict ahead'], self.__baseline_params['fare'], self.__baseline_params['predictant'])
        model_arima = ARIMA_predictor(df, self.__arima_params['predict ahead'], self.__arima_params['fare'], self.__arima_params['predictant'], 
                                      self.__arima_params['p'], self.__arima_params['d'], self.__arima_params['q'])
        model_arimax = ARIMAX_predictor(df, self.__arimax_params['predict ahead'], self.__arimax_params['fare'], self.__arimax_params['predictant'],
                                        self.__arimax_params['predictor'], self.__arimax_params['p'], self.__arimax_params['d'], self.__arimax_params['q'])
        model_lstm = LSTM_predictor(df, self.__lstm_params['predict ahead'], self.__lstm_params['fare'], self.__lstm_params['predictant'],
                                    self.__lstm_params['number of filters'], self.__lstm_params['batch size'], self.__lstm_params['number of epochs'])
        model_mv_lstm = MV_LSTM_predictor(df, self.__mv_lstm_params['predict ahead'], self.__mv_lstm_params['fare'], self.__mv_lstm_params['predictant'],
                                    self.__mv_lstm_params['predictor'], self.__lstm_params['number of filters'], self.__lstm_params['batch size'], 
                                    self.__lstm_params['number of epochs'])
        pred_b, true_b = model_b.pred, model_b.true
        pred_a, true_a = model_arima.pred, model_arima.true
        pred_ax, true_ax = model_arimax.pred, model_arimax.true
        pred_l, true_l = model_lstm.pred, model_lstm.true
        pred_ml, true_ml = model_mv_lstm.pred, model_mv_lstm.true
        return pred_b, true_b, pred_a, true_a, pred_ax, true_ax, pred_l, true_l, pred_ml, true_ml

data = pd.read_csv("data/thegurus-opendata-renfe-trips.csv")
PP = PricePrediction()
PP.set_baseline_model(predict_ahead = 4, fare = 'Flexible', predictant = 'mean')
PP.set_ARIMA_model(predict_ahead = 4, fare = 'Flexible', predictant = 'mean', p = 1, d = 0, q = 1)
PP.set_ARIMAX_model(predict_ahead = 4, fare = 'Flexible', predictant = 'mean', predictor = 'ave', p = 1, d = 0, q = 1)
PP.set_LSTM_model(predict_ahead = 4, fare = 'Flexible', predictant = 'mean', NFILTERS = 4, BATCH_SIZE = 1, NB_EPOCHS = 100)
PP.set_MV_LSTM_model(predict_ahead = 4, fare = 'Flexible', predictant = 'mean', predictor = 'ave',NFILTERS = 4, BATCH_SIZE = 1, NB_EPOCHS = 100)
pred_b, true_b, pred_a, true_a, pred_ax, true_ax, pred_l, true_l, pred_ml, true_ml = PP.infer(df = data)