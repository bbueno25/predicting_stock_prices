"""
DOCSTRING
"""
import csv
import matplotlib.pyplot as pyplot
import numpy
import sklearn.svm as svm
import textblob
import tweepy

class Challenge:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        consumer_key= 'CONSUMER_KEY_HERE'
        consumer_secret= 'CONSUMER_SECRET_HERE'
        access_token='ACCESS_TOKEN_HERE'
        access_token_secret='ACCESS_TOKEN_SECRET_HERE'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        public_tweets = api.search('company_name')
        for tweet in public_tweets:    
            analysis = textblob.TextBlob(tweet.text)
            print(analysis.sentiment)
        dates = []
        prices = []

    def __call__(self):
        """
        DOCSTRING
        """
        self.get_data('your_company_stock_data.csv')
        predicted_price = self.predict_price(dates, prices, 29)
        print(predicted_price)

    def get_data(self, filename):
        """
        DOCSTRING
        """
        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)
            for row in csvFileReader:
                dates.append(int(row[0].split('-')[0]))
                prices.append(float(row[1]))

    def predict_price(self, dates, prices, x):
        """
        In this function, build your neural network model using Keras, train it,
        then have it predict the price on a given day. 
        """
        return 0

class Demo:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        pyplot.switch_backend('newbackend')
        dates = []
        prices = []

    def __call__(self):
        """
        DOCSTRING
        """
        self.get_data('data/aapl.csv')
        predicted_price = self.predict_price(dates, prices, 29)

    def get_data(self, filename):
        """
        DOCSTRING
        """
        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)
            for row in csvFileReader:
                dates.append(int(row[0].split('-')[0]))
                prices.append(float(row[1]))

    def predict_price(self, dates, prices, x):
        """
        DOCSTRING
        """
        dates = numpy.reshape(dates, (len(dates), 1))
        svr_lin = svm.SVR(kernel= 'linear', C= 1e3)
        svr_poly = svm.SVR(kernel= 'poly', C= 1e3, degree= 2)
        svr_rbf = svm.SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
        svr_rbf.fit(dates, prices)
        svr_lin.fit(dates, prices)
        svr_poly.fit(dates, prices)
        pyplot.scatter(dates, prices, color= 'black', label= 'Data')
        pyplot.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
        pyplot.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model')
        pyplot.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model')
        pyplot.xlabel('Date')
        pyplot.ylabel('Price')
        pyplot.title('Support Vector Regression')
        pyplot.legend()
        pyplot.show()
        return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

if __name__ == '__main__':
    demo = Demo()
    demo()