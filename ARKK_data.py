# Sourcing the data for all the stocks of S&P500
# Load libraries
import pandas as pd
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
import csv
# Yahoo for dataReader
import yfinance as yf
yf.pdr_override()

import warnings
warnings.filterwarnings('ignore')

# Load dataset


def get_data_from_yahoo():
    ticker = "ARKK"
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    dataset = yf.download(ticker, start=start, end=end)
    dataset.to_csv("ARKK_data.csv")
    return dataset.to_csv("ARKK_data.csv")


get_data_from_yahoo()
