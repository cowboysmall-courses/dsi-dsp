
"""

Global market indices of interest:

    NSEI:  Nifty 50
    DJI:   Dow Jones Index
    IXIC:  Nasdaq
    HSI:   Hang Seng
    N225:  Nikkei 225
    GDAXI: Dax
    VIX:   Volatility Index

"""



# %% 1 - import required libraries
import pandas as pd

import nltk
# nltk.download('all', quiet = True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from string import punctuation, digits


# %% 1 - import data and check the head
with open("./data/tweets/Tweets 12-6-24.txt") as file:
    tweets = [line.rstrip() for line in file]



# %% 1 - 
data = pd.DataFrame([line for line in tweets if len(line) > 0], columns= ["Tweets"])
data.head()
#                                               Tweets
# 0  #bankNifty 50100 ce looks good at 70+-2 for a ...
# 1  "#market #banknifty #OptionsTrading #optionbuy...
# 2  PENNY STOCK MADHUCON PROJECTS LTD cmp-11 FOLLO...
# 3  #Nifty50 has been in a healthy uptrend since t...
# 4  #Gravita #livetrading #stockstowatch #stocksin...



# %% 1 - 
stop_words    = set(stopwords.words('english'))
remove_punc   = str.maketrans('', '', punctuation)
remove_digits = str.maketrans('', '', digits)

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet.lower().translate(remove_punc).translate(remove_digits))
    return " ".join([word for word in tokens if word not in stop_words])



# %% 1 - 
data["Cleaned_Tweets"] = data["Tweets"].apply(preprocess_tweet)
data["Cleaned_Tweets"].head()
# 0           banknifty ce looks good target nifty nifty
# 1    market banknifty optionstrading optionbuying t...
# 2    penny stock madhucon projects ltd cmp followht...
# 3    nifty healthy uptrend since beginning year did...
# 4    gravita livetrading stockstowatch stocksinfocu...
# Name: Cleaned_Tweets, dtype: object



# %% 1 - 
tweet_words = data["Cleaned_Tweets"].str.cat(sep = " ")



# %% 1 - 
freq_dist    = nltk.FreqDist(tweet_words.split())
freq_dist_df = pd.DataFrame(freq_dist.most_common(30), columns=["Words", "Frequency"])
freq_dist_df.head(30)
#                Words  Frequency
# 0              nifty        399
# 1          banknifty        104
# 2        stockmarket         71
# 3          niftybank         45
# 4   stockmarketindia         44
# 5             sensex         43
# 6             stocks         38
# 7     optionstrading         36
# 8                bse         34
# 9     breakoutstocks         31
# 10           trading         30
# 11            market         29
# 12             india         26
# 13               ipm         25
# 14              good         24
# 15               nse         24
# 16            growth         24
# 17          nseindia         23
# 18               may         23
# 19   pharmaceuticals         23
# 20      indianpharma         23
# 21       stockstobuy         22
# 22       sharemarket         21
# 23      stockmarkets         20
# 24               amp         19
# 25            points         19
# 26     stocksinfocus         19
# 27              time         19
# 28              bank         18
# 29             today         18
