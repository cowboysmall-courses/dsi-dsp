
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
import matplotlib.pyplot as plt
import itertools

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud

from string import punctuation, digits


# %% 1 - import data and check the head
with open("./data/raw/Tweets 12-6-24.txt") as file:
    tweets = [line.rstrip() for line in file]

data = pd.DataFrame([line for line in tweets if len(line) > 0], columns= ["Tweets"])
data.head()



# %% 1 - 
stop_words    = set(stopwords.words('english')) | set([])
remove_punc   = str.maketrans('', '', punctuation)
remove_digits = str.maketrans('', '', digits)

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet.lower().translate(remove_punc).translate(remove_digits))
    return " ".join([word for word in tokens if word not in stop_words])



# %% 1 - 
data["Cleaned_Tweets"] = data["Tweets"].apply(preprocess_tweet)
data.head()



# %% 1 - 
tweet_words = data["Cleaned_Tweets"].str.cat(sep = " ")



# %% 1 - 
fdist = nltk.FreqDist(tweet_words.split())
fdist.most_common(30)
# [('nifty', 399),
#  ('banknifty', 104),
#  ('stockmarket', 71),
#  ('niftybank', 45),
#  ('stockmarketindia', 44),
#  ('sensex', 43),
#  ('stocks', 38),
#  ('optionstrading', 36),
#  ('bse', 34),
#  ('breakoutstocks', 31),
#  ('trading', 30),
#  ('market', 29),
#  ('india', 26),
#  ('ipm', 25),
#  ('good', 24),
#  ('nse', 24),
#  ('growth', 24),
#  ('nseindia', 23),
#  ('may', 23),
#  ('pharmaceuticals', 23),
#  ('indianpharma', 23),
#  ('stockstobuy', 22),
#  ('sharemarket', 21),
#  ('stockmarkets', 20),
#  ('amp', 19),
#  ('points', 19),
#  ('stocksinfocus', 19),
#  ('time', 19),
#  ('bank', 18),
#  ('today', 18)]



# %% 1 - 
wordcloud = WordCloud(background_color = "white", collocations = False).generate(tweet_words)

plt.figure(figsize = (16, 10))
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()



# %% 1 - 



# %% 1 - 



# %% 1 - 



# %% 1 - 



# %% 1 - 



# %% 1 - 



# %% 1 - 
