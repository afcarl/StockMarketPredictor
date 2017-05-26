import tweepy
from textblob import TextBlob
import numpy as np
from twitter_credentials import *

def polarity(polarities):
    print(polarities)
    mean = np.array(polarities,dtype=np.float32).mean(axis=0)
    if mean > 0: return 1
    else: return 0


def authorize():
    auth = tweepy.OAuthHandler(credentials['ck'],credentials['cs'])
    auth.set_access_token(credentials['at'],credentials['ats'])
    api = tweepy.API(auth)
    return api

def search_tweets(api,search_val):
    polarities = []
    tweets = api.search(search_val,count=100,rpp=100)
    for tweet in tweets:
        blob = TextBlob(tweet.text)
        polarities.append(blob.sentiment.polarity)
    return polarity(polarities)



if __name__ == '__main__':
    api = authorize()
    print(search_tweets(api,'google'))
