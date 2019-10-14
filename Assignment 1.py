
# coding: utf-8

# # Assignment 1
#     Paul Kauffman

# ## Background:
#     In this scenerio I work for a local sports marketing firm that wants to know what was tweeted about during the first weekend of college football on September 1st. One of the executives requested that the three closest local schools be the    area of focus; 
#     K-State, KU, and Nebraska. 

# # Connecting to Twitter 
#   

# In[1]:


import pandas as pd
import tweepy
from tweepy import OAuthHandler

#API keys generated from app
consumer_key =  '9pYLWvGny3KDq249QI7z9lM2Q'
consumer_secret = 'kXUcPjmBzDiIA7J6XjlM9CeSalpTvL8qr0fQptFInwfRRsBCqH'
access_token= '3882545175-CO7kqAl1H9IzJu0USlDxVdVRnHxRfLspTAKuJEj'
access_secret= 'siApPytjdl9ibOwB8PDzJPbZwf9RcaloIkYsV8RybdpLQ'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


# ### Check Connection with timeline 

# In[2]:


timeline_tweets = api.home_timeline()
for tweet in timeline_tweets: print(tweet.text)


# # --T1--

# ## First API search 

# #### ~500 tweets including the hashtag (#) "college football"

# In[3]:


results = []

#searching for 500 tweets including the phrase "college football"
for tweet in tweepy.Cursor(api.search, q='#collegefootball').items(500):
    results.append(tweet)

# Verify the number of items returned
print(len(results))


# ### Name columns and create dataframe

# In[4]:


def toDataFrame(tweets):

    DataSet = pd.DataFrame()

    DataSet['tweetText'] = [tweet.text for tweet in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]

    return DataSet

tweet_frame = toDataFrame(results)
print(tweet_frame.shape)


# ##### 500 tweets and 5 column dataframe created

# ### View Tweets

# In[5]:


#set column with to view tweetText
pd.set_option('display.max_colwidth', 200)

tweet_frame.head(5)


# ### First API Search Analysis:
#     While the search was succesful at pulling in 500 tweets with "#collegefootball" the information was not specific to the local schools, and was not specific to the first weekend of college football. The data also included a number of retweets and unrealted information. 

# # Second API Search 
#     focus on local teams and change search to tweets containing the phrase 'college football'

# ### Adding geo tagging to tweepy. Lat/Long for Manhattan, KS with 200km radius search area

# ![ManhattanRadius.PNG](attachment:ManhattanRadius.PNG)Desktop\Rockhurst\BIA 6304-Text Mining\Homework 1\ManhattanRadius.png)

# In[50]:


results = []

for tweet in tweepy.Cursor(api.search, q='college football', geocode= '39.183609,-96.571671,200km').items(500):
    results.append(tweet)

print(len(results))



# In[508]:


def toDataFrame(tweets):

    DataSet = pd.DataFrame()

    DataSet['tweetText'] = [tweet.text for tweet in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]

    return DataSet

tweet_frame = toDataFrame(results)
tweet_frame.head()


# ### Second API Search Analysis:
#     With the geotagging restriction in place the search did pull more tweets within the area desired. The search parameter of tweetings containing the phrases "college football" also seemed to pull tweets from people rather than networks and companies. The search still needs to be pared down to the first college football weekend and retweets also could skew the results if a popular tweet is included. 

# # Third (and final) API Search
#     removing retweets and searching only on 9/1 (first college gameday of the season)

# In[509]:


results = []

#filter:retweets removes tweets containing "RT"
for tweet in tweepy.Cursor(api.search, q='college football-filter:retweets',  geocode= '39.183609,-96.571671,200km',
                          start = '2018-09-01' , until ='2018-09-02').items(500):
    results.append(tweet)

# Verify the number of items returned
print(type(results))
print(len(results))


# In[511]:


def toDataFrame(tweets):

    DataSet = pd.DataFrame()

    DataSet['tweetText'] = [tweet.text for tweet in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]

    return DataSet

tweet_frame = toDataFrame(results)
tweet_frame.head(10)


# ### Third(and final) API Search Analysis:
#     This search finally meets the qualifications needed. The area is restricted to the local area reqested and should not be skewed by retweeted tweets. The timeline is also restricted to the day of the games so that the information is relevant to that timeframe. 

# # --T2--

# # Vectorizers 

# #### Basic Vectorizer (binary = True)

# In[529]:


from sklearn.feature_extraction.text import CountVectorizer
import math

cv1 = CountVectorizer(binary=True)
cv1_chat = cv1.fit_transform(tweet_frame['tweetText'])

print(cv1_chat.shape)

# features 
cv1_features= cv1.get_feature_names()
cv1_features[:10]


# #### Vectorizer #2 (binary=False, using ngrams)
#     To help include common words used together like 'College Football' I use ngrams to show those common 2 word tokens

# In[535]:


cv2 = CountVectorizer(binary=False, ngram_range= (1,2)) 
cv2_chat = cv2.fit_transform(tweet_frame['tweetText'])

print(cv2_chat.shape)

cv2_features= cv2.get_feature_names()
cv2_features[:10]


# #### Vectorizer #3 (binary=True, stop_words)
#     With vectorizer #2 having over 7000 columns the word count needs to be pared down. Stop words will help get rid of the filler words that have little value in this case

# In[537]:


cv3 = CountVectorizer(binary=True, stop_words='english')
cv3_chat = cv3.fit_transform(tweet_frame['tweetText']) 


print(cv3_chat.shape)

cv3_features= cv3.get_feature_names()
cv3_features[:10]


# #### Vectorizer #4 (added ngrams back and min_df)
#     to further reduce the number of tokens I added a min-df parameter. This will show the tokens that are in at least 10% of the corpus 

# In[542]:


cv4 = CountVectorizer(binary=True, stop_words='english', ngram_range=(1,2), min_df=.1)
cv4_chat = cv4.fit_transform(tweet_frame['tweetText']) 


print(cv4_chat.shape)

cv4_features= cv4.get_feature_names()
cv4_features[:10]


# #### Vectorizer #5 (added max_df)
#     the min_df set to .1 removed too many tokens (only 5 were left). Adding max_df at .01 will only show the tokens included in less than 1% of the corpus

# In[548]:


cv5 = CountVectorizer(binary=True, max_df= .01, stop_words='english',ngram_range = (1,2))


cv5_chat = cv5.fit_transform(tweet_frame['tweetText'])

print(cv5_chat.shape)

cv5_features= cv5.get_feature_names()
cv5_features[:10]


# #### Vectorizer #6
#     This is the vectorizer of choice. This uses a combination of max-df, min-df, stop words, and ngram. 68 total features in the vector

# In[549]:


# try changing a few parameters
cv6 = CountVectorizer(binary=True, max_df =.15,min_df= .015, stop_words ='english',ngram_range = (1,2)) #define the transformation
# only asking it to make changes based on document frequency
# not using stop words, but it should still might help eliminate "is"

cv6_chat = cv6.fit_transform(tweet_frame['tweetText']) #apply the transformation

print(type(cv6_chat))
print(cv6_chat.shape)


# ## Feature Count 

# ### Top 25 token count
#     words like love, best, and happy are valuable words 

# In[481]:


names = cv6.get_feature_names()   #create list of feature names
print(type(names), len(names))

count = np.sum(cv6_chat.toarray(), axis = 0) # convert list to array to add up feature counts 
count2 = count.tolist()  # convert numpy array to list

print("") #this is just to add a break in the output
print("We started with", len(names), "and we ended with",len(count2))
print("")

count_df = pd.DataFrame(count2, index = names, columns = ['count']) # create a dataframe from the list
sorted_count = count_df.sort_values(['count'], ascending = False)  #arrange by count instead
print(sorted_count.head(25))


# # Weights

# In[572]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(use_idf=True, norm=None, stop_words ='english', max_df =.15,min_df= .015, ngram_range = (1,2))
tf_chat = tfidf.fit_transform(tweet_frame['tweetText'])


tf_chat_df =pd.DataFrame(tf_chat.toarray(),columns = tfidf.get_feature_names())
tf_chat_df.head(10)


# ## Sort Weights
#     very similar list to the count list with more confusion of how weights are calculated 

# In[573]:


names = tfidf.get_feature_names()   #create list of feature names
count = np.sum(tf_chat.toarray(), axis = 0) # add up feature counts 
count2 = count.tolist() 

count_df = pd.DataFrame(count2, index = names, columns = ['count']) # create a dataframe from the list
sorted_weights = count_df.sort_values('count', ascending = False)
print(sorted_weights.head(25))


# ### Overall, I am happy with the results. The count option does give some context to the audience’s feelings on 9/1. The weights option, while accurate, causes confusion when trying to describe how it is calculated. The count options generates basically the same list with much more clarity.
