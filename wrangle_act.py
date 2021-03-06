#!/usr/bin/env python
# coding: utf-8

# # Project Steps
# 
# The tasks in this project as below:
# 
# 1.Gathering data
# 
# 2.Assessing data
# 
# 3.Cleaning data
# 
# 4.Storing, analyze and report the data 

# # Importing the required libraries

# In[2]:


# Importing important libraries
import numpy as np
import pandas as pd
import requests
import tweepy
import re
import os
import json
import time
import warnings
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Gathering Data

# # Twitter Data

# In[3]:


# reading csv file as a DataFrame
twitter_arc = pd.read_csv('twitter-archive-enhanced.csv')


# # Handling Image predictions File

# In[4]:


url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)
with open('image-predictions.tsv', mode ='wb') as file:
    file.write(response.content)

# Reading the image predictions file as a dataframe
image_prediction = pd.read_csv('image-predictions.tsv', sep='\t' )


# # Handling The tweets

# In[5]:


'''import tweepy
from tweepy import OAuthHandler
import json
from timeit import default_timer as timer

# Query Twitter API for each tweet in the Twitter archive and save JSON in a text file
# These are hidden to comply with Twitter's API terms and conditions
consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# NOTE TO STUDENT WITH MOBILE VERIFICATION ISSUES:
# df_1 is a DataFrame with the twitter_archive_enhanced.csv file. You may have to
# change line 17 to match the name of your DataFrame with twitter_archive_enhanced.csv
# NOTE TO REVIEWER: this student had mobile verification issues so the following
# Twitter API code was sent to this student from a Udacity instructor
# Tweet IDs for which to gather additional data via Twitter's API
tweet_ids = twitter_arc.tweet_id.values
len(tweet_ids)

# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'w') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)'''


# In[6]:


# Creating dataframe by reading the file line by line 
tweets_data = []
with open('tweet_json.txt') as file:
    for line in file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

# Reading the tweets data as a dataframe with the required columns
api_df = pd.DataFrame(tweets_data, columns=list(tweets_data[0].keys()))


# In[7]:


# Choosing only the necessary columns
json_df = api_df[['id', 'retweet_count', 'favorite_count']]
 
json_df.head()


# # Assessing Data

# Checking the dataset for data quality issues and tidiness issues

# Visual Inspection of twitter archive dataframe:

# In[8]:


twitter_arc


# And here is Programmatic Inspection of twitter archive

# In[9]:


twitter_arc.info()


# In[10]:


twitter_arc.describe()


# In[11]:


#check for duplicated values
sum(twitter_arc.duplicated())


# In[12]:


twitter_arc['source'].value_counts()


# In[13]:


twitter_arc['rating_denominator'].value_counts()


# In[14]:


# Checking for retweets
len(twitter_arc[twitter_arc.retweeted_status_id.isnull() == False])


# In[15]:


# Sorting by values of rating_denominator 
twitter_arc.rating_denominator.value_counts().sort_index()


# In[16]:


# Sorting by values of rating_numerator 
twitter_arc.rating_numerator.value_counts().sort_index()


# # Notes after assessing data

# quality:
# 
#     - ID variables are sometimes numeric
#     - [in_reply_to] and [retweeted_status] variables are numeric
#     - there are retweets in the data 
#     - [timestamp] and [retweeted_status_timestamp] are not datetime objects
#     - some column names are confusing
#     - the dog names are not standardized

# Tidiness:
# 
#     - more than one stage is filled for a particular dog
#     - "source" and "expanded_urls" columns contains many informations
#     - columns "doggo", "floofer", "pupper" and "puppo" refer to the same measurement unit, i.e, dog stage

# # Image Predictions¶

# In[17]:


image_prediction


# In[18]:


image_prediction.head()


# In[19]:


image_prediction.tail()


# In[20]:


image_prediction.shape


# In[21]:


image_prediction.info()


# In[22]:


image_prediction.describe()


# In[23]:


sum(image_prediction.duplicated())


# In[24]:


image_prediction.img_num.unique()


# In[25]:


sum(image_prediction.jpg_url.duplicated()==True)


# In[26]:


image_prediction['img_num'].value_counts()


# # Notes after assessing  

# quality:
# 
#     *"tweet_id" and "tweet_id" are numeric and supposed to be categorical 
#     * p1, p2, and p3 contain underscores need to be replaced with spaces

# # JSON file

# In[27]:


json_df


# In[28]:


json_df.head()


# In[29]:


json_df.tail()


# In[30]:


json_df.info()


# In[31]:


json_df.describe()


# In[32]:


sum(json_df.duplicated()==True)


# # Cleaning the data 

# In[33]:


# creating copies of each dataset
twitter_arc_copy = twitter_arc.copy()
image_prediction_1 = image_prediction.copy()
df_json1 = json_df.copy()


# # Twitter Archive 

# # Define

# In[34]:


# replacing all the wrong names 
twitter_arc_copy[twitter_arc_copy.name.str.islower()==True]['name'].unique()


# # Code

# In[35]:


wrong_names = ['such', 'a', 'quite', 'not', 'one', 'incredibly', 'mad', 'an',
              'very', 'just', 'my', 'his', 'actually', 'getting', 'this',
              'unacceptable', 'all', 'old', 'infuriating', 'the', 'by',
              'officially', 'life', 'light', 'space']


# In[36]:


# replacing the wrong names with none
for x in wrong_names:
    twitter_arc_copy.name.replace(x, 'None',inplace=True)


# # Test

# In[37]:


twitter_arc.name.unique()


# In[38]:


# checking the name column one more time to be sure that everuthing is as expected 
twitter_arc_copy.name.unique()


# # Define 

# Assigning the value of 10 to rating_denominator column 

# In[39]:


twitter_arc_copy.rating_denominator= twitter_arc_copy['rating_denominator']=10


# In[40]:


# Testing
twitter_arc_copy.rating_denominator.unique()


# # Define

# Converting timestamp columns to datetime objects

# In[41]:


twitter_arc_copy['timestamp']= pd.to_datetime(twitter_arc_copy['timestamp'])
twitter_arc_copy['retweeted_status_timestamp']= pd.to_datetime(twitter_arc_copy['retweeted_status_timestamp'])


# In[42]:


# Testing
twitter_arc_copy.head()


# # Define

# Creating a column called "dog_stage" by combining "doggo", "floofer", "pupper" and "puppo" columns

# In[43]:


def dog_stage(row):
    stage = []
    if row['doggo'] == 'doggo':
        stage.append('doggo')
    if row['floofer'] == 'floofer':
        stage.append('floofer')
    if row['pupper'] == 'pupper':
        stage.append('pupper')
    if row['puppo'] == 'puppo':
        stage.append('puppo')
    
    if not stage:
        return "None"
    else:
        return ','.join(stage)
    
twitter_arc_copy['dog_stage'] = twitter_arc_copy.apply(lambda row: dog_stage(row), axis=1)


# In[44]:


# dropping 'doggo', 'floofer', 'pupper', 'puppo' columns
twitter_arc_copy.drop(["doggo", "floofer", "pupper", "puppo"], axis=1, inplace=True)


# In[45]:


# Testing
twitter_arc_copy.head()


# In[46]:


twitter_arc_copy.shape


# In[47]:


twitter_arc_copy['dog_stage'].value_counts()


# # Image Predictions

# # Define

# Capitalize the first letter of names of p1,p2,p3 columns

# In[48]:


image_prediction_1.p1= image_prediction_1.p1.str.capitalize()
image_prediction_1.p2= image_prediction_1.p2.str.capitalize()
image_prediction_1.p3= image_prediction_1.p3.str.capitalize()


# In[49]:


# Testing
image_prediction_1.head()


# In[50]:


image_prediction.head()


# # Define

# Replace _ and - with white space in names in p1,p2 and p3 columns 

# In[51]:


image_prediction_1.p1= image_prediction_1.p1.str.replace('_',' ')
image_prediction_1.p2= image_prediction_1.p2.str.replace('_',' ')
image_prediction_1.p3= image_prediction_1.p3.str.replace('_',' ')


# In[52]:


# Testing 
image_prediction_1.head()


# # JSON file 

# # Define 

# set the type of tweet_id column to string 

# In[53]:


df_json1['id'] = df_json1['id'].astype('str')


# In[54]:


# Testing 
df_json1.info()


# # Define
# 
# order the columns
# 

# In[55]:


df_json1 = df_json1[['id', 'favorite_count', 'retweet_count']]


# In[56]:


# Testing 
df_json1.head()


# # Define
# 
# Renaming the column names to merge the two tables easily.

# In[57]:


df_json1.rename(columns={ 'id': 'tweet_id', 'favorite_count': 'likes', 'retweet_count': 'retweets'}, inplace=True)


# # Testing

# In[58]:


df_json1.head()


# In[59]:


df_json1.describe()


# # Define
# 
# change tweet_id type from number to string for all three data sets.
# 

# In[60]:


twitter_arc_copy['tweet_id'] = twitter_arc_copy['tweet_id'].astype('str')
image_prediction_1['tweet_id'] = image_prediction_1['tweet_id'].astype('str')
df_json1['tweet_id'] = df_json1['tweet_id'].astype('str')


# In[61]:


# Testing 
type(twitter_arc_copy['tweet_id'].iloc[0])
type(image_prediction_1['tweet_id'].iloc[0])
type(df_json1['tweet_id'].iloc[0])


# # Merging Datasets

# # Define
# 
# Combine df_json1 and image_prediction1 to twitter_arc_copy

# In[62]:


df_merge = pd.merge(twitter_arc_copy, image_prediction_1, on=['tweet_id'], how='inner')
df_merge = pd.merge(df_merge, df_json1, on = 'tweet_id', how = 'inner' )


# In[63]:


df_merge.head()


# In[64]:


# Calulating the value of 'rating'
df_merge['rating'] = df_merge['rating_numerator'] / df_merge['rating_denominator']


# In[65]:


df_merge.head()


# In[66]:


# Renaming columns names
df_merge.rename(columns={'rating_numerator': 'numerator', 
                        'rating_denominator': 'denominator'}, inplace=True)

# Drop unrequired columns
df_merge.drop(['retweeted_status_id','retweeted_status_timestamp', 'retweeted_status_user_id' ], axis=1, inplace=True)

pd.set_option('display.max_columns', None)


# In[67]:


df_merge.head()


# # Storing Data

# In[68]:


df_merge.to_csv('twitter_archive_new.csv', encoding='utf-8',index=False)


# In[69]:


wrangled_df= pd.read_csv('twitter_archive_new.csv')


# In[70]:


# Testing
wrangled_df.head()


# In[71]:


wrangled_df.describe()


# In[72]:


wrangled_df.info()


# # Analyze and Visualize

# In[73]:


# Set style 
plt.style.use('ggplot')


# In[74]:


# Convert columns to appropriate data types 
# set timestamp as index 

wrangled_df['tweet_id'] = wrangled_df['tweet_id'].astype(object)
wrangled_df['timestamp'] = pd.to_datetime(wrangled_df.timestamp)
wrangled_df['dog_stage'] = wrangled_df['dog_stage'].astype('category')
wrangled_df.set_index('timestamp', inplace=True)


# In[75]:


# Testing 
wrangled_df.info()


# In[76]:


wrangled_df.head()


# In[77]:


wrangled_df.describe()


# In[78]:


# Checking the relationship between retweets and likes 

wrangled_df.plot(kind='scatter',x='retweets',y='likes', alpha = 0.5, figsize=(6,6), color='blue')
plt.xlabel('Likes')
plt.ylabel('Retweets')
plt.title('Relation between retweets and likes')


# From the plot above we can see that when likes increse retweets increase too 

# In[79]:


# Plotting the relation showing the best fit line 
lm = sns.lmplot(x="likes", 
           y="retweets", 
           data=df_json1,
           size = 6,
           aspect=1.3,
           scatter_kws={'alpha':1/6})
axes = lm.axes
axes[0,0].set_xlim(0,)
axes[0,0].set_ylim(0,)
plt.title('Retweet vs Likes')
plt.xlabel('RETWEET')
plt.ylabel('LIKES')


# In[80]:


# this is the list of dogs for rating > 10
wrangled_df.loc[wrangled_df['rating'] > 10]


# In[81]:


# Checking the relationship between dog stage and number of retweets 
plt.figure(figsize=(10,5))
g = sns.boxplot(x='dog_stage',y='retweets',data= wrangled_df,palette='spring')
g.axes.set_title('Relationship between dog stage and no. of retweets', fontsize=15);
g.set_xticklabels(g.get_xticklabels(), rotation=45)


# The plot above shows that most dogs are in Puppo category and the highest retweets are about doggo category

# In[82]:


# Using this function with the help of a github repo and stackoverflow articles 

def remove_link(x):
    '''
        This function is to remove links and apply it to get cleaner texts. 
        Remove http:// and followed links for better cloud words.
    '''
    http_pos = x.find("http")
    # If no link, retain row
    if http_pos == -1:
        x = x
    else:
        # Remove space before link to end
        x = x[:http_pos - 1]
    return x


# In[83]:


wrangled_df.text = wrangled_df.text.apply(remove_link)


# In[84]:


#text in word cloud

text1 = " ".join(review for review in wrangled_df.text) 
# Create list of stopwords
stopwords = set(STOPWORDS)
stopwords.update(["puppy", "Bruno", "dog", "good", "boy", 'cool'])

# create a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text1)

# Display image:

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[93]:


# Checking the relationship between dog stage and number of retweets

plt.figure(figsize=(10,5))
wrangled_df.plot(y ='numerator', ylim=[0,16], style = '*', alpha = .5, color = 'green')
plt.title('Rating plot over period of time')
plt.xlabel('Date')
plt.ylabel('Rating')


# We can conclude that numerator rating of 12 is the most common over a certain period of time.
