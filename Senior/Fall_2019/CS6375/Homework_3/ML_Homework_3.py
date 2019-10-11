#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import pandas as pd
import numpy  as np
import re
import copy
import sys
import datetime
import time

def update_progress(progress, seconds):
    time = str(datetime.timedelta(seconds=seconds))
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "\nHalt...\r\n"
    if progress >= 1:
        progress = 1
        status = "\nDone...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}%, Time Taken: {2}".format( "#"*block + "-"*(barLength-block), progress*100, time)
    sys.stdout.write(text)
    sys.stdout.flush()


# In[2]:


netflix_training = open("TrainingRatings.txt").read()
netflix_testing = open("TestingRatings.txt").read()


# In[4]:


print "Reading Datasets..."
# Tranform text files into DataFrames
netflix_training_df = pd.DataFrame([i.split(',',2) for i in re.sub('[\r]', '', netflix_training).split('\n')[:-1]])
netflix_testing_df  = pd.DataFrame([i.split(',',2) for i in re.sub('[\r]', '', netflix_testing).split('\n')[:-1]])
netflix_testing_df.columns = netflix_training_df.columns = ['MovieID', 'UserID', 'Rating']


# In[5]:


# Transform all relevant string values to int
netflix_testing_df['MovieID'] = netflix_testing_df['MovieID'].astype(int)
netflix_testing_df['UserID'] = netflix_testing_df['UserID'].astype(int)
netflix_testing_df['Rating'] = netflix_testing_df['Rating'].astype(np.float64)

netflix_training_df['MovieID'] = netflix_training_df['MovieID'].astype(int)
netflix_training_df['UserID'] = netflix_training_df['UserID'].astype(int)
netflix_training_df['Rating'] = netflix_training_df['Rating'].astype(np.float64)


# In[6]:


class Collaborative_Model:
    user_ids  = None
    movie_ids = None
    ratings   = None
    all_users_avg  = None
    all_users_vote = None
    all_users_movies = None
    
    def __init__(self, train):
        self.Prepare_Data(train)
    
    def Prepare_Data(self, train):
        self.user_ids  = train['UserID'].values
        self.movie_ids = np.array(train['MovieID'].values)
        self.ratings   = train['Rating'].values
        unique_users = np.unique(self.user_ids)
        self.ui = {user_id : np.argwhere(user_id == self.user_ids).flatten()
                   for user_id in unique_users}
        self.all_users_avg = dict(train.groupby('UserID')['Rating'].mean())
        self.all_users_vote = {user_id :
                               {movie_id : self.ratings[self.ui[user_id]]
                                [np.argwhere(self.movie_ids[self.ui[user_id]] ==movie_id)][0][0] 
                                for movie_id in self.movie_ids[self.ui[user_id]]}
                               for user_id in unique_users}
        self.all_users_movies = dict(train.groupby('UserID')['MovieID'].apply(lambda x: set(x.values)))
        
    def Predict_Data(self, test):
        t0 = time.time()
        length = len(test)
        return [self.Predict_Instance(time.time() - t0, (i + 1) / length, t)
                for i, t in enumerate(test.values)]
    
    def Predict_Instance(self, seconds, progress, instance):
        update_progress(round(progress, 5), seconds)
        return self.Predict_User_Vote(instance[1], instance[0])

    #Algorithm:
    #First find the users average and what movie ratings they've made
    #Now find relevant users that have voted on the target movie, 
    # then filter out the ones that are actually related to the user
    #Use these users to find the coefficient relationship between the users and the target user
    #Calculate the vote on movie minus average vote
    #Set k to 1 / absolute sum of coefficients. k is 0 when there are no useful ratings in the data set
    #Retrun user avg + k * sum(coefficients * vote difference)
    def Predict_User_Vote(self, tuid, tmi):
        user_avg   = self.all_users_avg[tuid]
        user_votes = self.all_users_vote[tuid]
        users_votes = self.all_users_vote
        users_avg   = self.all_users_avg
        users = np.unique(np.array([self.user_ids[i] for i in np.argwhere(self.movie_ids == tmi).flatten()]))
        common_users = [x for x in [[user_id, self.Get_Common_Movies(tuid, user_id)] for user_id in users]
                        if x[1] != set()]
        coefficients = np.array([self.Pearson_Coefficient(user_avg, users_avg[user[0]],
                                                          user_votes,users_votes[user[0]],
                                                          user[1])
                                 for user in common_users])
        voting_diff = np.array([users_votes[user[0]][tmi] - users_avg[user_id] for user in common_users])
        absolute_sum = np.sum(np.absolute(coefficients))
        if (absolute_sum != 0):
            k = 1 / absolute_sum
        else:
            return user_avg
        weight_sum = np.sum(coefficients * voting_diff)
        return user_avg + k * weight_sum

    def Get_Common_Movies(self, tu, ou):
        return self.all_users_movies[tu] & self.all_users_movies[ou]

    def Pearson_Coefficient(self, user_avg, other_user_avg, user_votes, other_user_votes, common_movies):
        A = np.array([user_votes[common_movie] for common_movie in common_movies]) - user_avg
        B = np.array([other_user_votes[common_movie] for common_movie in common_movies]) - other_user_avg
        numerator   = np.sum(A * B)
        denominator = np.sqrt(np.sum(A**2) * np.sum(B**2))
        if (denominator == 0):
            return 0
        return numerator / denominator


# In[7]:


def Mean_Absolute_Error(predictions, target):
    return np.sum(np.absolute(predictions - target)) / len(predictions)

def Root_Mean_Square_Error(predictions, target):
    return np.sqrt( np.sum((predictions - target) ** 2) / len(predictions) )


# In[8]:


print "Preparing data into optimal data structures..."
predictor = Collaborative_Model(netflix_training_df)


# In[9]:


predictions = predictor.Predict_Data(netflix_testing_df)


# In[ ]:


print Root_Mean_Square_Error(predictions, netflix_testing_df['Rating'])


# In[ ]:


print Mean_Absolute_Error(predictions, netflix_testing_df['Rating'])

