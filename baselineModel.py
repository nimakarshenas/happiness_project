
import numpy as np
import pandas as pd
from scipy.special import softmax


class BaselineModel:
    def __init__(self, ffm_vector, ffm_data, n_nearest_neighbours=0, update_weight=1,
                 activities_data=("file", "activity_embeddings.json"), 
                 quotes_data=("file", "quote_embeddings.json")):
        """
        activities ("file"/"df", "path/pd.DataFrame"): pass in a tuple specifying whether
            the activity embeddings are a file path or a pandas dataframe object.
        quotes ("file"/"df", "filepath"/pd.DataFrame)    : pass in a tuple specifying whether
            the quotes embeddings are a file path or a pandas dataframe object.
        """

        
        #----------------- LOAD ACTIVITIES -----------------------
        if activities_data[0] == "file":
           self.df_activities = pd.read_json(activities_data[1])
           self.activity_path = activities_data[1] 
        elif activities_data[0] == "df":
           self.df_activities = activities_data[1]
           self.activity_path = "" 
        else: 
            raise ValueError('activities must be a tuple: ("file"/"df", "filepath"/pd.DataFrame)')
        #----------------- LOAD QUOTES -----------------------
        if quotes_data[0] == "file":
           self.df_quotes = pd.read_json(quotes_data[1])
           self.quote_path = quotes_data[1]
        elif quotes_data[0] == "df":
           self.df_quotes = quotes_data[1]
           self.quote_path = "" 
        else: 
            raise ValueError('quotes must be a tuple: ("file"/"df", "filepath"/pd.DataFrame)')
        #-----------------  -----------------------

        self.update_weight = update_weight
        self.n_nearest_neighbours = n_nearest_neighbours
        self.prev_activity_index = 0
        self.quote_index = 0
        self.current_quote = ""
        self.ffm_vector = ffm_vector
        self.ffm_data = ffm_data
        self.prob_weights = np.zeros((1, ffm_data.shape[0]))
        
        self.current_activity = self.make_initial_suggestion(self.n_nearest_neighbours)
        

    ## ------------ 
    # CLASS METHODS 
    # --------------------

    # make suggestions
    def make_initial_suggestion(self, n_nearest_neighbours=0):
        for j in range(self.ffm_data.shape[0]):
            self.prob_weights[0, j] = np.sum(np.multiply(self.ffm_data[j, :], self.ffm_vector))

        if n_nearest_neighbours:
            relevant_weights = np.sort(self.prob_weights[0])[-n_nearest_neighbours:]
            indexes = np.argsort(self.prob_weights[0])[-n_nearest_neighbours:]
        else:
            relevant_weights = self.prob_weights[0]
            indexes = np.arange(self.ffm_data.shape[0])

        probabilities = softmax(relevant_weights)
        new_index = np.random.choice(indexes, p=probabilities)
        recommended_action = self.df_activities["Activity"][new_index]
        self.activity_index = new_index
        return recommended_action
    
    def make_suggestion(self, n_nearest_neighbours=0):

        if n_nearest_neighbours:
            neighbour_indexes = np.asarray(sorted(range(len(self.df_activities["ProbabilityWeights"][self.activity_index])), 
                                     key=lambda x: self.df_activities["ProbabilityWeights"][self.activity_index][x])[-n_nearest_neighbours:])
        else:
            neighbour_indexes = np.arange(0, len(self.df_activities["ProbabilityWeights"][self.activity_index]))   
        
        # return list of probabilities for each of the n neighbours
        probabilities = softmax(np.array([self.df_activities["ProbabilityWeights"][self.activity_index][i] for i in neighbour_indexes.tolist()]))
        
        # sample from that probability distribution
        new_index = np.random.choice(neighbour_indexes, p=probabilities)

        # get associated activity for the sampled index
        recommended_activity = self.df_activities["Activity"][new_index]
        self.prev_activity_index = self.activity_index
        self.activity_index = new_index
        self.current_activity = recommended_activity
        return recommended_activity, self.make_quote_suggestion(self.df_quotes, new_index, n_nearest_neighbours=n_nearest_neighbours)
    
    def make_quote_suggestion(self, df, index, n_nearest_neighbours=0):
        if n_nearest_neighbours:
            neighbour_indexes = np.asarray(sorted(range(len(df["ProbabilityWeights"][index])), 
                                     key=lambda x: df["ProbabilityWeights"][index][x])[-n_nearest_neighbours:])
        else:
            neighbour_indexes = np.arange(0, len(df["ProbabilityWeights"][index]))   
        
        probabilities = softmax(np.array([df["ProbabilityWeights"][index][i] for i in neighbour_indexes.tolist()]))
        new_index = np.random.choice(neighbour_indexes, p=probabilities)
        self.quote_index = new_index
        recommended_quote = df["Quote"][new_index]
        return recommended_quote
    
    # Update Probability Weighhts
    def update_prob_weights(self, activity_reward, quote_reward, export=False):
        if self.df_activities["ProbabilityWeights"][self.prev_activity_index][self.activity_index] + activity_reward*self.update_weight > 0:
            self.df_activities["ProbabilityWeights"][self.prev_activity_index][self.activity_index] =  self.df_activities["ProbabilityWeights"][self.prev_activity_index][self.activity_index] + activity_reward*self.update_weight
        if self.df_activities["ProbabilityWeights"][self.prev_activity_index][self.quote_index] + quote_reward*self.update_weight > 0:
            self.df_activities["ProbabilityWeights"][self.prev_activity_index][self.quote_index] =  self.df_activities["ProbabilityWeights"][self.prev_activity_index][self.quote_index] + quote_reward*self.update_weight
        if export:
            if self.activity_path:
                self.df_activities.to_json(self.activity_path)
            else:
                self.df_activities.to_json("activity_embeddings.json") 
            if self.quote_path:
                self.df_activities.to_json(self.quote_path)
            else:
                self.df_activities.to_json("activity_embeddings.json") 
    
    