import os
import gc
import time
import joblib
import argparse
import pandas as pd
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def dataset_generator():
    # movies dataset
    df_movies = pd.read_csv("Dataset/movies.csv",usecols=['movieId', 'title'])
    # rating dataset
    df_ratings = pd.read_csv("Dataset/ratings.csv",usecols=['userId', 'movieId', 'rating'])
    # Creating a studio Dataset 
    studios = pd.read_csv("Dataset/studios.csv")
    # Renaming columns
    df_movies.rename(columns={'movieId': 'studioId', 'title': 'studioName'}, inplace=True)
    df_ratings.rename(columns={'movieId': 'studioId'}, inplace=True)
    # Overwriting Columns
    df_movies['studioName'][0:124]=studios['Studios'][0:125]
    # Selecting the ratings from the ratings dataset whose studio id is between 0 to 124
    range = [(0, 124)]
    def check_studios(value):
        for a, b in range:
            if a <= value <= b:
                return True
        return False
    df_ratings = df_ratings[ df_ratings['studioId'].apply(check_studios) ]
    df_movies=df_movies[0:124]
    # Saving the new datasets
    df_movies.to_csv("Dataset/usedStudios.csv")
    df_ratings.to_csv("Dataset/usedRatings.csv")


def feature_generator():
    # Loading Datasets 
    df_ratings = pd.read_csv("Dataset/UsedRatings.csv")
    df_studios = pd.read_csv("Dataset/UsedStudios.csv")
    # pivot ratings into studio features
    df_studio_features = df_ratings.pivot(index='studioId',columns='userId',values='rating').fillna(0)
    # convert dataframe of movie features to scipy sparse matrix
    studio_features = csr_matrix(df_studio_features.values)
    joblib.dump(studio_features, 'studio_features.pkl')
    # create mapper from movie title to index
    hashmap = {
        studio: i for i, studio in
        enumerate(list(df_studios.set_index('studioId').loc[df_studio_features.index].studioName)) # noqa
    }
    joblib.dump(hashmap, 'studio_hashmap.pkl')

def model_training():
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    data=joblib.load('studio_features.pkl')
    model_knn.fit(data)
    joblib.dump(model_knn, 'KNN_model.pkl')


def fuzzy_matching(input_studio):
    match_tuple = []
    hashmap=joblib.load('studio_hashmap.pkl')
    # get match
    for studioName, idx in hashmap.items():
        ratio = fuzz.ratio(studioName.lower(), input_studio.lower())
        if ratio >= 60:
            match_tuple.append((studioName, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
    else:
        return match_tuple[0][1]

def inference(input_studio):
    model=joblib.load('KNN_model.pkl')
    hashmap=joblib.load('studio_hashmap.pkl')
    data=joblib.load('studio_features.pkl')
    # get input movie index
    print('You have input studio:', input_studio)
    idx = fuzzy_matching(input_studio)
    # inference
    distances, indices = model.kneighbors(data[idx],n_neighbors=10+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(
            list(
                zip(
                    indices.squeeze().tolist(),
                    distances.squeeze().tolist()
                )
            ),
            key=lambda x: x[1]
        )[:0:-1]
    # return recommendation (movieId, distance)
    return raw_recommends



def make_recommendations(input_studio, n_recommendations=10):
    hashmap=joblib.load('studio_hashmap.pkl')
    # get recommendations
    raw_recommends = inference(input_studio)
    # print results
    reverse_hashmap = {v: k for k, v in hashmap.items()}
    print('Recommendations for {}:'.format(input_studio))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance '
                'of {2}'.format(i+1, reverse_hashmap[idx], dist))

def test_datasetGenerator():
    usedRatings = pd.read_csv("Dataset/UsedRatings.csv")
    usedStudios = pd.read_csv("Dataset/UsedStudios.csv")
    hashmap=joblib.load('studio_hashmap.pkl')
    reverse_hashmap = {v: k for k, v in hashmap.items()}
    test_data = pd.DataFrame(columns=['studioName','recommendationClass'])
    temp_studios=[]
    temp_class=[]
    # get recommendations
    for input_studio in reverse_hashmap:
        raw_recommends = inference(usedStudios['studioName'][input_studio])
        original_rating= usedRatings['rating'][fuzzy_matching(usedStudios['studioName'][input_studio])]
        ratings=[]
        difference=[]
        # This loop gets the rating of all the recommendations being for a studio
        for i, (idx, dist) in enumerate(raw_recommends):
            ratings.append(usedRatings['rating'][idx])
        # This iterates over the ratings of recommendations
        for index in range(0,10):
            # If the rating of the current output and the recommended studio has the difference greater than equal to 0 then 
            # class is 0 otherwise its 1
            diff=abs(original_rating-ratings[index])
            if diff>=1.7:
                difference.append(0)
            else:
                difference.append(1)
        # Final Class for studio recommendations is done on total counts of 0's and 1's  in dataset
        if difference.count(1)>difference.count(0):
            temp_studios.append(usedStudios['studioName'][input_studio])
            temp_class.append(1)
        else:
            temp_studios.append(usedStudios['studioName'][input_studio])
            temp_class.append(0)
    # Saving testing data set
    test_data['studioName']=temp_studios
    test_data['recommendationClass']=temp_class
    test_data.to_csv("Dataset/testDataset.csv")

def model_testing():
    testDataset = pd.read_csv("Dataset/testDataset.csv")
    print(testDataset['recommendationClass'].value_counts(normalize=True) * 100)



if __name__ == "__main__":
    os.system('cls')
    # This function generates the model set used for model training
    #dataset_generator()
    
    # Dataset Size
    usedRatings = pd.read_csv("Dataset/UsedRatings.csv")
    usedStudios = pd.read_csv("Dataset/UsedStudios.csv")
    print("Total Studios: ",len(usedStudios))
    print("Total Ratings: ",len(usedRatings))
    
    # This function is used for creating the customer interaction matri
    #feature_generator()

    # This function is used to create a dataset for testing
    #test_datasetGenerator()

    # This function is used to evaluate the model
    #model_testing()
        
    # This function is used to get indiviual recommendations
    #make_recommendations("Dossaniplus Studio", 10)
    os.system("pause")
    