import heapq
import math

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

""" Non-personalized """


def get_k_recommended(rating, k):
    # load data
    metadata = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
    # find avg rating for any book_id
    books_avg_rating = pd.Series(rating.groupby("book_id")['rating'].mean(), name='avg')
    # merge according 'book_id' with save data from left table
    metadata = metadata.merge(books_avg_rating, on='book_id', how='left')
    # find num of vote for any book_id
    books_count = pd.Series(rating.groupby("book_id")['rating'].count(), name='num_vote')
    # merge according 'book_id' with save data from left table
    metadata = metadata.merge(books_count, on='book_id', how='left')
    # Calculate mean of vote average column
    C = metadata['avg'].mean()
    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['num_vote'].quantile(0.9)
    # Make a copy of the metadata, then select rows with at least m votes
    q_books = metadata.copy().loc[metadata['num_vote'] >= m]

    # weighted_rating function
    def weighted_rating(x, m=m, C=C):
        v = x['num_vote']
        R = x['avg']
        return ((v / (v + m)) * R) + ((m / (m + v)) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    # Sort movies based on score calculated above
    q_books = q_books.sort_values('score', ascending=False)
    # print k recommended books
    result = q_books[['book_id', 'title', 'score']].head(k)
    print(result)
    return result


def get_simply_recommendation(k):
    # load data
    rating = pd.read_csv('ratings.csv', low_memory=False)
    # get k recommended books
    return get_k_recommended(rating, k)


def get_simply_place_recommendation(place, k):
    # load data
    rating = pd.read_csv('ratings.csv', low_memory=False)
    users = pd.read_csv('users.csv', low_memory=False)
    # filtering rating by users place:
    # Make a copy of users dataframe, then select rows with the given place in the location column
    users = users.copy().loc[users['location'] == place]
    # Make a copy of rating dataframe, then select rows whose "user id" is in the modified users dataframe above.
    rating = rating.copy().loc[rating['user_id'].isin(users['user_id'])]
    # get k recommended books
    return get_k_recommended(rating, k)


def get_simply_age_recommendation(age, k):
    # load data
    rating = pd.read_csv('ratings.csv', low_memory=False)
    users = pd.read_csv('users.csv', low_memory=False)
    # filtering rating by users age
    x1 = age - (age % 10) + 1
    y0 = age - (age % 10) + 10
    users = users.copy().loc[users['age'] <= y0]
    users = users.copy().loc[users['age'] >= x1]
    rating = rating.copy().loc[rating['user_id'].isin(users['user_id'])]
    # get k recommended books
    return get_k_recommended(rating, k)


""" Collaborative filtering """


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


book_ids_list = ""  # AN ARRAY THAT WILL SAVE THE ORIGINAL BOOK IDS
data_matrix = ""  # A MATRIX THAT WILL SAVE THE RATING OF EACH USER FOR EACH BOOK


def build_CF_prediction_matrix(sim):
    """load data"""
    global book_ids_list, data_matrix
    ratings = pd.read_csv('ratings.csv', low_memory=False)

    """ Create normalized rating matrix """
    # calculate the number of unique users and movies.
    n_users = ratings.user_id.unique().shape[0]
    n_items = ratings.book_id.unique().shape[0]

    # used to convert(normalize) book IDs to range 0 to 4999(the list indices).
    book_ids_list = list(ratings.book_id.unique())

    # create ranking table - that table is sparse
    data_matrix = np.empty((n_users, n_items))
    data_matrix[:] = np.nan
    for line in ratings.itertuples():
        user = line[1] - 1
        book_id = line[2]
        book_norm_id = book_ids_list.index(book_id)
        rating = line[3]
        data_matrix[user, book_norm_id] = rating

    # calc mean
    mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)

    ratings_diff = (data_matrix - mean_user_rating)
    # replace nan -> 0
    ratings_diff[np.isnan(ratings_diff)] = 0

    """ calculate user x user similarity matrix """
    if sim == "jaccard":
        user_similarity = 1 - pairwise_distances(np.array(ratings_diff, dtype=bool), metric=sim)
    else:
        user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)

    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    k = 10
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])

    """ Generate predicted ratings matrix """
    # since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    # pred.round(2)
    return pred  # return the prediction matrix


def get_CF_recommendation(user_id, k, pred=None):
    global book_ids_list, data_matrix
    user = user_id - 1
    sim = 'cosine'
    if pred is None:  # if we didnt got a pred matrix we will make one with cosine
        # data_matrix is the normalized rating matrix(that the users already rated)
        pred = build_CF_prediction_matrix(sim)
    predicted_ratings_row = pred[user]
    data_matrix_row = data_matrix[user]
    items = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
    # convert book IDs back
    predicted_ratings_row_original_ids = np.zeros(max(book_ids_list))
    for book_id in range(len(predicted_ratings_row)):
        predicted_ratings_row_original_ids[book_ids_list[book_id] - 1] = predicted_ratings_row[book_id]

    user_already_rated = ~np.isnan(data_matrix_row)
    for i in range(len(user_already_rated)):
        if user_already_rated[i]:
            predicted_ratings_row_original_ids[book_ids_list[i] - 1] = 0

    idx = np.argsort(-predicted_ratings_row_original_ids)
    # IDs of k top (predicted) rated books for the current user
    sim_scores = idx[0:k]
    # Return top k movies
    title = pd.Series(name="title", dtype=str)
    book_id = pd.Series(name="book_id", dtype=int)
    pd.options.display.max_colwidth = 1000
    for score in sim_scores:
        score += 1  # increase the score by 1
        row = items.loc[items['book_id'] == score]  # get the whole row
        # insert the book id and the title
        title = title.append(row['title'])
        book_id = book_id.append(row['book_id'])

    result = pd.concat([title, book_id], axis=1)
    print(result)
    return result


""" Content-based filtering """

metadata = ""  # THIS WILL SAVE THE LOADING OF THE BOOKS TABLE


# this function builds content similarity matrix
def build_contact_sim_metrix():
    global metadata
    metadata = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")  # load the books table
    book_tags = pd.read_csv('books_tags.csv', low_memory=False)  # load the books_tags table
    tags = pd.read_csv('tags.csv', low_memory=False)  # load the tags table

    # convert the goodreads book id to be int
    metadata['goodreads_book_id'] = metadata['goodreads_book_id'].astype(int)
    book_tags['goodreads_book_id'] = book_tags['goodreads_book_id'].astype(int)

    # convert the tag_id to be int
    book_tags['tag_id'] = book_tags['tag_id'].astype(int)
    tags['tag_id'] = tags['tag_id'].astype(int)

    # merge the book tags with the tags according to tag id
    book_tags = book_tags.merge(tags, on='tag_id')

    # merge the metadata with the book tags according the goodreads_book_id
    # do a group by on the goodreads_book_id and create a list from the tag name for each book, then join according the
    # goodreads_book_id column and save the data from the left table
    metadata = metadata.merge(book_tags.groupby('goodreads_book_id')['tag_name'].apply(list), on='goodreads_book_id',
                              how='left')

    # this function remove spaces from the authors names
    def clean_data(x):
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        elif isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            return ''  # remove nan values by ''

    metadata['authors'] = metadata['authors'].apply(clean_data)  # clean and format the authors name
    metadata['language_code'] = metadata['language_code'].apply(clean_data)  # clean the language code
    metadata['tag_name'] = metadata['tag_name'].apply(clean_data)  # clean the tag name

    def create_soup(x):  # chain all the relevant data of the book
        return x['authors'] + ' ' + x['language_code'] + ' ' + str(x['original_title']) + ' ' + str(
            x['title']) + ' ' + ' '.join(x['tag_name'])

    # Create a new soup feature
    metadata['soup'] = metadata.apply(create_soup, axis=1)

    count = CountVectorizer()
    count_matrix = count.fit_transform(metadata['soup'])

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)  # create a cosine similarity matrix

    metadata = metadata.reset_index()  # reset the index
    return cosine_sim2  # return the cosine similarity matrix


def get_contact_recommendation(book_name, k):
    global metadata
    cosine_sim_mat = build_contact_sim_metrix()
    indices = pd.Series(metadata.index, index=metadata['original_title'])

    # Find the first matched for the book that the user wanted
    # check if we more than one value for the same key
    if isinstance(indices[book_name], pd.core.series.Series):
        idx = indices[book_name][0]  # in case of duplicate keys we will take the first
    else:
        idx = indices[book_name]

    # Get the pairwise similarity scores of all books with that movie
    sim_scores = list(enumerate(cosine_sim_mat[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the k most similar books (the first is the book we asked)
    sim_scores = sim_scores[1:k + 1]

    # Get the book indices
    books_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    title_result = metadata['title'].iloc[books_indices]  # take the title
    book_id_result = metadata['book_id'].iloc[books_indices]  # take the book id
    result = pd.concat([book_id_result, title_result], axis=1)
    print(result)
    return result


""" Evaluation metrics """


def precision_k(k):
    global book_ids_list, data_matrix
    test = pd.read_csv('test.csv', low_memory=False)  # load the test table

    # drop the rows which their rating value is lower than 4
    test = test.drop(test[test.rating < 4].index)

    # count the number of books each user rated, and check if its lower than k
    should_drop = test.groupby("user_id")['book_id'].count() < k
    test_copy = test.copy()
    # run on the copy and drop all users who rated less than k books
    for idx in test_copy.index:
        if should_drop[int(test_copy.loc[idx, :]["user_id"])]:
            test = test.drop(idx)

    # run on all possible sims
    sims = ["cosine", "euclidean", "jaccard"]
    results = []
    for sim in sims:
        # calculate the pred matrix according the given sim
        pred = build_CF_prediction_matrix(sim)
        hits_all_users = 0
        users_count = 0
        # run on all the users in test
        for user_id, group in test.groupby('user_id'):
            users_count += 1
            # get the k recommendation for the user
            recommendations = get_CF_recommendation(user_id, k, pred)
            hits_current_user = 0
            # run on all the values of book_id that exist in the recommendations
            for book_id in recommendations["book_id"].values:
                # check if the book id exist in the group
                if int(book_id) in group["book_id"].values:
                    hits_current_user += 1
            # at the end divide the number of hits of the user by k
            hits_all_users += hits_current_user / k
        # at the end divide all the number of hits by all users in test
        sim_res = hits_all_users / users_count
        results.append(sim_res)
    result = dict(zip(sims, results))
    print(result)
    return result


def ARHR(k):
    global book_ids_list, data_matrix
    test = pd.read_csv('test.csv', low_memory=False)  # load the test table

    # drop the rows which their rating value is lower than 4
    test = test.drop(test[test.rating < 4].index)

    # count the number of books each user rated, and check if its lower than k
    should_drop = test.groupby("user_id")['book_id'].count() < k
    test_copy = test.copy()
    # run on the copy and drop all users who rated less than k books
    for idx in test_copy.index:
        if should_drop[int(test_copy.loc[idx, :]["user_id"])]:
            test = test.drop(idx)

    # run on all possible sims
    sims = ["cosine", "euclidean", "jaccard"]
    results = []
    for sim in sims:
        # calculate the pred matrix according the given sim
        pred = build_CF_prediction_matrix(sim)
        hits_all_users = 0
        users_count = 0
        # run on all the users in test
        for user_id, group in test.groupby('user_id'):
            users_count += 1
            # get the k recommendation for the user
            recommendations = get_CF_recommendation(user_id, k, pred)
            hits_current_user = 0
            book_num = 1  # will save the position of the books
            # run on all the values of book_id that exist in the recommendations
            for book_id in recommendations["book_id"].values:
                if int(book_id) in group["book_id"].values:
                    hits_current_user += 1 / book_num  # calculating according the position
                book_num += 1
            hits_all_users += hits_current_user  # sum the positioned hit for all users
        # divide the sum of positioned hit for all users by the number of users
        sim_res = hits_all_users / users_count
        results.append(sim_res)
    result = dict(zip(sims, results))
    print(result)
    return result


def RMSE():
    global book_ids_list
    test = pd.read_csv('test.csv', low_memory=False)  # load the test table
    rows = test["rating"].count()  # take the number of rows in the table
    sims = ["cosine", "euclidean", "jaccard"]
    results = []
    for sim in sims:
        # take the prediction matrix and the original book ids
        pred = build_CF_prediction_matrix(sim)
        differences = 0
        for user_id, group in test.groupby('user_id'):
            user = user_id - 1
            # take the predicted row of the user from the predicted matrix
            predicted_ratings_row = pred[user]
            # convert book IDs back
            predicted_ratings_row_original_ids = np.zeros(max(book_ids_list))
            for book_id in range(len(predicted_ratings_row)):
                predicted_ratings_row_original_ids[book_ids_list[book_id] - 1] = predicted_ratings_row[book_id]

            # run on the book id and rate of each book in the books that the current user in test rated
            for book_id, rating in zip(group["book_id"].values, group["rating"].values):
                # take the predicted rate for the book_id
                predicted_rating = predicted_ratings_row_original_ids[book_id - 1]
                # calculate the difference between the original rate and the predicted rate
                differences += math.pow(predicted_rating - rating, 2)
        # at the end do sqrt on the sum of the differences / number of rows in test table
        sim_res = math.sqrt(differences / rows)
        results.append(sim_res)
    result = dict(zip(sims, results))
    print(result)
    return result


# if __name__ == '__main__':
#     # Part A: Non-personalized
#     get_simply_recommendation(10)
#     get_simply_place_recommendation('Ohio', 10)
#     get_simply_age_recommendation(28, 10)
#
#     # Part B: Collaborative filtering
#     # cosine! cannot change function signatures so in order to change the similarity function you have to change it inside get_CF_recommendation!
#     get_CF_recommendation(1, 10)
#
#     # Part C: Content based filtering
#     get_contact_recommendation('Twilight', 10)
#
#     # Part D: Evaluation metrics
#     precision_k(10)
#     ARHR(10)
#     RMSE()
