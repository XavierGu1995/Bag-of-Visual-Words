import numpy as np
from collections import deque


def nearest_neighbour(data, code_book):
    distance = 0
    data_code = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        min_dist = np.inf
        index = 0

        for j in range(code_book.shape[0]):
            j_dist = np.linalg.norm(data[i] - code_book[j])

            if j_dist < min_dist:
                min_dist = j_dist
                index = j

        data_code[i] = index
        distance = distance + min_dist

    return data_code, distance / data.shape[0]


def update_cluster_means(data, data_code, k):
    code_book = np.zeros((k, data.shape[1]))
    cluster_num = np.zeros(k)

    for i in range(k):
        for j in range(data.shape[0]):
            if data_code[j] == i:
                code_book[i] = code_book[i] + data[j]
                cluster_num[i] = cluster_num[i] + 1

    for i in range(k):
        code_book[i] = code_book[i] / cluster_num[i]

    return code_book


def _kmeans(data, guess, thresh):
    code_book = np.asarray(guess)
    diff = np.inf
    prev_avg_dists = deque([diff], maxlen=2)

    while diff > thresh:
        # compute membership and distances between data and code_book
        data_code, distance = nearest_neighbour(data, code_book)
        prev_avg_dists.append(distance)

        # recalc code_book as centroids of associated obs
        code_book = update_cluster_means(data, data_code, code_book.shape[0])

        diff = prev_avg_dists[0] - prev_avg_dists[1]
        print(diff)

    return code_book, prev_avg_dists[1]


def kmeans(data, k):
    thresh = 1e-5
    guess = data[np.random.choice(data.shape[0], size=k, replace=False)]
    book = _kmeans(data, guess, thresh)
    return book
