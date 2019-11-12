import numpy as np
import os
import glob
import cv2


# hyper parameters
K = 32
N = 8   # num of retrieval pictures
NORM_TYPE = "L1 norm"


def norm(a, b):
    if NORM_TYPE == "L2 norm":
        return np.linalg.norm(a - b)
    else:
        return np.linalg.norm(a - b, ord=1)


# find nearest neighbor of feature in code book
def nearest_neighbor(a, code_book):
    min_dist = 10000
    index = 0

    for i in range(code_book.shape[0]):
        if np.linalg.norm(a - code_book[i]) < min_dist:
            min_dist = np.linalg.norm(a - code_book[i])
            index = i

    return index


def retrieval(code, code_table):
    retrievals = np.zeros(N)
    visited = np.zeros(code_table.shape[0])
    for j in range(N):
        index = 0
        min_dist = np.inf

        for i in range(code_table.shape[0]):
            if norm(code, code_table[i]) < min_dist and visited[i] == 0:
                index = i
                min_dist = norm(code, code_table[i])

        visited[index] = 1
        retrievals[j] = index

    return retrievals


if __name__ == '__main__':

    # train data and test data
    train_data_path = "./dataset/train_data"
    train_img_list = glob.glob(os.path.join(train_data_path, '*/*.ppm'))
    test_data_path = "./dataset/test_data"
    test_img_list = glob.glob(os.path.join(test_data_path, '*.ppm'))

    sift = cv2.xfeatures2d.SIFT_create()    # SIFT operator

    # load code book and visual words table
    code_book = np.load("code_book.npy")
    code_table = np.load("code_table.npy")

    # classify picture
    i = 0
    for img_name in test_img_list:
        img = cv2.imread(img_name)
        kp, feature = sift.detectAndCompute(img, None)
        code = np.zeros(K)

        for ft in feature:
            index = nearest_neighbor(ft, code_book)
            code[index] = code[index] + 1

        code = code / feature.shape[0]  # normalization

        retrievals = retrieval(code, code_table)
        print("picture names: ", img_name, "\tretrievals result: ", retrievals)

        retrievals = retrievals // 5

        hit = 0
        for j in range(N):
            if i == retrievals[j]:
                hit = hit + 1

        print("retrievals: ", hit / N, "\trecall: ", hit / 5)

        i = i + 1

    # precision and recallA
