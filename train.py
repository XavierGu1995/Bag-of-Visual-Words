import cv2
import glob
import os
import numpy as np
from kmeans import kmeans


# hyper parameter
K = 32


# find nearest neighbor of feature in code book
def nearest_neighbor(a, code_book):
    min_dist = 10000
    index = 0

    for i in range(code_book.shape[0]):
        if np.linalg.norm(a - code_book[i]) < min_dist:
            min_dist = np.linalg.norm(a - code_book[i])
            index = i

    return index


if __name__ == '__main__':

    # train data
    data_path = "./dataset/train_data"
    img_list = glob.glob(os.path.join(data_path, '*/*.ppm'))

    sift = cv2.xfeatures2d.SIFT_create()    # SIFT operator
    features = np.zeros((1, 128))  # features of all pictures

    for img_name in img_list:
        img = cv2.imread(img_name)
        kp, ft = sift.detectAndCompute(img, None)   # SIFT feature
        features = np.vstack((features, ft))

    # K-Means cluster code_bool
    code_book = kmeans(features, K)

    # storage code book
    np.save("code_book.npy", code_book)

    # visual words table
    code_table = []
    for img_name in img_list:
        img = cv2.imread(img_name)
        kp, feature = sift.detectAndCompute(img, None)
        code = np.zeros(K)  # code of feature

        for ft in feature:
            index = nearest_neighbor(ft, code_book)
            code[index] = code[index] + 1

        code = code / feature.shape[0]   # normalization
        code_table.append(code)

    # storage visual words table
    code_table = np.array(code_table)
    np.save("code_table.npy", code_table)
