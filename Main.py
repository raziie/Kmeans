import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from EDA import EDA
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    dataset = pd.read_csv('./facebook/Live.csv')
    print(dataset)
    print(dataset.info())

    data_EDA = EDA(dataset)
    data_EDA.clean_dataset()
    print(data_EDA.get_data())
    print(data_EDA.get_data().info())

    data_EDA.scale_features()
    print(data_EDA.get_data())

    # apply kmeans clustering

    # X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)
    #
    # fig = plt.figure(0)
    # plt.grid(True)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
