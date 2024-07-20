from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from EDA import EDA
import warnings
warnings.filterwarnings('ignore')


def elbow(data):
    # Distortion = 1/n * Σ(distance(point, centroid)^2)
    # Inertia = Σ(distance(point, centroid)^2)
    # Find optimum number of cluster
    wcss = []  # SUM OF SQUARED ERROR
    for k in range(1, 11):
        classifier = KMeans(n_clusters=k, random_state=42)
        classifier.fit(data)
        wcss.append(classifier.inertia_)

        # or

        # wcss.append(sum(np.min(cdist(data, classifier.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    for key, val in dict(enumerate(wcss)).items():
        print(f'{key + 1} : {val}')
    plot_elbow(wcss)


def plot_elbow(wcss):
    sns.set_style("whitegrid")
    diagram = sns.lineplot(x=range(1, 11), y=wcss)
    diagram.set(xlabel="Number of clusters (k)", ylabel="Within-Cluster Sum of Square", title='Elbow Method')
    plt.show()


if __name__ == "__main__":
    dataset = pd.read_csv('./facebook/Live.csv')
    print(dataset)
    print(dataset.info())
    print(dataset.columns)
    print(dataset.shape)
    print(dataset.head())
    print(dataset.tail())
    print(dataset.isnull().sum())
    print((dataset.isnull().sum()/(len(dataset)))*100)
    print(dataset.describe())

    data_EDA = EDA(dataset)
    data_EDA.clean_dataset()
    print(data_EDA.get_data())
    print(data_EDA.get_data().info())

    # data_EDA.remove_outliers()
    # print(data_EDA.get_data())
    # print(data_EDA.get_data().info())

    data_EDA.scale_features()
    print(data_EDA.get_data())

    # apply kmeans clustering
    elbow(data_EDA.get_data())

    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
    kmeans.fit(data_EDA.get_data())
    print(kmeans.cluster_centers_)
    print(kmeans.inertia_)

    prediction = kmeans.fit_predict(data_EDA.get_data())
    print(prediction)
    print(*Counter(prediction))

    # plot clusters for features 4,5
    X = data_EDA.get_data().iloc[:, [4, 5]].values
    plt.scatter(X[prediction == 0, 0], X[prediction == 0, 1], s=60, c='red', label='Cluster1')
    plt.scatter(X[prediction == 1, 0], X[prediction == 1, 1], s=60, c='blue', label='Cluster2')
    plt.scatter(X[prediction == 2, 0], X[prediction == 2, 1], s=60, c='green', label='Cluster3')
    plt.scatter(X[prediction == 3, 0], X[prediction == 3, 1], s=60, c='violet', label='Cluster4')
    # plt.scatter(X[prediction == 4, 0], X[prediction == 4, 1], s=60, c='yellow', label='Cluster5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.show()
