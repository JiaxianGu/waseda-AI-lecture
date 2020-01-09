class KMeans:
    # K: the number of clusters
    def __init__(self, K):
        self.K = K

        # The features of centroids; set when `fit` is called
        self.cluster_centers_ = None

    # fit: make clusters
    # X: data points to cluster -- np.ndarray
    #      (shape: [# of data points, # of features])
    def fit(self, X):
        # TODO: IMPLEMENT ME
        #   Store the feature ndarray of centroids in `self.cluster_centers_`
        #   The shape of `self.cluster_centers_` has to be
        #     [self.K, # of features]
        pass

    # predict: Predict the cluster indices of input data points
    # X: data points predicted -- np.ndarray
    #      (shape: [# of data points, # of features])
    # Return an ndarray with shape [# of data points] where
    #   each element is an integer from 0 to self.K-1
    def predict(self, X):
        # TODO: IMPLEMENT ME
        pass


# check this is a main file
if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.metrics import mean_squared_error
    np.random.set_state(0)

    K = 3
    iris_dataset = load_iris()
    kmeans = KMeans(K)
    kmeans.fit(iris_dataset.data)
    predict = kmeans.predict(iris_dataset.data)
    for k in range(K):
        indices = np.where(predict == k)
        features = iris_dataset.data[indices]
        MSE = mean_squared_error(
            np.tile(kmeans.cluster_centers_[k], (features.shape[0], 1)),
            features)
        print('Cluster', k, 'MSE', MSE)
