# If you implement random sampling of training data points with replacement, the
# number of sampled points may be the same as that of given trainng data points
# (but each point may be sampled multiple times; see
# https://web.ma.utexas.edu/users/parker/sampling/repl.htm)

class RandomForestClassifier:
    def __init__(self, n_trees, n_sampled_features, criterion='entropy'):
        self.n_trees = n_trees
        self.n_sampled_features

        # 'entropy' or 'gini' (parameter to sklearn.tree.DecisionTreeClassifier)
        self.criterion = criterion 

    # fit: train this model on training inputs X and outputs Y
    # X: training inputs -- np.ndarray
    #      (shape: [# of data points, # of features])
    # Y: training outputs -- np.ndarray
    #      (shape: [# of data points])
    def fit(self, X, Y):
        # TODO: IMPLEMENT ME
        pass

    # predict: classify given data points
    # X: inputs to the classifier -- np.ndarray
    #      (shape: [# of data points, # of features])
    def predict(self, X):
        # TODO: IMPLEMENT ME
        pass


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer_dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer_dataset.data,
        cancer_dataset.target,
        stratify=cancer_dataset.target,
        random_state=0)

    # Initiliazation may change depending on implemented randomness
    rf = RandomForestClassifier(10, criterion='entropy')
    rf.fit(X_train, y_train)
    X_test_predict = rf.predict(X_test)
    accuracy = np.sum(y_test == X_test_predict) / y_test.shape[0]
    assert(accuracy > 0.9)
    print('acc:', accuracy)

    # TODO: Add code to train and evaluate a decision tree using
    #   X_train, X_test, y_train, and y_test
    pass
