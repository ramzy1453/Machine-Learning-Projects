import numpy as np

class KNearestNeighbors:
    
    def __init__(self, n_neighbors) -> None:
        self.n_neighbors = n_neighbors
        
    def __repr__(self) -> str:
        return f"KNearestNeighbors(n_neighbors={self.n_neighbors})"
    
    def __euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(x1-x2)**2)
    
    def train_model(self, X_train, y_train):
        self.__X_train = X_train
        self.__y_train = y_train
        
    def __predict_one(self, x):
        distances = [self.__euclidean_distance(x, x_train) for x_train in self.__X_train]
        k_neighbors_indices = np.argsort(distances)[:self.n_neighbors]
        k_labels = self.__y_train[k_neighbors_indices]
        
        labels, counts = np.unique(k_labels, return_counts=True)
        return labels[np.argsort(counts)][0]
    
    def predict(self, X):
        return np.array([self.__predict_one(x) for x in X])
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.count_nonzero(y_pred - y) / len(y)
        