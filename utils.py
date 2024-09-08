import numpy as np

def standard_scaler(x):
    """
     convert the data to have a mean of 0 and a standard deviation of 1
    """
    
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    if isinstance(centers, int):
        centers = np.random.uniform(center_box[0], center_box[1], size=(centers, n_features))
    else:
        centers = np.array(centers)

    if len(centers) < 1:
        raise ValueError("The number of centers must be at least 1")

    n_clusters = len(centers)

    n_samples_per_cluster = [n_samples // n_clusters] * n_clusters
    for i in range(n_samples % n_clusters):
        n_samples_per_cluster[i] += 1

    X = []
    y = []

    for i, (n, center) in enumerate(zip(n_samples_per_cluster, centers)):
        cluster_points = np.random.normal(loc=center, scale=cluster_std, size=(n, n_features))
        X.append(cluster_points)
        y += [i] * n

    X = np.vstack(X)
    y = np.array(y)

    return X, y

def to_categorical(y):
    """
        [0, 4, 2] -> [ [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0] ]
    """

    categorical = np.zeros((y.size, y.max()+1))
    categorical[np.arange(y.size), y] = 1
    return categorical

def train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True):
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    n_test = int(X.shape[0] * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    return X_train, X_test, y_train, y_test


