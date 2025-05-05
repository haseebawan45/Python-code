#kon from sklearn

from sklearn import neighbors, datasets

#import some data to play with iris datasets.load_iris()

# we only take the first two features for demonstration

X = iris.data[:, :2]

y = iris.target

clf = neighbors.KNeighborsClassifier(n_neighbors=15)

clf.fit(X, y)