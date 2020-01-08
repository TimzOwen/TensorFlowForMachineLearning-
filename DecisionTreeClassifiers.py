# install scikt learn 
pip install sklearn

from sklearn import tree

# have inputs ti the trauning model

features = [[140,1], [130,1], [150,0], [170,0]]

# give labels to learn 
labels = [0, 0, 1, 1]

# using Decision trees to classity
clf = tree.DecisionTreeClassifier()

# fit the objects into the model
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))

# Gives an output based on the input weights
