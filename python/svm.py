from numpy import genfromtxt

from sklearn import svm

csv = genfromtxt('data.csv', delimiter=',')
data = csv[:,:2]
categories = csv[:,2]


inputs = [ [3.5,3], [1.5,3], [1.8,1.9] ]

k = 3

clf = svm.SVC(kernel='rbf', gamma = 0.7, C=1.0)
clf.fit(data, categories)

predictions = clf.predict(inputs)
print(predictions)

