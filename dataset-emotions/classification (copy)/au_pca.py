from pca import pca, project, reconstruct

from numpy import genfromtxt

emotions=["fear","happiness","anger","surprise","sadness","disgust"]

csv = genfromtxt('../emotions_action_units_training.csv', delimiter=',', skip_header=1)
training = csv[:,1:]
categories = csv[:,0]

csv = genfromtxt('../emotions_action_units_test.csv', delimiter=',', skip_header=1)

testing = csv[:,1:]
test_categories = csv[:,0]

################################################################################

eigenvalues, eigenvectors, mu = pca(training)
