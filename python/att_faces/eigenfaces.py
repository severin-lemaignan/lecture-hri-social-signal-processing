from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

images = [np.asarray(Image.open(img)).flatten() for img in Path(".").glob("*/*.pgm")]
X = np.asarray(images)

# calculate 400 componentns
pca = PCA(n_components=400)
pca.fit(X)

# plot the first 16 'eigenfaces'
images = []
f, axarr = plt.subplots(4,4)
for i in range(16):
    image = pca.components_[i,:].reshape(112,92)
    axarr[i%4,i//4].imshow(image*255)

plt.show()

# reconstruct one face:

W = pca.components_
mu = pca.mean_

def project(W, X, mu=None):
    if mu is None:
        return np.dot(W,X)
    return np.dot(W,X - mu)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(W.T,Y)
    return np.dot(W.T,Y) + mu

images = []
for nb_evs in range(10, 310, 20):
    P = project(W[0:nb_evs,:], X[0], mu) # you can also use pca.transform
    print(P.shape)
    R = reconstruct(W[0:nb_evs,:], P, mu) # you can also use pca.inverse_transform
    images.append(R.reshape(112,92))

print(len(images))
for i in range(len(images)):
    axarr[i//4,i%4].imshow(images[i]*255)
plt.show()

