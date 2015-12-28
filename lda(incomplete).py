import pylab as pl
import numpys as np
from scipy import linalg as la

def lda(data,labels,redDim):

	#center data
	data-=data.mean(axis=0)
	ndata=np.shape(data)[0]
	nDim = np.shape(data)[1]

	Sw=np.zeros((nDim.nDim))
	Sb=np.zeros((nDim.nDim))

	C = np.cov(np.transpose(data))

	#loop over classes
	classes=np.unique(labels)
	for i in range(len(classes)):
		#find the relevant datapoins
		indices = np.squeeze(np.where(labels==classes[i]))
		d = np.squeeze(data[indices,:])
		classcov = np.cov(np.transpose(d))
		Sw+=np.float(np.shape(indices[0])/ndata)*classcov

	Sb = C - Sw
	