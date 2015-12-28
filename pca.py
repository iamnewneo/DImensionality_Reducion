import numpy as np

def pca(data,nRedDim=0,normalise=1):
	m=np.mean(data,axis=0)
	#print("Actual data")
	#print(data)
	data-=m

	#covariance matrix
	c=np.cov(np.transpose(data))

	#compute eigenvalues and eigenvectors
	evals,evecs=np.linalg.eig(c)
	indices = np.argsort(evals)
	#to sort in decending
	indices=indices[::-1]
	evecs=evecs[:,indices]
	evals=evals[indices]
	if nRedDim>0:
		evecs=evecs[:,:nRedDim]

	if normalise:
		for i in range(np.shape(evecs)[1]):
			evecs[:,i]/np.linalg.norm(evecs[:,i]*np.sqrt(evals[i]))

	#produce the new data matrix
	x = np.dot(np.transpose(evecs),np.transpose(data))
	#compute origial data again
	y=np.dot(evecs,x)
	print("Actual data")
	print(data)
	print("\n\n")
	print("Aprroximated value in k reduced dimensions")
	print(y.T)
	approx = y.T
	# to calculate accuracy checl if variance is retained
	summation1=0
	summation2=0
	for i in range(np.shape(data)[0]):
		for j in range(np.shape(data)[1]):
			summation1+=(data[i,j]-approx[i,j])**2
			summation2+=data[i,j]**2
	accuracy = summation1/summation2
	print("accuracy = " + str(100-accuracy))
	return c,y,evals,evecs
iris= np.loadtxt('iris_proc.data',delimiter=',')
pca(iris[:10,:4],1)