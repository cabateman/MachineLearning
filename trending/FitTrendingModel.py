from numpy import *
from scipy import *
import numpy
import scipy
import matplotlib.pyplot as plt
"""
	Title:  FitTrendingModel.py
	Author: Chris Bates
	Description:  This is the python version of the FitTrendingModel.m file.  It accomplishes the same task, 
	namely after being supplied a training and test set it can form a model which can then be used to do predictions.  
	Also included are two helper functions- a FileToArray function which acts basically like the load() function in matlab, 
	and the cleanData() function which eliminates any outliers that may skew the results.

	Matplotlib is used to perform the scatter plots
"""
def FileToArray(file,limit=None):
	"""
	Supposed to mimic Matlab's load function
	Arguments: 
		file - file object
		limit - optional parameter to limit the number of rows in matrix
	"""
	filelist = file.read().split("\n")
	rowcount = len(filelist)
	fieldlist = filelist[0].split("\t")
	columncount = len(fieldlist)
	# Pre-allocate memory for array
	if limit is not None:
		result = zeros(shape=(limit,columncount),dtype=int)
	else:
		result = zeros(shape=(rowcount-1,columncount),dtype=int)
	count=0
	for line in filelist:
		if len(line)>0:
			fields = line.split("\t")
			if limit is not None:
				if limit == count:
					break
				result[count,:]=fields
				count = count+1
			else:
				result[count,:]=fields
				count = count+1

	return result
def cleanData(data):
	x = zeros(shape=(len(data),1))
	x[:,0] = data
	std_x = std(x,0)
	for i in range(len(x)):
		if x[i]>10*std_x:
			x[i]=std_x
	return x

def FitTrendingModel(testdata,traindata,normalize,order,Lambda):
    	"""
	This function builds a polynomial regression model that is used to do 
	trending analysis.

	It solves the least squares normal equations including the diagonally loaded case
	"""
	
	# INITIALIZE
	x		= testdata[:,1]
	y 		= array(testdata[:,2],float)/array(normalize,float)
	xn		= array(traindata[:,0])
	tn		= cleanData(traindata[:,1])
	NumPts		= len(traindata[:,0])
	xn.shape	= (1,NumPts)
	xn		= xn.transpose()
	NumPts_test 	= len(testdata[:,0])
	x.shape 	= (1,NumPts_test)
	x		= x.transpose()
	exparr 		= tile(range(order+1),(NumPts,1))
	exparr_test	= tile(range(order+1),(NumPts_test,1))
	Aall		= tile(xn,(1,order+1))**exparr
	Aall_test	= tile(x,(1,order+1))**exparr_test
	exparr		= tile(range(order+1),(NumPts,1))
	powerall	= tile(x,(1,order+1))**exparr

	# FORM THE A MATRIX
	A 		= array(Aall[:,0:order+1],float)
	A_t		= array(Aall[:,0:order+1],float)	
	A_t		= A_t.transpose()

	
	# SOLVE NORMAL EQUATIONS FOR THE NON-REGULARIZED CASE
	w 		= dot(dot(linalg.pinv(dot(A_t,A)),A_t),tn)

	#BUILD THE NON-REGULARIZED APPROXIMATION POLYNOMIAL
	powers 		= powerall[:,0:order+1]
	approx 		= dot(powers,w)

	
	# FORM THE DIAGONALLY LOADED A MATRIX
	nsize		= A.shape[1]
	AL		= dot(A_t,A)+dot(Lambda,eye(nsize))
	

	# SOLVE DIAGONALLY LOADED NORMAL EQUATIONS FOR THE REGULARLIZED CASE
	w 		= dot(dot(linalg.pinv(AL),A_t),tn)
	approx 		= dot(powers,w)
	
	plt.scatter(xn,tn)
	plt.plot(xn,approx)
	plt.show()

if __name__ == "__main__":
    
    	# TODO: Must declare what testdata, traindata, and normalization data will be!!
	FitTrendingModel(testdata,traindata,normalizedata,3,.01)
