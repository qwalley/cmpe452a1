import numpy as np
from pprint import pprint as pprint

def loadData (filename):
	dataArray = []
	
	with open(filename, "r") as file:
		for line in file:
			# split comma separated data and remove '\n'
			splitLine = line[:-1].split(',')
			numericList = [float(val) for val in splitLine]
			# store numeric data
			dataArray.append(numericList)
	# return data in numpy array
	return np.array(dataArray)

# get z-score of each data point with respect to its column
def normalizeData (npArray):
	mean = np.mean(npArray, axis=0)
	stdDev = np.std(npArray, axis=0)

	def normalizeRow(npRow):
		row = [(val - mean[i]) / stdDev[i] if i < (len(npRow) - 1) else val for (i,), val in np.ndenumerate(npRow)]
		return row
	
	return np.apply_along_axis(normalizeRow, axis=1, arr=npArray)

# output of node 1 for single row of data
def node1Predict(inputs, weights):
	# TODO
	return 1

# output of node 2 for single row of data
def node2Predict(inputs, weights):
	# TODO
	return 1

# determine new weights using simple feedback learning
def calculateNewWeights(inputs, y, weights):
	# learning rate
	c = 0.001
	# desired output
	d = row[len(row) - 1]
	# return value
	newWeights = []

	if y > d:
		newWeights = [weights[i] - (c * inputs[i]) for i in range(len(weights))]
	elif y < d:
		newWeights = [weights[i] + (c * inputs[i]) for i in range(len(weights))]
	else:
		newWeights = weights
	
	return newWeights

# load training data, save final weights as files
def trainNetwork(dataFile):
	# TODO
	return

# load testing data and node weights, save predictions as file
def testNetwork(dataFile, node1File, node2File):
	# TODO
	return

# track history of weight changes by adding new row of weights...
#  ...every learning interation
# make initial weights random
node1Weights = [[1, 1, 1, 1, 1, 1, 1]]
node2Weights = [[1, 1, 1, 1, 1, 1, 1]]


data = loadData("trainSeeds.csv")
pprint(data)
normData = normalizeData(data)
pprint(normData)