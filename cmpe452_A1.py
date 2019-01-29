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
	# get mean of each column
	mean = np.mean(npArray, axis=0)
	# get standard deviation of each column
	stdDev = np.std(npArray, axis=0)

	def normalizeRow(npRow):
		# normalize all but last value in a row, last value is d
		row = [(val - mean[i]) / stdDev[i] if i < (len(npRow) - 1) else val for (i,), val in np.ndenumerate(npRow)]
		return row
	
	return np.apply_along_axis(normalizeRow, axis=1, arr=npArray)

# output of a node for single row of data
def nodePredict (inputs, weights):
	# since threshold is 1 for both nodes, they can use...
	# ...the same function with different success criteria
	activation = 0
	threshold = 1
	# sum product of weights and values, last input element is d
	for i in range(len(inputs) - 1):
		activation += inputs[i] * weights[i]
	# add bias weight to activation
	activation += threshold * weights[-1]
	# return tuple of (node prediction, desired output, activation)
	return activation

# might not need these....
def node1Success(activation, d):
	success = False
	output = activation >= 0
	correctOutput = (False, False, True)
	return output == correctOutput[d - 1]

# might not need these....
def node2Success(activation, d):
	success = False
	output = activation >= 0
	correctOutput = (False, True, False)
	return output == correctOutput[d - 1]

# determine new weights using simple feedback learning
def calculateNewWeights(inputs, y, weights):
	# learning rate
	c = 0.01
	# desired output
	d = row[len(row) - 1]
	# return value
	newWeights = []
	# ======================================================================
	# FIX THIS y is [0,1] and d is [1,2,3] make this work on a per node basis
	# ======================================================================
	if y > d:
		# adjust regular weights
		newWeights = [weights[i] - (c * inputs[i]) for i in range(len(weights) - 1)]
		# adjust bias weight
		weights[-1] -= c
	elif y < d:
		# adjust regular weights
		newWeights = [weights[i] + (c * inputs[i]) for i in range(len(weights) - 1)]
		# adjust bias weight
		weights[-1] += c
	else:
		newWeights = weights
	
	return newWeights

# load training data, save final weights as files
def trainNetwork():
	# track history of weight changes by adding new row of weights...
	#  ...every learning iteration
	# ======================================================================
	# make the intial weights random
	# ======================================================================
	node1Weights = [[1, 1, 1, 1, 1, 1, 1, 1]]
	node2Weights = [[1, 1, 1, 1, 1, 1, 1, 1]]
	
	# store misclassifications for each node separately
	node1Faults = []
	node2Faults = []

	# load data
	data = loadData("trainSeeds.csv")
	# normalize data
	normData = normalizeData(data)

	def processInput(npRow):
		node1activation = nodePredict(npRow, node1Weights)
		node2activation = nodePredict(npRow, node2Weights)

		if !node1Success(node1activation, npRow[-1]): node1Faults.append((node1activation, npRow))
		if !node2Success(node2activation, npRow[-1]): node2Faults.append((node2activation, npRow))

	# while not stopping condition
	np.apply_along_axis(processInput, axis=1, arr=normData)
	# TODO
	# sort faults by activation
	# adjust weights using the most-correct misclassification

	# save weights to file
	return

# load testing data and node weights, save predictions as file
def testNetwork(dataFile, node1File, node2File):
	# TODO
	return



# data = loadData("trainSeeds.csv")
# pprint(data)
# normData = normalizeData(data)
# pprint(normData)