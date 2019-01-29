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
	return activation

# if a nodes output was correct
def nodeSuccess(activation, d, correctOutput):
	output =  1 if activation >= 0 else 0
	return output == correctOutput[int(d) - 1]

# determine new weights using simple feedback learning
def calculateNewWeights(activation, correctOutput, inputs, weights):
	# learning rate
	c = 0.01
	# desired output
	d = correctOutput[inputs[-1] - 1]
	# actual output
	y = 1 if a >= 0 else 0
	# return value
	newWeights = []

	if y > d:
		# adjust regular weights
		newWeights = [weights[i] - (c * inputs[i]) for i in range(len(weights) - 1)]
		# adjust bias weight
		newWeights.append(weights[-1] - c)
	elif y < d:
		# adjust regular weights
		newWeights = [weights[i] + (c * inputs[i]) for i in range(len(weights) - 1)]
		# adjust bias weight
		newWeights.append(weights[-1] + c)
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
	
	# each correct ouput is at index (d - 1)
	node1CorrectOutput = (0, 0, 1)
	node2CorrectOutput = (0, 1, 0)

	# load data
	data = loadData("trainSeeds.csv")
	# normalize data
	normData = normalizeData(data)

	# apply a set of inputs to each node
	def processInput(npRow, i):
		faults = [None, None]
		# calculate node outputs
		node1activation = nodePredict(npRow, node1Weights[-1])
		node2activation = nodePredict(npRow, node2Weights[-1])
		# check each node for misclassification, npRow[-1] = d
		if not nodeSuccess(node1activation, npRow[-1], node1CorrectOutput): 
			faults[0] = (node1activation, i)
		if not nodeSuccess(node2activation, npRow[-1], node2CorrectOutput): 
			faults[1] = (node2activation, i)
		return faults

	def sortFaults(fault):
		# sort faults using absolute value of activation
		return abs(fault[0])

	# while not stopping condition
	
	# store misclassifications for each node separately...
	# ...for training purposes
	node1Faults = []
	node2Faults = []
	
	for i in range(normData.shape[0]):
		faults = processInput(normData[i], i)
		if faults[0] != None: node1Faults.append(faults[0])
		if faults[1] != None: node2Faults.append(faults[1])

	# sort faults to get lowest activation i.e closest to dividing line
	node1Faults.sort(key=sortFaults)
	node2Faults.sort(key=sortFaults)

	for fault in node1Faults[:5]:
		print 'fault:', fault, ' inputs:', normData[fault[1]]
	# adjust weights using the most-correct misclassification
	# TODO

	# save weights to file
	return

# load testing data and node weights, save predictions as file
def testNetwork(dataFile, node1File, node2File):
	# TODO
	return


trainNetwork()
# data = loadData("trainSeeds.csv")
# pprint(data)
# normData = normalizeData(data)
# pprint(normData)