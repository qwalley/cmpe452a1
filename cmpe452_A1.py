import numpy as np
import random as random
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
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
def newWeights(activation, correctOutput, inputs, weights):
	# learning rate
	c = 0.1
	# desired output
	d = correctOutput[int(inputs[-1]) - 1]
	# actual output
	y = 1 if activation >= 0 else 0
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

# generate list of random floats
def randomList(lower, upper, length):
	ret = [random.uniform(lower, upper) for i in range(length)]
	return ret

# save tab-delimited list to file
def writeWeights(weights, filename):
	line = ''
	for w in range(len(weights) - 1):
		line += '{:10.8f}\t'.format(weights[w])
	line += '{:10.8f}\n'.format(weights[-1])

	with open(filename, 'w') as file:
		file.write(line)

def loadWeights (filename):
	with open(filename, "r") as file:
		line = file.readline()
		# split tab separated data and remove '\n'
		splitLine = line[:-1].split('\t')
		numericList = [float(val) for val in splitLine]
	# return data
	return numericList

# load training data, save final weights as files
def trainNetwork():
	# track history of weight changes by adding new row of weights...
	#  ...every learning iteration
	node1Weights = [randomList(-2, 2, 8)]
	node2Weights = [randomList(-2, 2, 8)]
	
	# each correct ouput is at index (d - 1)
	node1CorrectOutput = (0, 0, 1)
	node2CorrectOutput = (0, 1, 0)

	# error per run: # of faults / # of data rows
	# assume 100% at run -1
	node1Error = [1]
	node2Error = [1]

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
	for x in range(30):
		# store misclassifications for each node separately...
		# ...for training purposes
		node1Faults = []
		node2Faults = []
		
		# process trainning set
		for i in range(normData.shape[0]):
			faults = processInput(normData[i], i)
			if faults[0] != None: node1Faults.append(faults[0])
			if faults[1] != None: node2Faults.append(faults[1])

		# calculate fraction of rows that were misclassified by each node
		node1Error.append(float(len(node1Faults)) / float(normData.shape[0]))
		node2Error.append(float(len(node2Faults)) / float(normData.shape[0]))

		# sort faults to get lowest activation i.e closest to dividing line
		node1Faults.sort(key=sortFaults)
		node2Faults.sort(key=sortFaults)

		print 'run', x
		print 'Node 1 error {:4.3f}'.format(node1Error[-1]), ' Node 2 error {:4.3f}'.format(node2Error[-1])

		# adjust weights using the most-correct misclassification from faults list
		node1Weights.append(newWeights(node1Faults[0][0], node1CorrectOutput, normData[node1Faults[0][1]], node1Weights[-1]))
		node2Weights.append(newWeights(node2Faults[0][0], node2CorrectOutput, normData[node2Faults[0][1]], node2Weights[-1]))

	# save weights to file
	writeWeights(node1Weights[0], "initialNode1Weights.csv")
	writeWeights(node1Weights[-1], "node1Weights.csv")
	writeWeights(node2Weights[0], "initialNode2Weights.csv")
	writeWeights(node2Weights[-1], "node2Weights.csv")
	
	return

# load testing data and node weights, save predictions as file
def testNetwork(normData):
	# load weights
	node1Weights = loadWeights('node1Weights.csv')
	node2Weights = loadWeights('node2Weights.csv')
	# store classifications as (datapoint index, actual classification, given classification)
	results = []
	# classify each datapoint
	for i in range(normData.shape[0]):
		inputs = normData[i]
		# determine node outputs
		y1 = 1 if nodePredict(inputs, node1Weights) >= 0 else 0
		y2 = 1 if nodePredict(inputs, node2Weights) >= 0 else 0
		# determine overall classification 
		y = 0
		if (y1 == 0) and (y2 == 0): 
			y = 1
		elif (y1 == 0) and (y2 == 1): 
			y = 2
		elif (y1 == 1) and (y2 == 0): 
			y = 3 
		# store classification
		results.append(y)
	return results


# load training data
dataT = loadData("trainSeeds.csv")
# normalize data
normDataT = normalizeData(dataT)

# train the scikit ANN
clf = Perceptron(alpha=0.1, fit_intercept=False, max_iter=30, n_iter=30)
clf.fit(normDataT[:, :-1], normDataT[:, -1])

# load testing data
data = loadData("testSeeds.csv")
# normalize data
normData = normalizeData(data)

# get predictions from both ANNs
skl_results = clf.predict(normData[:, :-1])
my_results = testNetwork(normData)

# created confusion matrices
trueY = normData[:, -1]
skl_confuse = confusion_matrix(trueY, skl_results)
my_confuse = confusion_matrix(trueY, my_results)

print('skl')
print(skl_confuse)

print('mine')
print(my_confuse)

with open('results.txt', 'w') as file:
	file.write('actual\tmine\tscikit\n')
	for i in range(len(trueY)):
		line = '{:6}\t'.format(trueY[i]) + '{:4}\t'.format(my_results[i]) + '{:6}\n'.format(skl_results[i])
		file.write(line)