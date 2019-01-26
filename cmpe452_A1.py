import numpy as np

def loadAndNormalizeData (filename):
	dataArray = []
	# store max values for each column to normalize data
	maxValues = [0, 0, 0, 0, 0, 0, 0]
	
	with open(filename, "r") as file:
		for line in file:
			# split comma separated data and remove '\n'
			splitLine = line[:-1].split(',')
			numericList = [0, 0, 0, 0, 0, 0, 0, 0]
			for i in range(7):
				# convert string data to numeric values
				numericList[i] = float(splitLine[i])
				# compare each value to the current maximums
				if numericList[i] > maxValues[i]:
					maxValues[i] = numericList[i]
			# store desired output as integer
			numericList[7] = int(splitLine[7])
			# store numeric data
			dataArray.append(numericList)
		# normalize data to be in range (0,1]
		for item in dataArray:
			for i in range(7):
				# store values as a percentage of max value
				item[i] = item[i] / maxValues[i]
	# return data in numpy array
	return np.array(dataArray)

# output of node 1 for single row of data
def node1Predict(row, weights):
	# TODO
	return 1

# output of node 2 for single row of data
def node2Predict(row, weights):
	# TODO
	return 1

# determine new weights using simple feedback learning
def calculateNewWeights(row, prediction, weights):
	# TODO
	return

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
node1Weights = [[1, 1, 1, 1, 1, 1, 1]]
node2Weights = [[1, 1, 1, 1, 1, 1, 1]]

# learning rate
c = 0.001

data = loadAndNormalizeData("trainSeeds.csv")
print data[:10]