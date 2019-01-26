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

data = loadAndNormalizeData("trainSeeds.csv")
print data[:10]