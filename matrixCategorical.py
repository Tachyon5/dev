__author__ = 'silkspace'

from collections import Counter
import numpy as np
from time import time
import operator


def checkIfFloatOrCategorical(entry):
    try:
        float(entry)
        return 'float'
    except:
        return 'categorical'

def removeMissingValuesFromSet(items, missingValues):
    items = set(items)
    for missingval in missingValues:
        items.discard(missingval)
    return np.asarray(sorted(list(items))) #sort them to get a better chance of alignment across
    # the different data sets (ie, train and test where some values might be missing.

def distributionOfValuesInColumn(column, missingValues=['']):
    uniqueItems = np.unique(column)
    uniqueItems = removeMissingValuesFromSet(uniqueItems, missingValues=missingValues)
    return Counter(uniqueItems)
    
def distributionOfValuesInData(dataMatrix, missingValues=['']):
    return [distributionOfValuesInColumn(column, missingValues=missingValues) for column in dataMatrix.T]
    

def checkColumnForDistributionOfValuesOrCategories(column, missingValues=['']):
    uniqueItems = np.unique(column)
    uniqueItems = removeMissingValuesFromSet(uniqueItems, missingValues=missingValues)
    floatOrCategoricalCounter = Counter(map(checkIfFloatOrCategorical, uniqueItems))
    return floatOrCategoricalCounter


def distributionOfFloatOrCategoricalColumnValuesInData(dataMatrix, missingValues=['']):
    counters = [checkColumnForDistributionOfValuesOrCategories(column, missingValues=missingValues) for column in dataMatrix.T]
    return counters


def checkIfMixedOrHomogeneousColumnVariable(counterObject, types=['float','categorical']):
    counts=[]
    for varType in types:
        counts.append(counterObject[varType])
    if 0 not in counts: #signifies if there are mixed types, which need to be dealt with
        return 'Mixed_Column_Type'
    else:
        return types

def columnType(counterObject, types=['float','categorical']):
    isMixed = checkIfMixedOrHomogeneousColumnVariable(counterObject, types=types)
    if isMixed[0] in types:
        return counterObject.keys()[0]
    else:
        return isMixed


def dataTypeOfColumnsInData(dataMatrix, missingValues=['']):
    typeCounters = distributionOfFloatOrCategoricalColumnValuesInData(dataMatrix, missingValues=missingValues)
    return zip(np.arange(dataMatrix.shape[1]), map(columnType, typeCounters))


def mapToFloats(column, missingValues=[''], convertMissingTo=0.):
    for index, item in enumerate(column):
        if item in missingValues:
            column[index]=convertMissingTo
    return map(float, column)

############################################################
# Now that we have identified the types in the data matrix,
# we move onto making it a full matrix for algorithms
############################################################

def matrixFromCategories(column, missingValues=[''], categoryStrength=1):
    uniqueFeatures, ColumnIndices = np.unique(column, return_inverse=True)
    uniqueFeatures2 = removeMissingValuesFromSet(uniqueFeatures, missingValues=missingValues)
    featureIndexDict = {feature: index for index, feature in enumerate(uniqueFeatures2)}
    #allocate memory
    M = np.zeros((len(column), len(uniqueFeatures2)))
    for i, columnValue in enumerate(column):
        if columnValue in uniqueFeatures2: #the missing values will not be included
            j = featureIndexDict[columnValue]
            M[i,j]+=categoryStrength
    return M, featureIndexDict


def buildMatrix(dataMatrix, missingValues=[''], headers= None, categoryStrengthDict=None, treatMixedColumnTypeAsCategorical=True):
    t = time()
    indexDataType = dataTypeOfColumnsInData(dataMatrix, missingValues=missingValues)
    print "indexed all the data types across the %d columns in %f seconds"%(dataMatrix.shape[1], time()-t)
    supraFeatureIndex={}
    matrixDict={}
    Mtotal = np.zeros((dataMatrix.shape[0], 1)) #allocate some memory.
    print "Building Super Matrix"
    for index, type in indexDataType:
        if type in ('categorical'):
            matrix, featureIndexDict = matrixFromCategories(dataMatrix[:,index])
            matrixDict[index] = matrix
            supraFeatureIndex[index] = featureIndexDict
        elif type in ('float'):
            matrix = mapToFloats(dataMatrix[:,index], missingValues=missingValues, convertMissingTo=0.)
            matrixDict[index] = matrix
            supraFeatureIndex[index] = {}
        elif type in ('Mixed_Column_Type') and treatMixedColumnTypeAsCategorical:
            # if treatMixedColumnTypeAsCategorical is False, it skips over this 'feature'
            matrix, featureIndexDict = matrixFromCategories(dataMatrix[:,index])
            matrixDict[index] = matrix
            supraFeatureIndex[index] = featureIndexDict
        #put the different matrices together recursively
        Mtotal = np.c_[Mtotal, matrix]
    print "Super Matrix Built in %s seconds"%(time()-t)
    #kill the first column that we had to include to make M_total recursively built
    return Mtotal[:,1:], alignFeaturesIndexDict(supraFeatureIndex, headers=headers)


def alignFeaturesIndexDict(superFeaturesIndex, headers=None):
    masterDict = {}
    if headers is None:
        headers = range(len(superFeaturesIndex))
    columIndex = 0
    for index, featureDict in superFeaturesIndex.items():
        columnWidth = len(featureDict)
        if columnWidth ==0: #just a numerical column
            columnWidth=1
            masterDict[columIndex]=headers[index]
        else:
            featureIndexSorted = sorted(featureDict.iteritems(), key=operator.itemgetter(1))
            for feature, insideIndex in featureIndexSorted:
                masterDict[insideIndex+ columIndex] = feature
        columIndex+=columnWidth
    return masterDict


def realignMatrix(correctAlignmentDict, newMasterDict, Mcorrect, Mtest):
    master2Inverse = {v:k for k,v in newMasterDict.iteritems()}
    Mnew = np.zeros_like(Mcorrect)
    for correctIndex, name in correctAlignmentDict.items():
        if name in master2Inverse:
            #this will skip over any element that is not shared between both dictionaries
            ## but that is okay for now...
            oldIndex = master2Inverse[name]
            Mnew[:,correctIndex] = Mtest[:,oldIndex]
    return Mnew


#################################################################
# Now we want to calculate the information quantity in data
#
#
