__author__ = 'silkspace'


from itertools import combinations
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from time import time
from matplotlib import pyplot as plt
from collections import OrderedDict


class cleanData():

    def __init__(self, row, col, data):
        self.rows=np.asarray(row)
        self.cols=np.asarray(col)
        #don't wrap the data as nparray or it will not be None if it is input at None
        self.data=data
        self._checkThatDataIsInNormalForm()

    def _checkThatDataIsInNormalForm(self):
        assert len(self.rows) == len(self.cols), "Input rows and cols must be the same length"
        if self.data is not None:
            self.data=np.asarray(self.data)
            assert len(self.data) == len(self.cols), "Input data must be the same length as rows and cols"
            #print "Data must be integer or real valued"


    def _findIndicesOfNaNs(self, datas):
        badIndex=[]
        for index, data in enumerate(datas):
            if not isinstance(data, str):
                if np.isnan(data):
                    badIndex.append(index)
        badIndex = np.asarray(badIndex)
        return badIndex

    def _findAndFillNans(self, datas, nansAsZero):
        #the
        if datas is not None:
            badIndex = self._findIndicesOfNaNs(datas)
            if len(badIndex)>0:
                if nansAsZero:
                    datas[badIndex] = 0.0
                else:
                    for index in badIndex:
                        print index
                        try:
                            datas[index] = "Missing::Value::Error::@_"+str(index)
                        except:
                            datas[index] = index+100000000000000000000000
        return datas

    def returnCleanData(self):
        rows = self._findAndFillNans(self.rows, nansAsZero=False)
        cols = self._findAndFillNans(self.cols, nansAsZero=False)
        #data should only be real or integer valued, so we can safely coerce it to 0.0
        data = self._findAndFillNans(self.data, nansAsZero=True)
        return rows, cols, data


# take in an inhomoneous array of arrays of labels and build a matrix
data = [['foo',4], ['foo','bar',4], ['bar','nest',7]]
## should make a matrix: array([[ 1.,  0.,  0.,  1.,  0.],
##                              [ 1.,  0.,  1.,  1.,  0.],
##                               [ 0.,  1.,  1.,  0.,  1.]] etc

def matrixCreate(data):
    #creates a binary inclusion matrix on observed features
    # can take any type of data, but turns features into strings...
    allFeatures = map(str, np.unique([item for row in data for item in row]))
    colIndexDict = {feature: index for index, feature in enumerate(allFeatures)}
    numberOfRows = len(data)
    numberOfColumns = len(colIndexDict)
    #allocate memory
    M = np.zeros((numberOfRows, numberOfColumns))
    for i, row in enumerate(data):
        for feat in row:
            M[i, colIndexDict[str(feat)]]+=1
    return M, colIndexDict


class matrixCreate():
    """
        Build a matrix from Columnar Data (x, y, data).
        x and y are the observation names and feature names, resp.
        x, y can be strings, indices, real numbers, or even objects (like datetime).
        If data are None, then the binary inclusion (Indicator) matrix is built.
        data needs to be real valued.

        matrixClass = matrixCreate(x,y,data)
        matrixClass.buildMatrix()  #a csc matrix
        matrix = matrixClass.matrix

    """
    def __init__(self, rows, cols, data=None, rowtype=None, coltype=None, name=None, rowname='Observations', colname="Features", debug=False):

        self.rows, self.cols, self.data = cleanData(rows, cols, data).returnCleanData()
        self.namerow = rowname
        self.namecol = colname
        self.name=name
        # use these to assert values stay same...for example,
        # if ids are entered that are integer
        # TODO assert row,col type on final outputs
        self.rowtype=rowtype
        self.coltype = coltype
        self.debug=debug


    def _buildMatrix(self):
        if self.data is None:
            #binary inclusion
            self.data = np.ones(len(self.cols))

        #map the features to a numeric representation for building the coo_matrix
        self.observationsUnique, self.rowIndices = np.unique(self.rows, return_inverse=True)
        self.featuresUnique, self.colIndices = np.unique(self.cols, return_inverse=True)

        if self.debug:
            print "%d, %d"%(len(self.observationsUnique), len(self.featuresUnique))
        #make a dictionary of indices for the input features
        self.rowDict = dict(zip(np.arange(len(self.observationsUnique)),self.observationsUnique))
        self.colDict = dict(zip(np.arange(len(self.featuresUnique)),self.featuresUnique))

        if self.debug:
            print "%d, %d"%(len(self.rowDict), len(self.colDict))
        #make the inverse dictionaries of indices to features
        self.rowInverseDict = {observation:index for index, observation in self.rowDict.iteritems()}
        self.colInverseDict = {feature:index for index, feature in self.colDict.iteritems()}

        if self.debug:
            print "%d, %d"%(len(self.rowInverseDict), len(self.colInverseDict))

        assert len(self.rowInverseDict) == len(self.rowDict)
        assert len(self.observationsUnique) == len(self.rowDict)
        assert len(self.colInverseDict) == len(self.colDict)
        assert len(self.featuresUnique) == len(self.colDict)

        #build the coo_matrix
        self.matrix = coo_matrix((self.data, (self.rowIndices, self.colIndices)), shape=(len(self.observationsUnique),len(self.featuresUnique))).tocsc()


    def buildMatrix(self, timing=True, debug=False):
        if timing:
            t=time()
        self._buildMatrix()
        if timing:
            t = time()-t
            x,y = self.matrix.shape
            print "Building <{ %s }> of size %d,%d took %f seconds" %(self.name, x,y,t)
        return self.matrix


    def viewMatrix(self, thresh=[300,1000]):
        #visualize the matrix
        m = self.matrix
        if m.shape[0]<thresh[0]:
            thresh[0] = m.shape[0]
        if m.shape[1]<thresh[1]:
            thresh[1] = m.shape[1]
        plt.figure()
        plt.imshow(m[:thresh[0],:thresh[1]].toarray(), aspect='auto', interpolation='nearest')
        plt.colorbar()
        if self.name is None:
            plt.title("Freshly Pressed Matrix from Tabular Data \n showing only (%d,%d) window "%(thresh[0], thresh[1]))
        else:
            plt.title("%s :\n showing only (%d,%d) window "%(self.name, thresh[0], thresh[1]))
        plt.xlabel(self.namecol)
        plt.ylabel(self.namerow)



class matrixStatistics(matrixCreate):

    def __init__(self, rows, cols, data=None, name=None, rowname='Observations', colname='Features'):
        matrixCreate.__init__(self, rows=rows, cols=cols, data=data, name=name, rowname=rowname, colname=colname)
        self.buildMatrix()
        self.hasComputedLargestFeatures=False
        self.hasComputedLargestObservations=False

    def _rowSums(self):
        self.rowSums=np.asarray(self.matrix.sum(axis=1).T)[0]

    def _largestObservations(self):
        self._rowSums()
        tops = np.argsort(self.rowSums)[::-1]
        obs = np.asarray([self.rowDict[topIndex] for topIndex in tops])
        self.observationSumsDict = OrderedDict(zip(obs, self.rowSums[tops]))
        self.rowTopIndices = tops
        self.hasComputedLargestObservations=True

    def _colSums(self):
        self.colSums=np.asarray(self.matrix.sum(axis=0))[0]

    def _largestFeatures(self):
        self._colSums()
        tops = np.argsort(self.colSums)[::-1]
        features = np.asarray([self.colDict[topIndex] for topIndex in tops])
        self.featureSumsDict = OrderedDict(zip(features, self.colSums[tops]))
        self.colTopIndices = tops
        self.hasComputedLargestFeatures=True

    def _computeLargest(self):
        self._largestObservations()
        self._largestFeatures()

    def describeFeatures(self, topN=10):
        if not self.hasComputedLargestFeatures:
            self._largestFeatures()
        i =0
        print "-"*40
        for k,v in self.featureSumsDict.iteritems():
            print "%s %s has %s nonzero values across the %s"%(self.namecol, str(k), str(int(v)), self.namerow)
            i+=1
            if i>topN:
                break

    def describeObservations(self, topN=10):
        if not self.hasComputedLargestObservations:
            self._largestObservations()
        i =0
        print "-"*40
        for k,v in self.observationSumsDict.iteritems():
            print "%s %s has %s nonzero values across the %s"%(self.namerow, str(k), str(int(v)), self.namecol)
            i+=1
            if i>topN:
                break




class matrixPrune(matrixStatistics):

    def __init__(self, rows, cols, data=None, name=None, rowname='Observations', colname='Features'):
        matrixStatistics.__init__(self, rows=rows, cols=cols, data=data, name=name, rowname=rowname, colname=colname)


    def _removeColSingletons(self, colThresh=5):
        #this should only be used for binary count data in the matrix
        columnCounts = np.asarray(self.matrix.sum(0))[0]
        self.goodColumnIndices = np.where(columnCounts>colThresh)[0]
        self.matrix = self.matrix[:, self.goodColumnIndices]
        self.colDict = {i:self.colDict[k] for i, k in enumerate(self.goodColumnIndices)}
        self.colInverseDict = {k:v for v, k in self.colDict.iteritems()}
        assert len(self.colInverseDict) == len(self.colDict)


    def _removeRowSingletons(self, rowThresh=4):
        #this should only be used for binary count data in the matrix
        rowCounts = np.asarray(self.matrix.sum(1).T)[0]
        self.goodRowIndices = np.where(rowCounts>rowThresh)[0]
        self.matrix = self.matrix[self.goodRowIndices]
        self.rowDict = {i:self.rowDict[k] for i, k in enumerate(self.goodRowIndices)}
        self.rowInverseDict = {k:v for v, k in self.rowDict.iteritems()}
        assert len(self.rowInverseDict) == len(self.rowDict)


    def prune(self, rowThresh=2, colThresh=3):
        print "-"*80
        print "%s, pre-pruned matrix is (%d, %d) dimensional"%(self.name, self.matrix.shape[0], self.matrix.shape[1])
        self._removeColSingletons(colThresh=colThresh)
        self._removeRowSingletons(rowThresh=rowThresh)
        print "-"*60
        print "minimum number of %s's across any given %s: %s, \n minimum number of %s's shared by any %s: %s"%(self.namecol,self.namerow, rowThresh, self.namerow, self.namecol, colThresh)
        print "-"*60
        print "%s, Pruned matrix is now (%d, %d) dimensional"%(self.name, self.matrix.shape[0], self.matrix.shape[1])
        print "-"*80
        return self.matrix.shape



    def pruneHedge(self,rowThresh=2, colThresh=3):
        nold, mold = self.matrix.shape
        # FIXME there is an error if it cycles twice killing ALL rows
        for k in range(2*(max(rowThresh, colThresh))):
            n,m = self.prune(rowThresh, colThresh)
            if (n, m) == (nold, mold):
                break
            nold, mold = n, m



class matrixTest():

    def __init__(self, potentialX=100, potentialY=150, density=2000, debug=False):
        self.px = potentialX
        self.py = potentialY
        self.density = density
        self.debug=debug
        self.test()

    def _randomData(self):
        self.x=np.random.random_integers(0, self.px, self.density)
        self.y=np.random.random_integers(0, self.py, self.density)
        self.data=np.random.random_sample(self.density)

    def _randomBadData(self):
        # put in some nans, mixed data types (like strings with numbers, etc)
        pass

    def _expectations(self):
        x=np.unique(self.x)
        y=np.unique(self.y)
        self.expectedMatrixShape=(len(x), len(y))

    def _buildMatrixAndTestExpectation(self):
        self._randomData()
        self._expectations()
        self.matrixClass = matrixCreate(self.x, self.y, self.data, debug=self.debug)
        self.matrixClass.buildMatrix()
        self.checkExpectation()

    def _buildMatrixUsingBadDataAndTest(self):
        pass

    def checkExpectation(self):
        assert self.matrixClass.matrix.shape == self.expectedMatrixShape, "something is broken"
        print "Passed Tests"

    def test(self):
        self._buildMatrixAndTestExpectation()
        self.matrixClass.viewMatrix()



class matrixBuilderFromPandas():
    """
        Build matrices (or matrix) from a pandas DataFrame

    """
    def __init__(self, pandasData):
        assert type(pandasData) == type(pd.DataFrame()), "input object must be a pandas DataFrame"
        self.data = pandasData

    def _columns(self):
        return self.data.columns

    def _chooseColumns(self, col1, col2, dataColumn=None):
        assert type(col1) == str, 'Column names must be strings'
        assert type(col2) == str, 'Column names must be strings'
        if dataColumn is not None:
            assert type(dataColumn) == str, 'Column names must be strings'
            return self.data[[col1, col2, dataColumn]]
        else:
            return self.data[[col1, col2]]

    def _enumeratePossibleMatrices(self):
        columnNames = self._columns()
        self.combos = np.asarray([k for k in combinations(columnNames, 2)])
        self.noCombos = len(self.combos)

    def enumeratePossibleMatrices(self, thresh=100):
        self._enumeratePossibleMatrices()
        if self.noCombos>thresh:
            print "The number of combinations %d is greater than threshold %d, truncating to first %d"%(self.noCombos, thresh,thresh)
        print "Possible Matrix Indices are:"
        for comb in self.combos[:thresh]:
            print comb
            print "-"*20


    def _buildMatrix(self, colName1, colName2, dataColumn=None):
        if dataColumn is not None:
            dataFrame = self._chooseColumns(colName1, colName2, dataColumn)
            xcols, ycols, dataColumn = dataFrame.as_matrix()
        else:
            dataFrame = self._chooseColumns(colName1, colName2)
            xcols, ycols = dataFrame.as_matrix()
        return matrixCreate(xcols, ycols, dataColumn)

    def buildMatrix(self, colName1, colName2, dataColumn):
        self.matrixClass = self._buildMatrix(colName1, colName2, dataColumn)

    def returnMatrix(self):
        return self.matrixClass.buildMatrix()