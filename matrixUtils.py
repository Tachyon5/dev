__author__ = 'silkspace'

import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
from time import time
import logging
from scipy.optimize import nnls

# take in an inhomoneous array of arrays of labels and build a matrix
data = [['foo',4], ['foo','bar',4], ['bar','nest',7]] #this will break if you have '4' and 4 as data...shit...
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


def matrixCreateFromDict(dataDict):
    ## takes {'row1': arrayData1, 'row2', arrayData2} and creates matrix
    rowNames=[]
    data = []
    for rowName, dataArray in dataDict.iteritems():
        rowNames.append(rowName)
        data.append(dataArray)
    rowNamesDict = {name:k for k, name in enumerate(rowNames)}
    data = np.asarray(data)
    M, colIndexDict = matrixCreate(data)
    return M, rowNamesDict, colIndexDict

def matrixCreateFromDictReturnDict(dataDict):
    M, rowNamesDict, colIndexDict = matrixCreateFromDict(dataDict)
    return {'matrix': M, 'rowDict':rowNamesDict, 'colDict': colIndexDict}


def guessTopicNumber(matrix):
    topicNumberGuess = int((np.prod(matrix.shape)/((2*np.pi)**2))**(1/4.))
    print "We estimate that there are %d latent topics in this corpus"%topicNumberGuess
    return topicNumberGuess



def nmfModel(matrix, nTopics):
    t=time()
    print "Starting Factorization"
    nmf = ProjectedGradientNMF(nTopics, max_iter=220, sparseness='data', init='nndsvd')
    W = nmf.fit_transform(matrix)
    H = nmf.components_
    print "Factorization took %s minutes"%(round((time()-t)/60., 2))
    return W, H, nmf


def _topFeaturesPerAttribute(H, vocabDict, topN=100, relativeAsCounts=True):
    topfa = {}
    for z, row in enumerate(H):
        #will just kill attributes that are zero, if that is the case...
        if sum(row) != 0:
            topIndex = np.argsort(row)[::-1][:topN]
            topVals = row[topIndex]
            if relativeAsCounts:
                topVals /=(min(topVals) + 0.000001)
                topVals = map(int, topVals+1)
            topFeatures = [vocabDict[index] for index in topIndex]
            topfa[z] = zip(topFeatures, topVals)
        else:
            logging.debug("sum(H[%d]) is zero"%z)
            logging.warning("H[%d] has no consequences"%z)
    print "%d components contributing out of %d" %(len(topfa), H.shape[0])
    return topfa


def quickLogTfidf(matrix):
    # FIXME too slow
    M = np.zeros_like(matrix)
    df = np.asarray(matrix.sum(0))[0]
    print df.shape
    for i, row in enumerate(matrix):
        for j in np.nonzero(row)[0]:
            M[i,j] += -matrix[i, j]*np.log((matrix[i, j]+1)/(df[j]+1))

    return M




class query():
    """
        General query class using a model that has colDict and H (or rowDict, and W.T)

    """
    def __init__(self, model, H=None, colDict=None):
        self.model = model
        if H is None:
            self.H = self.model.H
            self.colDict = self.model.colDict
        else:
            self.H = H
            self.colDict = colDict
        self.colIds = np.asarray([self.colDict[k] for k in range(len(self.colDict))])


    def _topFeaturesPerAttribute(self, H, vocabDict, topN=60, relativeAsCounts=True):
        topfa = {}
        for z, row in enumerate(H):
            #will just kill attributes that are zero, if that is the case...
            if sum(row) != 0:
                topIndex = np.argsort(row)[::-1][:topN]
                topVals = row[topIndex]
                if relativeAsCounts:
                    topVals /=(min(topVals) + 0.000001)
                    topVals = map(int, topVals+1)
                topFeatures = [vocabDict[index] for index in topIndex]
                topfa[z] = zip(topFeatures, topVals)
            else:
                logging.debug("sum(H[%d]) is zero"%z)
                logging.warning("H[%d] has no consequences"%z)
        print "%d components contributing out of %d" %(len(topfa), H.shape[0])
        return topfa

    def computeTopFeaturesPerAttribute(self):
        self.topfa = self._topFeaturesPerAttribute(self.H, self.colDict)


### to query
    def _makeVector(self, indices, H):
        queryVector = np.zeros_like(H[0])
        indices = np.asarray(indices)
        if len(indices)==0:
            return queryVector
        queryVector[indices]+=1
        return queryVector

    def _vectorFromIds(self, colIds, H):
        indices=[]
        for id in colIds:
            if id in self.colIds:
                index = np.where(self.colIds == id)[0]
                indices.append(index)
        queryVector = self._makeVector(indices, H)
        return queryVector

    def _autoEncode(self, queryVector, H, nnlsRegress=False):
        if nnlsRegress:
            # a very clean topic assignment, most topics are zero.
            latentVector, _ = nnls(H.T, queryVector)
            recs, _ = nnls(H, latentVector)
        else:
            # slightly more noisy, which might be good
            latentVector = np.dot(H, queryVector)
            recs = np.dot(latentVector, H)
        return recs, latentVector


    def _returnRecommendedIds(self, indices):
        return np.asarray([self.colDict[k] for k in indices])

    def _recommend(self, queryVector, H, topN=8, nnlsRegress=False, normalizeStrengths=True):
        recs, latentVector = self._autoEncode(queryVector, H, nnlsRegress=nnlsRegress)
        topIndices = np.argsort(recs)[::-1][:topN]
        vals = recs[topIndices]
        if normalizeStrengths:
            vals /= sum(vals)
        return self._returnRecommendedIds(topIndices), vals

    def _query(self, colIds, H, topN=8, nnlsRegress=False):
        # return semantically similar mixes
        queryVector = self._vectorFromIds(colIds, H)
        if sum(queryVector) != 0:
            return self._recommend(queryVector, H, topN=topN, nnlsRegress=nnlsRegress)
        else:
            return None, None

    def query(self, colIds, topN=8, nnlsRegress=True):
        return self._query(colIds=colIds,H=self.H, topN=topN, nnlsRegress=nnlsRegress)




def _normalizeNgrams(ngramsWeights):
    """expects an array like [(u'sushi', 5),
        (u'shrimp', 5),
         (u'menu', 5),
         (u'rolls', 4),
         (u'crab', 4),
         (u'spicy', 4),
         (u'tuna', 4),
         (u'dishes', 4),
         (u'rice', 4),
         (u'seafood', 4)]
    """
    totalWeight = sum([k[1] for k in ngramsWeights])
    return sorted([(ngram, round(100*weight/totalWeight,2)) for ngram, weight in ngramsWeights if round(100*weight/totalWeight,2)>0], key=lambda tup: tup[1])[::-1]


def topNgramsFromW(w, res, topN=20):
    H = res.H
    Hdf = H.sum(0)
    ngrams, err = nnls(H, w)
    topIndices = np.argsort(ngrams)[::-1][:topN]
    print err, len(topIndices)
    #weights = weightFunction(Hdf, ngrams)
    return _normalizeNgrams([(res.colDict[k],ngrams[k]*Hdf[k]) for k in topIndices if ngrams[k]*Hdf[k]>0])


def topNgramsPerTopicNNLS(res, topN=20):
    topNgramsPerTopic={}
    for n in range(res.W.shape[1]):
        lv = np.zeros(res.W.shape[1])
        lv[n]+=1
        topNgramsPerTopic[n] = topNgramsFromW(lv, res)
    return topNgramsPerTopic


