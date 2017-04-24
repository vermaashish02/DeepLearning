"""
Author: Ashish Verma
This code was developed to give a clear understanding of what goes behind the curtains in multi-layer feedforward backpropagation neural network.
Feel free to use/modify/improve/etc.

Caution: This may not be an efficient code for production related usage so thoroughly review and test the code before any usage.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import spatial

class multiLayerFeedForwardNN:
    def __init__(self, listOfNeuronsEachLayer):
        self.listOfNeuronsEachLayer = listOfNeuronsEachLayer
        self.dictOfWeightsEachLayer = {}
        self.dictOfActivationsEachLayer = {}
        self.dictOfDeltaEachLayer = {}
        self.dictOfZeeEachLayer = {}
        self.dictOfGradientsEachLayer = {}
        self.listAvgCost = []
        #Initialize weights with random values (this also includes bias terms)
        for iLayer in range(1,len(listOfNeuronsEachLayer)):
            self.dictOfWeightsEachLayer[iLayer + 1] = np.mat(np.random.random((listOfNeuronsEachLayer[iLayer], listOfNeuronsEachLayer[iLayer-1]+1)))

    def sigmoidFunc(self, z):
        result = 1/(1 + np.exp(-1*z))
        return result

    def feedForward(self, inputChunk):
        #Loop over all layers
        for iLayer in range(1, len(self.listOfNeuronsEachLayer)):
            #Save input itself as the activation for the input layer. We will use it later in the code
            if(iLayer == 1):
                self.dictOfActivationsEachLayer[1] = inputChunk
            #Insert extra element for bias having value 1 in input elements
            inputChunk = np.hstack([inputChunk, np.mat(np.ones(len(inputChunk))).T])
            #Calculate z using matrix multiplication of weights and inputs
            self.dictOfZeeEachLayer[iLayer+1] = np.dot(inputChunk, self.dictOfWeightsEachLayer[iLayer+1].T)
            #Calculate 'a' after non-linearizing z
            self.dictOfActivationsEachLayer[iLayer+1] = self.sigmoidFunc(self.dictOfZeeEachLayer[iLayer+1])
            #Consider activation of the current layer as input to the next layer
            inputChunk = self.dictOfActivationsEachLayer[iLayer+1]
       
    def calculateDelta(self, outputChunk):
        #Calculate delta backwards (output layer to 2nd layer)
        for iLayer in range(len(self.listOfNeuronsEachLayer),1,-1):
            #For last layer calculate delta
            if(iLayer == len(self.listOfNeuronsEachLayer)):
                self.dictOfDeltaEachLayer[iLayer] = np.multiply((self.dictOfActivationsEachLayer[iLayer] - outputChunk), np.multiply(self.dictOfActivationsEachLayer[iLayer],(1-self.dictOfActivationsEachLayer[iLayer])))
            #For rest of the layers calculate delta using delta of next layer
            else:
                wDelta = np.dot(self.dictOfDeltaEachLayer[iLayer+1], np.delete(self.dictOfWeightsEachLayer[iLayer + 1],self.dictOfWeightsEachLayer[iLayer + 1].shape[1]-1,1))
                dadz = np.multiply(self.dictOfActivationsEachLayer[iLayer],(1-self.dictOfActivationsEachLayer[iLayer]))
                self.dictOfDeltaEachLayer[iLayer] = np.multiply(wDelta, dadz)

    def calculateErrorGradient(self):
        #Calculate error gradient for all layers
        for iLayer in range(2,len(self.listOfNeuronsEachLayer)+1):
            #Gradient w.r.t all weights
            self.dictOfGradientsEachLayer[iLayer] = np.dot(self.dictOfDeltaEachLayer[iLayer].T,self.dictOfActivationsEachLayer[iLayer-1])
            #Gradient w.r.t. biases and add it to the other gradients
            self.dictOfGradientsEachLayer[iLayer] = np.hstack([self.dictOfGradientsEachLayer[iLayer], np.sum(self.dictOfDeltaEachLayer[iLayer],0).T])
    
    def updateWeights(self, learningRate):
        #For all layers update weights using average of gradients over all examples
        for iLayer in range(2, len(self.listOfNeuronsEachLayer)+1):
            self.dictOfWeightsEachLayer[iLayer] = self.dictOfWeightsEachLayer[iLayer] - (learningRate/len(self.dictOfActivationsEachLayer[1]))*self.dictOfGradientsEachLayer[iLayer]
            
    def calculateAverageCost(self, actualOutput, expectedOutput):
        #Calculate average of error function
        avgOverLayer = np.average([(1/2.)*x**2 for x in np.array(actualOutput-expectedOutput)],1)
        return np.average(avgOverLayer)            

    def trainBatchFFBP(self, learningRate, inputData, outputData, miniBatchSize, maxIterations):
        #If the size of the data is large and resources are less, then divide it into batches for computation.
        #Define batch size accordingly or leave empty if data size is not big
        if(miniBatchSize == ''):
            miniBatchSize = len(inputData)
        #Initialize iteration counter
        iterations = 0
        #Iterate over all examples 'maxIterations' number of times
        #We can even let the program iterate untill the desired error reduction is achieved (this code doesn't implement it).
        for iterations in range(maxIterations):
            #iBatch takes care of dividing the full data in batches (if we chose to, incase the data is big)
            startRecord = 0
            for iBatch in range((len(inputData)/miniBatchSize)):
                endRecord = startRecord + miniBatchSize
                inputChunk = inputData[startRecord:endRecord]
                outputChunk = outputData[startRecord:endRecord]
                self.feedForward(inputChunk)
                self.calculateDelta(outputChunk)
                self.calculateErrorGradient()
                self.updateWeights(learningRate)
                startRecord = endRecord
                #print "iBatch", iBatch
                #--------------Just stack up all calculated output for average error calculation-----------------#
                if(iBatch == 0):
                    activationMat = self.dictOfActivationsEachLayer[len(self.listOfNeuronsEachLayer)].copy()
                else:
                    activationMat = np.vstack([activationMat,self.dictOfActivationsEachLayer[self.listOfNeuronsEachLayer]])
                #------------------------------------------------------------------------------------------------#                    
            if(endRecord < len(inputData)):
                inputChunk = inputData[startRecord:]
                outputChunk = outputData[startRecord:]
                self.feedForward(inputChunk)
                self.calculateDelta(outputChunk)
                self.calculateErrorGradient()
                self.updateWeights(learningRate)
                activationMat = np.vstack([activationMat,self.dictOfActivationsEachLayer[self.listOfNeuronsEachLayer]])
            self.listAvgCost.append(self.calculateAverageCost(activationMat, outputData))
            iterations = iterations + 1
            print iterations

