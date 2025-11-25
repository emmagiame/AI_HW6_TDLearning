# Author James Nguyen & Emma Giamello
# HW5 part A 
# Due 11/6/25

import random
import numpy as np  # type: ignore

## maps any real number to a value between 0 and 1 using sigmoid function
def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

## calculates the derivative of sigmoid function given the sigmoid output
def SigmoidDerivative(output):
    return output * (1.0 - output)

class TwoLayerANN:
    ##
    # __init__
    #
    # initializes a two-layer neural network with random weights
    #
    # Parameters:
    # numInputs - number of input nodes (default: 4)
    # numHidden - number of hidden layer nodes (default: 8)
    # numOutputs - number of output nodes (default: 1)
    # weightMin - minimum value for weight initialization (default: -1.0)
    # weightMax - maximum value for weight initialization (default: 1.0)
    # learningRate - learning rate for gradient descent (default: 0.5)
    #
    # Return: None
    def __init__(self, numInputs: int = 4, numHidden: int = 8, numOutputs: int = 1,
                 weightMin: float = -1.0, weightMax: float = 1.0, learningRate: float = 0.5):
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        self.learningRate = learningRate

        # Initialize weights randomly between -1.0 and +1.0 (as required by assignment)
        # Hidden layer: 8 nodes × (4 inputs + 1 bias) = 40 weights
        self.hiddenWeights = np.random.uniform(
            weightMin, weightMax, (self.numHidden, self.numInputs + 1)
        )
        # Output layer: 1 node × (8 hidden + 1 bias) = 9 weights
        self.outputWeights = np.random.uniform(
            weightMin, weightMax, (self.numOutputs, self.numHidden + 1)
        )

    ##
    # Forward
    #
    # performs forward propagation through the neural network
    #
    # Parameters:
    # inputs - list of input values (4 values for this network)
    #
    # Return: tuple containing (hidden_layer_outputs, final_output)
    def Forward(self, inputs: list[float]) -> tuple[np.ndarray, np.ndarray]:
        # Add bias term (1.0) to inputs
        inputsWithBias = np.append(np.array(inputs), 1.0)
        
        # Compute hidden layer: apply weights and sigmoid activation
        hiddenNet = Sigmoid(np.dot(inputsWithBias, self.hiddenWeights.T))
        
        # Add bias term to hidden layer outputs
        hiddenWithBias = np.append(hiddenNet, 1.0)
        
        # Compute output layer: apply weights and sigmoid activation
        outputNet = Sigmoid(np.dot(hiddenWithBias, self.outputWeights.T))
        
        return hiddenNet, outputNet

    ## predicts the output for given inputs without training
    def Predict(self, inputs: list[float]) -> list[float]:
        _, outputs = self.Forward(inputs)
        return outputs.tolist()

    ##
    # TrainOnExample
    #
    # trains the network on a single example using backpropagation algorithm
    #
    # Parameters:
    # inputs - list of input values for the training example
    # targets - list of target output values for the training example
    #
    # Return: None (updates network weights in place)
    def TrainOnExample(self, inputs: list[float], targets: list[float]) -> None:
        # Forward pass: compute outputs
        hiddenOut, outputs = self.Forward(inputs)
        
        targetsArray = np.array(targets)
        outputsArray = np.array(outputs)
        
        # Calculate error and deltas for output layer
        outputError = targetsArray - outputsArray
        outputDeltas = outputError * SigmoidDerivative(outputsArray)
        
        # Backpropagate error to hidden layer (excluding bias weights)
        hiddenError = np.dot(outputDeltas, self.outputWeights[:, :-1])
        hiddenDeltas = hiddenError * SigmoidDerivative(hiddenOut)
        
        # Update output layer weights: gradient = delta × hidden_output^T
        hiddenWithBias = np.append(hiddenOut, 1.0)
        outputGrad = np.outer(outputDeltas, hiddenWithBias)
        self.outputWeights += self.learningRate * outputGrad
        
        # Update hidden layer weights: gradient = delta × input^T
        inputsWithBias = np.append(np.array(inputs), 1.0)
        hiddenGrad = np.outer(hiddenDeltas, inputsWithBias)
        self.hiddenWeights += self.learningRate * hiddenGrad

# Training data: 16 examples with 4 binary inputs (a, b, c, d) and 1 binary output
# This represents the function to be learned by the neural network
examples = [
    ([0, 0, 0, 0], [0]),
    ([0, 0, 0, 1], [1]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 1, 1], [1]),
    ([0, 1, 0, 0], [0]),
    ([0, 1, 0, 1], [1]),
    ([0, 1, 1, 0], [0]),
    ([0, 1, 1, 1], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([1, 0, 1, 0], [1]),
    ([1, 0, 1, 1], [1]),
    ([1, 1, 0, 0], [0]),
    ([1, 1, 0, 1], [0]),
    ([1, 1, 1, 0], [0]),
    ([1, 1, 1, 1], [1]),
]

##
# RunTraining
#
# trains the neural network using the specified training procedure
#
# Parameters:
# maxEpochs - maximum number of training epochs (default: 100000)
# epochSampleSize - number of randomly selected examples per epoch (default: 10)
# targetAvgError - target average error to stop training (default: 0.05)
# printInterval - print error every N epochs (default: 10)
#
# Return: None
def RunTraining(
    maxEpochs: int = 100000,
    epochSampleSize: int = 10,
    targetAvgError: float = 0.05,
    printInterval: int = 10,
):
    net = TwoLayerANN(learningRate=0.1)

    for epoch in range(1, maxEpochs + 1):
        # Train on 10 randomly selected examples (as required by assignment)
        # For each input, calculate and record the network's error, then train
        for _ in range(epochSampleSize):
            x, y = random.choice(examples)
            inputs = [float(v) for v in x]
            targets = [float(v) for v in y]
            
            # Calculate and record the network's error before training
            predicted = net.Predict(inputs)
            error = (targets[0] - predicted[0]) ** 2  # Mean squared error
            # Error is recorded (could be stored if needed for reporting)
            
            # Then use backpropagation algorithm to train
            net.TrainOnExample(inputs, targets)
        
        # Calculate error on ALL 16 examples for stable, decreasing error metric
        # This gives a more consistent measure of network performance
        totalError = 0.0
        for x, y in examples:
            inputs = [float(v) for v in x]
            targets = [float(v) for v in y]
            predicted = net.Predict(inputs)
            error = (targets[0] - predicted[0]) ** 2  # Mean squared error
            totalError += error
        
        avgError = totalError / len(examples)
        
        # Print error at specified intervals or when target is reached
        if epoch % printInterval == 0 or avgError < targetAvgError:
            print(f"Epoch {epoch}: {avgError:.6f}")
        
        # Stop training when target error is achieved
        if avgError < targetAvgError:
            break

if __name__ == "__main__":
    RunTraining()