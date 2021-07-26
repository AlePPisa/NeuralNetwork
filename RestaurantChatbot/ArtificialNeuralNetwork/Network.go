package ArtificialNeuralNetwork

import (
	actf "RestaurantChatbot/ArtificialNeuralNetwork/ActivationFunctions"
	"errors"
	"gonum.org/v1/gonum/mat"
)

var (
	ErrIncorrectInputSize = errors.New("incorrect input size")
)

type Network struct {
	inputLayer Layer
	hiddenLayers []Layer
	outputLayer Layer
}

// CreateNetwork creates a new network with the specified number of input and output neurons
// to add hidden layers use Network.AddHidden()
func CreateNetwork(inputNeurons, outputNeurons int, activationFunction actf.ActivationFunction) Network {
	net := Network{inputLayer: CreateLayer(inputNeurons, inputNeurons, activationFunction),
					outputLayer: CreateLayer(outputNeurons, inputNeurons, activationFunction)}
	return net
}

// ForwardPropagation calculates the prediction (output) for the given input
func (net *Network) ForwardPropagation(inputData []float64) (mat.Matrix, error) {
	if len(inputData) != net.inputLayer.numNeurons {
		return nil, ErrIncorrectInputSize
	}

	inputs := mat.NewDense(len(inputData), 1, inputData)
	currentOutput := net.inputLayer.CalculateOutput(*inputs) // Handle input layer

	// Handle hidden layers
	for i := 0; i < len(net.hiddenLayers); i++ {
		currentOutput = net.hiddenLayers[i].CalculateOutput(currentOutput)
	}

	//Handle output layer
	currentOutput = net.outputLayer.CalculateOutput(currentOutput)
	return &currentOutput, nil
}

// AddLayer adds a hidden layer with the specified number of neurons and with the given activation function
func (net *Network) AddLayer(numberOfNeurons int, activationFunction actf.ActivationFunction) {
	if len(net.hiddenLayers) == 0 {
		net.hiddenLayers = append(net.hiddenLayers, CreateLayer(numberOfNeurons, net.inputLayer.numNeurons, activationFunction))
	} else {
		net.hiddenLayers = append(net.hiddenLayers, CreateLayer(numberOfNeurons, net.hiddenLayers[len(net.hiddenLayers)-1].numNeurons, activationFunction))
	}

	//Need to modify output layer in case the number of neurons in prev layer changes, weights need to be recalculated
	net.outputLayer = CreateLayer(net.outputLayer.numNeurons, net.hiddenLayers[len(net.hiddenLayers)-1].numNeurons, activationFunction)
}