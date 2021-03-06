package ArtificialNeuralNetwork

import (
	actf "RestaurantChatbot/ArtificialNeuralNetwork/ActivationFunctions"
	mh "RestaurantChatbot/MatrixHelper"
	"errors"
	"gonum.org/v1/gonum/mat"
	"os"
)

var (
	ErrIncorrectInputSize = errors.New("incorrect input size")
)

type Network struct {
	inputLayer   Layer
	hiddenLayers []Layer
	outputLayer  Layer
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

	//TODO: Fix input layer management (does not process info, pass method is redundant)
	inputs := mat.NewDense(len(inputData), 1, inputData)
	currentOutput := net.inputLayer.Pass(*inputs) // Handle input layer

	// Handle hidden layers
	for i := 0; i < len(net.hiddenLayers); i++ {
		currentOutput = net.hiddenLayers[i].CalculateOutput(currentOutput)
	}

	//Handle output layer
	currentOutput = net.outputLayer.CalculateOutput(currentOutput)
	return &currentOutput, nil
}

// BackPropagation propagates the error backwards and adjusts the layer weights
func (net *Network) BackPropagation(targetOutput []float64) {
	//TODO: FIX THIS SHIT
	//Propagate Error through layers
	net.propagateError(targetOutput)

	// Adjust weights
	net.adjustWeights(1)
}

func (net *Network) propagateError(targetOutput []float64) {
	// Figure out error for output layer
	target := mat.NewDense(len(targetOutput), 1, targetOutput)
	outputErr := net.outputLayer.OutputLayerError(*target)
	net.outputLayer.SetError(outputErr)

	// This is in case there are no hidden layers
	if len(net.hiddenLayers) == 0 {
		return
	}

	prevLayerErr := net.outputLayer.PrevLayerError()
	net.hiddenLayers[len(net.hiddenLayers)-1].SetError(prevLayerErr)

	// Propagate error backwards
	for i := len(net.hiddenLayers) - 2; i >= 0; i-- {
		prevLayerErr := net.hiddenLayers[i+1].PrevLayerError()
		net.hiddenLayers[i].SetError(prevLayerErr)
	}
}

func (net *Network) adjustWeights(learningRate float64) {
	// First adjust weights for output layer
	product := mh.MulM(net.outputLayer.error,
		mh.ApplyM(net.outputLayer.activationFunction.ActivationPrime, net.outputLayer.lastOutput))

	if len(net.hiddenLayers) == 0 {
		net.outputLayer.weights = mh.AddM(net.outputLayer.weights,
			mh.ScaleM(learningRate,
				mh.MulM(product, net.inputLayer.lastOutput.T())))
		return
	}

	net.outputLayer.weights = mh.AddM(net.outputLayer.weights,
		mh.ScaleM(learningRate,
			mh.MulM(product, net.hiddenLayers[len(net.hiddenLayers)-1].lastOutput.T())))

	// Adjust weights for hidden layers
	for i := len(net.hiddenLayers) - 1; i >= 1; i-- {
		currentLayer := net.hiddenLayers[i]
		currentLayer.weights = mh.AddM(currentLayer.weights,
			mh.ScaleM(learningRate,
				mh.MulM(mh.MulM(currentLayer.error,
					mh.ApplyM(currentLayer.activationFunction.ActivationPrime, currentLayer.lastOutput)), net.hiddenLayers[i-1].lastOutput.T())))
	}

	// Last layer has to be done separately as it connects to input layer
	currentLayer := net.hiddenLayers[0]
	currentLayer.weights = mh.AddM(currentLayer.weights,
		mh.ScaleM(learningRate,
			mh.MulM(mh.MulM(currentLayer.error,
				mh.ApplyM(currentLayer.activationFunction.ActivationPrime, currentLayer.lastOutput)), net.inputLayer.lastOutput.T())))
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

// SaveModelWeights saves the current weights to the given file. Will NOT save the structure of the model!
func (net *Network) SaveModelWeights(fileName string) {
	file, err := os.Create(fileName+".model")

	// Check for error
	if err != nil {
		print(err.Error())
		return
	}

	defer func(file *os.File) {
		err := file.Close()
		if err != nil {

		}
	}(file)

	// Save weights of hidden layers
	for i := 0; i < len(net.hiddenLayers); i++ {
		// Dumb conversion to mat.Dense since casting is not an option
		var weights mat.Dense
		r, c := net.hiddenLayers[i].weights.Dims()
		weights.Add(net.hiddenLayers[i].weights, mh.NewZeroes(r, c))

		_, err := weights.MarshalBinaryTo(file)

		if err != nil {
			print(err.Error())
			return
		}
	}

	// Save weights of output layer
	var weights mat.Dense
	r, c := net.outputLayer.weights.Dims()
	weights.Add(net.outputLayer.weights, mh.NewZeroes(r, c))

	_, err = weights.MarshalBinaryTo(file)

	if err != nil {
		print(err.Error())
		return
	}
}

// LoadModelWeights loads weights from given file. It does NOT create the model, if the structure of the Network is different from the one originally used to write the file there will be a critical error.
func (net *Network) LoadModelWeights(fileName string) {
	file, err := os.Open(fileName+".model")

	if err != nil {
		print(err.Error())
		return
	}

	defer func(file *os.File) {
		err := file.Close()
		if err != nil {

		}
	}(file)

	// Load network weights with given numbers
	for i := 0; i < len(net.hiddenLayers); i++ {
		var weights mat.Dense
		_, err := weights.UnmarshalBinaryFrom(file)

		if err != nil {
			print(err.Error())
			return
		}

		net.hiddenLayers[i].weights = &weights
	}

	// Load output layer weights
	var weights mat.Dense
	_, err = weights.UnmarshalBinaryFrom(file)

	if err != nil {
		print(err.Error())
		return
	}

	net.outputLayer.weights = &weights
}