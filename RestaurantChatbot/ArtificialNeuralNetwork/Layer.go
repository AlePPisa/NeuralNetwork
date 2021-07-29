package ArtificialNeuralNetwork

import (
	actf "RestaurantChatbot/ArtificialNeuralNetwork/ActivationFunctions"
	"RestaurantChatbot/MatrixHelper"
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	numNeurons         int
	weights            mat.Matrix
	lastOutput         mat.Matrix
	error              mat.Matrix
	activationFunction actf.ActivationFunction
}

//CreateLayer returns a new Layer struct with the given number of neurons, activation function, and a uniformly randomized weight matrix.
func CreateLayer(numNeurons, numNeuronsPrevLayer int, function actf.ActivationFunction) Layer {
	weight := mat.NewDense(numNeurons, numNeuronsPrevLayer, MatrixHelper.RandomWeightArray(numNeurons*numNeuronsPrevLayer, float64(numNeuronsPrevLayer)))
	return Layer{numNeurons: numNeurons, weights: weight, activationFunction: function}
}

//CalculateOutput returns a mat.Dense (technically a vector) as the output of the given layer.
// Applies weights and activation function.
func (layer *Layer) CalculateOutput(inputs mat.Dense) mat.Dense {
	var output mat.Dense
	output.Mul(layer.weights, &inputs)
	output.Apply(layer.activationFunction.Activation, &output)
	layer.lastOutput = &output
	return output
}

//Pass is a redundant function that should nto exist. It does nothing but allowing me to keep the code structure
// I had. It needs to go
func (layer *Layer) Pass(inputs mat.Dense) mat.Dense {
	layer.lastOutput = &inputs
	return inputs
}

// PrevLayerError returns the error belonging to the previous layer
func (layer *Layer) PrevLayerError() mat.Dense {
	var err mat.Dense
	err.Mul(layer.weights.T(), layer.error)
	return err
}

// OutputLayerError is ONLY supposed to be called on the OUTPUT LAYER. Failure to do so will result in faulty behavior.
// Returns the error for the output layer.
func (layer *Layer) OutputLayerError(target mat.Dense) mat.Dense {
	var err mat.Dense
	err.Sub(&target, layer.lastOutput)
	return err
}

// SetError sets the given value as the layers error
func (layer *Layer) SetError(err mat.Dense) {
	layer.error = &err
}

func (layer *Layer) PrintWeights() {
	MatrixHelper.PrintMatrix(layer.weights)
}
