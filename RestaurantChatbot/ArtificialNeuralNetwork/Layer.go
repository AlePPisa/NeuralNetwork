package ArtificialNeuralNetwork

import (
	actf "RestaurantChatbot/ArtificialNeuralNetwork/ActivationFunctions"
	"RestaurantChatbot/MatrixHelper"
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	numNeurons int
	weights mat.Matrix
	activationFunction actf.ActivationFunction
}

func CreateLayer(numNeurons, numNeuronsPrevLayer int, function actf.ActivationFunction) Layer {
	weight := mat.NewDense(numNeurons, numNeuronsPrevLayer, MatrixHelper.RandomWeightArray(numNeurons*numNeuronsPrevLayer, float64(numNeuronsPrevLayer)))
	return Layer{numNeurons: numNeurons, weights: weight, activationFunction: function}
}

func (layer *Layer) CalculateOutput(inputs mat.Dense) mat.Dense {
	var output mat.Dense
	output.Mul(layer.weights, &inputs)
	output.Apply(layer.activationFunction, &output)
	return output
}