package main

import (
	"RestaurantChatbot/ArtificialNeuralNetwork"
	"RestaurantChatbot/ArtificialNeuralNetwork/ActivationFunctions"
	"RestaurantChatbot/MatrixHelper"
)

func main() {
	net := ArtificialNeuralNetwork.CreateNetwork(3, 3, &ActivationFunctions.SigmoidActivation{})
	net.AddLayer(3, &ActivationFunctions.SigmoidActivation{})
	net.AddLayer(10, &ActivationFunctions.SigmoidActivation{})
	outputs, err := net.ForwardPropagation(MatrixHelper.RandomWeightArray(3, 3))
	if err != nil {
		print(err.Error())
		return
	}
	net.BackPropagation([]float64{0, 1, 0})

	//TODO: Create a load model from file (creates model with given layers, weights and bias)
	//net.CreateModelFromFile("askd")
	MatrixHelper.PrintMatrix(outputs)
}
