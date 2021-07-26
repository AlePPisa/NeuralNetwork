package main

import (
	"RestaurantChatbot/ArtificialNeuralNetwork"
	"RestaurantChatbot/ArtificialNeuralNetwork/ActivationFunctions"
	"RestaurantChatbot/MatrixHelper"
)

func main() {
	net := ArtificialNeuralNetwork.CreateNetwork(3, 10, ActivationFunctions.SigmoidActivation)
	net.AddLayer(3, ActivationFunctions.SigmoidActivation)
	net.AddLayer(10, ActivationFunctions.SigmoidActivation)
	outputs, err := net.ForwardPropagation(MatrixHelper.RandomWeightArray(3, 3))
	if err != nil {
		print(err.Error())
		return
	}

	//net.CreateModelFromFile("askd")
	MatrixHelper.PrintMatrix(outputs)
}