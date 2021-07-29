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

	net2 := ArtificialNeuralNetwork.CreateNetwork(3, 3, &ActivationFunctions.SigmoidActivation{})
	net2.AddLayer(3, &ActivationFunctions.SigmoidActivation{})
	net2.AddLayer(10, &ActivationFunctions.SigmoidActivation{})

	data := MatrixHelper.RandomWeightArray(3, 3)
	outputs, err := net.ForwardPropagation(data)
	if err != nil {
		print(err.Error())
		return
	}
	//net.BackPropagation([]float64{0, 1, 0})
	MatrixHelper.PrintMatrix(outputs)

	net.SaveModelWeights("model1")
	net2.LoadModelWeights("model1")

	print("\n\n\n")

	outputs, err = net2.ForwardPropagation(data)
	if err != nil {
		print(err.Error())
		return
	}
	//net.BackPropagation([]float64{0, 1, 0})
	MatrixHelper.PrintMatrix(outputs)

	//TODO: Create a load model from file (creates model with given layers, weights and bias)
	//net.CreateModelFromFile("askd")
}
