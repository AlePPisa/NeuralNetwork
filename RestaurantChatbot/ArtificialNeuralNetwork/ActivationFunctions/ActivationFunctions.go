package ActivationFunctions

import "math"

type ActivationFunction interface {
	Activation(a, b int, c float64) float64
	ActivationPrime(a, b int, c float64) float64
}

type SigmoidActivation struct {
}

func (sigmoid *SigmoidActivation) Activation(a, b int, c float64) float64 {
	return 1.0 / (1 + math.Exp(-1*c))
}

func (sigmoid *SigmoidActivation) ActivationPrime(a, b int, c float64) float64 {
	return math.Exp(-1*c) / math.Pow(1+math.Exp(-1*c), 2)
}
