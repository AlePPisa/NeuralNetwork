package ActivationFunctions

import "math"

type ActivationFunction func(a, b int, c float64) float64

func SigmoidActivation(a, b int, c float64) float64 {
	return 1.0/(1+math.Exp(-1*c))
}
