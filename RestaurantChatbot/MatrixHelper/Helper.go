package MatrixHelper

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

// PrintMatrix prints the given matrix onto the console with a nice format.
func PrintMatrix(matrix mat.Matrix) {
	A := mat.Formatted(matrix, mat.Squeeze())
	fmt.Printf("%4.6f", A)
	print("\n")
}

// AddScalar adds a scalar to each individual element.
// and saves the result on the given matrix.
func AddScalar(matrix mat.Dense, scalar float64) {
	r, c := matrix.Dims()
	scalarM := NewOnes(r, c)
	scalarM.Scale(scalar, scalarM)
	matrix.Add(scalarM, &matrix)
}

// NewOnes returns a new Ones matrix with the given dimensions.
func NewOnes(r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		data[i] = 1
	}
	return mat.NewDense(r, c, data)
}

// NewZeroes returns a new Zeroes matrix with the given dimensions.
func NewZeroes(r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		data[i] = 0
	}
	return mat.NewDense(r, c, data)
}

// RandomWeightArray returns a data array used to initialize of the specified size with random entries using uniform distribution.
func RandomWeightArray(size int, lastLayerSize float64) []float64 {
	data := make([]float64, size)
	distribution := distuv.Uniform{ //Apparently this is a common type of distribution in neural networks
		Min: -1 / math.Sqrt(lastLayerSize),
		Max: 1 / math.Sqrt(lastLayerSize),
	}
	for i := 0; i < size; i++ {
		data[i] = distribution.Rand()
	}

	return data
}

// MulM wrapper for Mul in the Gonum Mat. Returns the product of the multiplication.
func MulM(A, B mat.Matrix) mat.Matrix {
	var product mat.Dense
	product.Mul(A, B)
	return &product
}

// AddM wrapper for Add in the Gonum Mat. Returns the result of the addition.
func AddM(A, B mat.Matrix) mat.Matrix {
	var sum mat.Dense
	sum.Add(A, B)
	return &sum
}

// ScaleM wrapper for Scale in the Gonum Mat. Returns the scaled matrix.
func ScaleM(alpha float64, A mat.Matrix) mat.Matrix {
	var scaledM mat.Dense
	scaledM.Scale(alpha, A)
	return &scaledM
}

// ApplyM wrapper for Apply in the Gonum Mat. Returns the resulting matrix after applying a function to each element.
func ApplyM(fn func(a, b int, c float64) float64, A mat.Matrix) mat.Matrix {
	var result mat.Dense
	result.Apply(fn, A)
	return &result
}