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
	fmt.Printf("%4.3f", A)
	print("\n")
}

// AddScalar adds a scalar to each individual element.
// and saves the result on the given matrix
func AddScalar(matrix mat.Dense, scalar float64) {
	r, c := matrix.Dims()
	scalarM := NewOnes(r,c)
	scalarM.Scale(scalar, scalarM)
	matrix.Add(scalarM, &matrix)
}

// NewOnes returns a new Ones matrix with the given dimensions
func NewOnes(r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		data[i] = 1
	}
	return mat.NewDense(r,c,data)
}

// RandomWeightArray returns a data array used to initialize of the specified size with random entries using uniform distribution
func RandomWeightArray(size int, lastLayerSize float64) []float64 {
	data := make([]float64, size)
	distribution := distuv.Uniform { //Apparently this is a common type of distribution in neural networks
		Min: -1/math.Sqrt(lastLayerSize),
		Max: 1/math.Sqrt(lastLayerSize),
	}
	for i := 0; i < size; i++ {
		data[i] = distribution.Rand()
	}

	return data
}