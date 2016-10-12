package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/milosgajdos83/gosom/som"

	"github.com/gonum/matrix/mat64"
)

func main() {
	data := data(50, 2, 2, 100.0, 0.0, 1.0, 1)

	//dims, _ := som.GridDims(data, "rectangle")
	dims := []int{2, 1}
	fmt.Printf("Dims: %v\n", dims)

	mUnits, _ := som.RandInit(data, dims)
	printMatrix(mUnits)

	coords, _ := som.GridCoords("rectangle", dims)
	printMatrix(coords)

	radius0 := 5.0
	learningRate0 := 0.5
	totalIterations, _ := data.Dims()
	for iteration := 0; iteration < totalIterations; iteration++ {
		dataRow := data.RowView(iteration)

		closest := closestMU(dataRow, mUnits)
		learningRate, _ := som.LearningRate(iteration, totalIterations, "exp", learningRate0)

		radius, _ := som.Radius(iteration, totalIterations, "exp", radius0)

		fmt.Printf("%d. Closest MU: %d, learningRate = %f, radius = %f\n", iteration, closest, learningRate, radius)

		for _, rwd := range allRowsInRadius(coords.RowView(closest), radius, coords) {
			updateMU(rwd.row, dataRow, learningRate, rwd.dist, radius, mUnits)
		}
	}

	printMatrix(mUnits)
}

func updateMU(muIndex int, dataRow *mat64.Vector, learningRate, distance, radius float64, mUnits *mat64.Dense) {
	mu := mUnits.RowView(muIndex)
	diff := mat64.NewVector(mu.Len(), nil)
	diff.AddScaledVec(dataRow, -1.0, mu)

	mul := learningRate
	if distance > 0.0 {
		mul *= som.Gaussian(distance, radius)
	}
	fmt.Printf("Updating MU %d (distance %f). Mul = %f, diff = %v\n", muIndex, distance, mul, diff)
	mu.AddScaledVec(mu, mul, diff)
}

type rowWithDist struct {
	row  int
	dist float64
}

func allRowsInRadius(vec *mat64.Vector, radius float64, matrix *mat64.Dense) []rowWithDist {
	rows, _ := matrix.Dims()
	rowsInRadius := []rowWithDist{}
	for row := 0; row < rows; row++ {
		r, _ := som.Distance("euclidean", vec, matrix.RowView(row))
		if r < radius {
			rowsInRadius = append(rowsInRadius, rowWithDist{row: row, dist: r})
		}
	}
	return rowsInRadius
}

func closestMU(dataRow *mat64.Vector, mUnits *mat64.Dense) int {
	rows, _ := mUnits.Dims()
	closest := 0
	dist := math.MaxFloat64
	for row := 0; row < rows; row++ {
		d, _ := som.Distance("euclidean", dataRow, mUnits.RowView(row))
		if d < dist {
			dist = d
			closest = row
		}
	}
	return closest
}

func data(rows, cols, clusters int, max, min float64, vari float64, randSeed int64) *mat64.Dense {
	rand.Seed(randSeed)

	data := mat64.NewDense(rows, cols, nil)

	clusterCentres := make([][]float64, clusters)
	for i := 0; i < clusters; i++ {
		clusterCentres[i] = randVector(max, min, cols)
		fmt.Printf("Cluster %d = %v\n", i, clusterCentres[i])
	}

	for i := 0; i < rows; i++ {
		clusterId := i % clusters
		rv := randVector(vari, -vari, cols)
		cc := make([]float64, cols)
		copy(cc, clusterCentres[clusterId])

		dataPoint := mat64.NewVector(cols, cc)
		rVec := mat64.NewVector(cols, rv)

		dataPoint.AddVec(dataPoint, rVec)
		data.SetRow(i, dataPoint.RawVector().Data)
	}

	return data
}

func randVector(max, min float64, cols int) []float64 {
	v := make([]float64, cols)
	for i := 0; i < cols; i++ {
		v[i] = rand.Float64()*(max-min) + min
	}
	return v
}

func printMatrix(matrix *mat64.Dense) {
	rows, _ := matrix.Dims()
	for i := 0; i < rows; i++ {
		fmt.Println(matrix.RawRowView(i))
	}
}
