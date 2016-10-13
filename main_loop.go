package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/milosgajdos83/gosom/som"

	"github.com/gonum/matrix/mat64"
)

func main() {
	data := data(10000, 2, 5, 1000.0, 0.0, 1.0, 10)

	TIME := time.Now()
	coords, mUnits, coordsDims := runSom(data)
	printTimer(TIME)
	som.CreateSVG(coords, mUnits, coordsDims, "Done", false)

	/*rows, cols := mUnits.Dims()
	repeatedMUnits := mat64.NewDense(rows*10, cols, nil)
	for i := 0; i < rows*2; i++ {
		repeatedMUnits.SetRow(i, mUnits.RowView(rand.Int()%rows).RawVector().Data)
	}
	c2, mu2, c2dims := runSom(repeatedMUnits)
	//CreateSVG(c2, mu2, c2dims)*/

}

func runSom(data *mat64.Dense) (*mat64.Dense, *mat64.Dense, []int) {
	TIME := time.Now()
	dims, _ := som.GridDims(data, "hexagon")
	fmt.Printf("Dims: %v\n", dims)

	mUnits, _ := som.LinInit(data, dims)
	//printMatrix(mUnits)

	coords, _ := som.GridCoords("hexagon", dims)
	//printMatrix(coords)
	printTimer(TIME)

	radius0 := 10.0
	learningRate0 := 0.5
	totalIterations, _ := data.Dims()
	for iteration := 0; iteration < totalIterations; iteration++ {
		dataRow := data.RowView(iteration)
		/*
			closest := closestMU(dataRow, mUnits)
			learningRate, _ := som.LearningRate(iteration, totalIterations, "exp", learningRate0)

			radius, _ := som.Radius(iteration, totalIterations, "exp", radius0)
		*/
		//fmt.Printf("%d. Closest MU: %d, learningRate = %f, radius = %f\n", iteration, closest, learningRate, radius)
		closest := 0
		learningRate := learningRate0
		radius := radius0

		for _, rwd := range som.AllRowsInRadius(coords.RowView(closest), radius, coords) {
			updateMU(rwd.Row, dataRow, learningRate, rwd.Dist, radius, mUnits)
		}
		/*if iteration%50 == 0 {
			appnd := iteration != 0
			CreateSVG(coords, mUnits, dims, fmt.Sprintf("Iteration %d", iteration), appnd)
		}*/
	}

	return coords, mUnits, dims
}

func updateMU(muIndex int, dataRow *mat64.Vector, learningRate, distance, radius float64, mUnits *mat64.Dense) {
	mu := mUnits.RowView(muIndex)
	diff := mat64.NewVector(mu.Len(), nil)
	diff.AddScaledVec(dataRow, -1.0, mu)

	mul := learningRate
	if distance > 0.0 {
		mul *= som.Gaussian(distance, radius)
	}
	//fmt.Printf("Updating MU %d (distance %f). Mul = %f, diff = %v\n", muIndex, distance, mul, diff)
	mu.AddScaledVec(mu, mul, diff)
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

func printTimer(TIME time.Time) {
	elapsed := float64((time.Now().UnixNano() - TIME.UnixNano()))
	units := []string{"ns", "us", "ms", "s"}
	for i, unit := range units {
		if elapsed > 1000.0 && i < (len(units)-1) {
			elapsed /= 1000.0
		} else {
			fmt.Printf("%f%s\n", elapsed, unit)
			break
		}
	}
}
