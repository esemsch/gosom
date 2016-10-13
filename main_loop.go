package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/milosgajdos83/gosom/som"

	"github.com/gonum/matrix/mat64"
)

func main() {
	data := data(10000, 2, 5, 1000.0, 0.0, 1.0, 10)

	TIME := time.Now()
	coords, mUnits, coordsDims := runSom(data)
	printTimer(TIME)
	createSVG(coords, mUnits, coordsDims, "Done", false)

	/*rows, cols := mUnits.Dims()
	repeatedMUnits := mat64.NewDense(rows*10, cols, nil)
	for i := 0; i < rows*2; i++ {
		repeatedMUnits.SetRow(i, mUnits.RowView(rand.Int()%rows).RawVector().Data)
	}
	c2, mu2, c2dims := runSom(repeatedMUnits)
	//createSVG(c2, mu2, c2dims)*/

}

func runSom(data *mat64.Dense) (*mat64.Dense, *mat64.Dense, []int) {
	dims, _ := som.GridDims(data, "hexagon")
	fmt.Printf("Dims: %v\n", dims)

	mUnits, _ := som.LinInit(data, dims)
	//printMatrix(mUnits)

	coords, _ := som.GridCoords("hexagon", dims)
	//printMatrix(coords)

	radius0 := 10.0
	learningRate0 := 0.5
	totalIterations, _ := data.Dims()
	for iteration := 0; iteration < totalIterations; iteration++ {
		dataRow := data.RowView(iteration)

		closest := closestMU(dataRow, mUnits)
		learningRate, _ := som.LearningRate(iteration, totalIterations, "exp", learningRate0)

		radius, _ := som.Radius(iteration, totalIterations, "exp", radius0)

		//fmt.Printf("%d. Closest MU: %d, learningRate = %f, radius = %f\n", iteration, closest, learningRate, radius)

		for _, rwd := range allRowsInRadius(coords.RowView(closest), radius, coords) {
			updateMU(rwd.row, dataRow, learningRate, rwd.dist, radius, mUnits)
		}
		/*if iteration%50 == 0 {
			appnd := iteration != 0
			createSVG(coords, mUnits, dims, fmt.Sprintf("Iteration %d", iteration), appnd)
		}*/
	}

	return coords, mUnits, dims
}

func createSVG(coords, mUnits *mat64.Dense, coordsDims []int, title string, appnd bool) {
	distMat, _ := som.DistanceMx("euclidean", mUnits)
	distMatRows, distMatCols := distMat.Dims()
	MAX := 0.0
	for i := 0; i < distMatRows; i++ {
		for j := i; j < distMatCols; j++ {
			if distMat.At(i, j) > MAX {
				MAX = distMat.At(i, j)
			}
		}
	}
	MUL := 50.0
	OFF := 10.0
	svg := fmt.Sprintf("<h1>%s</h1>\n", title)
	svg += fmt.Sprintf("<svg width=\"%f\" height=\"%f\">\n", float64(coordsDims[1])*MUL+2*OFF, float64(coordsDims[0])*MUL+2*OFF)
	rows, _ := coords.Dims()
	scale := func(x float64) float64 { return MUL*x + OFF }
	for row := 0; row < rows; row++ {
		coord := coords.RowView(row)
		svg += fmt.Sprintf("<circle cx=\"%f\" cy=\"%f\" r=\"5\" stroke=\"green\" stroke-width=\"3\" fill=\"yellow\" />\n", scale(coord.At(0, 0)), scale(coord.At(1, 0)))
		mu := mUnits.RowView(row)
		for _, rwd := range allRowsInRadius(coord, math.Sqrt2*1.01, coords) {
			if rwd.dist > 0.0 {
				coord2 := coords.RowView(rwd.row)
				otherMu := mUnits.RowView(rwd.row)
				muDist, _ := som.Distance("euclidean", mu, otherMu)
				color := int((1.0 - muDist/MAX) * 255.0)
				svg += fmt.Sprintf("<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" style=\"stroke:rgb(%d,%d,%d);stroke-width:2\" />\n", scale(coord.At(0, 0)), scale(coord.At(1, 0)), scale(coord2.At(0, 0)), scale(coord2.At(1, 0)), color, color, color)
			}
		}
	}
	svg += `</svg>`

	if appnd {
		file, err := os.OpenFile("umatrix.html", os.O_APPEND|os.O_RDWR, 0660)
		if err != nil {
			ioutil.WriteFile("umatrix.html", []byte(svg), os.ModePerm)
		} else {
			file.WriteString(svg)
			file.Close()
		}
	} else {
		ioutil.WriteFile("umatrix.html", []byte(svg), os.ModePerm)
	}
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
