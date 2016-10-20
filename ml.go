package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/milosgajdos83/gosom/som"

	"github.com/gonum/matrix/mat64"
)

func main() {
	data := data(1000, 5, 20, 100.0, 0.0, 1.0, 10)

	TIME := time.Now()
	//algo := "batch"
	algo := "seq"
	uShape := "hexagon"
	//uShape := "rectangle"
	mUnits, coordsDims := runSom(data, algo, uShape)
	printTimer(TIME)
	file, err := os.Create("umatrix_" + algo + ".html")
	if err != nil {
		panic(err)
	}
	clusters := findClusters(mUnits, coordsDims, uShape)
	som.UMatrixSVG(mUnits, coordsDims, uShape, algo, file, clusters)
	file.Close()
}

type cluster struct {
	id int
}

type edge struct {
	node1, node2 int
	dist         float64
}

func findClusters(mUnits *mat64.Dense, coordsDims []int, uShape string) map[int]int {
	coords, _ := som.GridCoords(uShape, coordsDims)
	coordsDistMat, _ := som.DistanceMx("euclidean", coords)
	distMat, _ := som.DistanceMx("euclidean", mUnits)

	clusters := make(map[int]*cluster)
	edges := []edge{}

	numOfCoords := coordsDims[0] * coordsDims[1]
	for ni := 0; ni < numOfCoords; ni++ {
		clusters[ni] = nil
		neighbors := allRowsInRadius(ni, math.Sqrt2*1.01, coordsDistMat)
		for _, nghb := range neighbors {
			if nghb > ni {
				edges = append(edges, edge{node1: ni, node2: nghb, dist: distMat.At(ni, nghb)})
			}
		}
	}

	ei := 0
	clusterId := 0
	threshold := 2.0
	for ei < len(edges) {
		e := edges[ei]
		//fmt.Printf("%d <-> %d (%f): ", e.node1, e.node2, e.dist)
		n1c := clusters[e.node1]
		n2c := clusters[e.node2]
		n1done := n1c != nil
		n2done := n2c != nil
		closeEnough := e.dist < threshold
		if n1done && n2done {
			if closeEnough {
				if n1c.id < n2c.id {
					//fmt.Printf("Merging %d+%d -> %d", n1c.id, n2c.id, n1c.id)
					n2c.id = n1c.id
				} else {
					//fmt.Printf("Merging %d+%d -> %d", n1c.id, n2c.id, n2c.id)
					n1c.id = n2c.id
				}
			}
		}
		if !n1done {
			if !n2done {
				c := cluster{id: clusterId}
				clusterId += 1
				clusters[e.node1] = &c
				n1c = &c
				//fmt.Printf("Created cluster %d. ", n1c.id)
			} else if closeEnough {
				clusters[e.node1] = n2c
				n1c = n2c
				//fmt.Printf("Added to cluster %d. ", n2c.id)
			}
		}
		if !n2done {
			if closeEnough {
				clusters[e.node2] = n1c
				//fmt.Printf("Added to cluster %d. ", n1c.id)
			} else {
				c := cluster{id: clusterId}
				clusterId += 1
				clusters[e.node2] = &c
				n2c = &c
				//fmt.Printf("Created cluster %d. ", n2c.id)

			}
		}
		//fmt.Printf("\n")

		ei++
	}

	retVal := make(map[int]int)
	unique := make(map[int]int)
	for k, v := range clusters {
		retVal[k] = v.id
		unique[v.id] += 1
	}

	const minCoordsInCluster = 10
	majorClusterId := 0
	majorClusters := make(map[int]int)
	for k, v := range retVal {
		if unique[v] >= minCoordsInCluster {
			if _, ok := majorClusters[v]; !ok {
				majorClusters[v] = majorClusterId
				majorClusterId++
			}
		} else {
			retVal[k] = -1
		}
	}
	for k, v := range retVal {
		if retVal[k] != -1 {
			retVal[k] = majorClusters[v]
		}
	}

	println("units:", numOfCoords, "clusters:", len(majorClusters))

	return retVal
}

func allRowsInRadius(selectedRow int, radius float64, distMatrix *mat64.Dense) []int {
	rowsInRadius := []int{}
	for i, dist := range distMatrix.RowView(selectedRow).RawVector().Data {
		if dist < radius {
			rowsInRadius = append(rowsInRadius, i)
		}
	}
	return rowsInRadius
}

func runSom(data *mat64.Dense, algo string, uShape string) (*mat64.Dense, []int) {
	TIME := time.Now()

	totalIterations, _ := data.Dims()
	// SOM configuration
	mConfig := &som.MapConfig{
		Dims:     []int{43, 36},
		InitFunc: som.LinInit,
		Grid:     "planar",
		UShape:   uShape,
	}
	// create new SOM map
	smap, _ := som.NewMap(mConfig, data)

	tConfig := &som.TrainConfig{
		Method:   algo,
		Radius:   10.0,
		RDecay:   "exp",
		NeighbFn: "gaussian",
		LRate:    0.5,
		LDecay:   "exp",
	}

	smap.Train(tConfig, data, totalIterations)
	printTimer(TIME)

	return smap.Codebook(), mConfig.Dims
}

func data(rows, cols, clusters int, max, min float64, vari float64, randSeed int64) *mat64.Dense {
	rand.Seed(randSeed)

	data := mat64.NewDense(rows, cols, nil)

	clusterCentres := make([][]float64, clusters)
	for i := 0; i < clusters; i++ {
		clusterCentres[i] = randVector(max, min, cols)
		//fmt.Printf("Cluster %d = %v\n", i, clusterCentres[i])
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
