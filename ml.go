package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/milosgajdos83/gosom/som"

	"github.com/gonum/matrix/mat64"
)

func main() {
	data := data(10000, 5, 5, 100.0, 0.0, 1.0, 10)

	TIME := time.Now()
	//algo := "batch"
	algo := "seq"
	uShape := "hexagon"
	mUnits, coordsDims := runSom(data, algo)
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
	id   int
	dist float64
	head node
}

type node struct {
	id       int
	previous *node
}

type edge struct {
	node1, node2 int
	dist         float64
}

type edgeList []edge

func (e edgeList) Len() int           { return len(e) }
func (e edgeList) Less(i, j int) bool { return e[i].dist <= e[j].dist }
func (e edgeList) Swap(i, j int) {
	tmp := e[i]
	e[i] = e[j]
	e[j] = tmp
}

func findClusters(mUnits *mat64.Dense, coordsDims []int, uShape string) map[int]int {
	coords, _ := som.GridCoords(uShape, coordsDims)
	coordsDistMat, _ := som.DistanceMx("euclidean", coords)
	distMat, _ := som.DistanceMx("euclidean", mUnits)

	clusters := make(map[int]*cluster)
	edges := edgeList{}

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

	sort.Sort(edges)

	ei := 0
	clusterId := 0
	for ei < len(edges) {
		e := edges[ei]
		n1c := clusters[e.node1]
		n2c := clusters[e.node2]
		n1done := n1c != nil
		n2done := n2c != nil
		if !n1done {
			if !n2done {
				c := cluster{
					id:   clusterId,
					dist: e.dist,
					head: node{id: e.node1, previous: nil},
				}
				clusterId += 1
				clusters[e.node1] = &c
				n1c = &c
			} else {
				nn := node{id: e.node1, previous: &n2c.head}
				n2c.dist = e.dist
				n2c.head = nn
				clusters[e.node1] = n2c
				n1c = n2c
			}
		}
		if !n2done {
			nn := node{id: e.node2, previous: &n1c.head}
			n1c.dist = e.dist
			n1c.head = nn
			clusters[e.node2] = n1c
		}

		ei++
	}

	println("units:", numOfCoords, "clusters:", clusterId)

	retVal := make(map[int]int)
	for k, v := range clusters {
		retVal[k] = v.id
	}

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

func runSom(data *mat64.Dense, algo string) (*mat64.Dense, []int) {
	TIME := time.Now()

	totalIterations, _ := data.Dims()
	// SOM configuration
	mConfig := &som.MapConfig{
		Dims:     []int{43, 36},
		InitFunc: som.RandInit,
		Grid:     "planar",
		UShape:   "hexagon",
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
