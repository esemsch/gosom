package som

import (
	"github.com/gonum/matrix/mat64"
)

type RowWithDist struct {
	Row  int
	Dist float64
}

func AllRowsInRadius(vec *mat64.Vector, radius float64, matrix *mat64.Dense) []RowWithDist {
	rows, _ := matrix.Dims()
	rowsInRadius := []RowWithDist{}
	for row := 0; row < rows; row++ {
		r, _ := Distance("euclidean", vec, matrix.RowView(row))
		if r < radius {
			rowsInRadius = append(rowsInRadius, RowWithDist{Row: row, Dist: r})
		}
	}
	return rowsInRadius
}
