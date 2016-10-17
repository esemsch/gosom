package som

import (
	"github.com/gonum/matrix/mat64"
)

type RowWithDist struct {
	Row  int
	Dist float64
}

func AllRowsInRadius(selectedRow int, radius float64, distMatrix *mat64.Dense) []RowWithDist {
	rowsInRadius := []RowWithDist{}
	for i, dist := range distMatrix.RowView(selectedRow).RawVector().Data {
		if dist < radius {
			rowsInRadius = append(rowsInRadius, RowWithDist{Row: i, Dist: dist})
		}
	}
	return rowsInRadius
}
