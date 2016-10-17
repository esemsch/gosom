package som

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/gonum/matrix/mat64"
)

func CreateSVG(coords, mUnits *mat64.Dense, coordsDims []int, coordsType string, title string, appnd bool) {
	distMat, _ := DistanceMx("euclidean", mUnits)
	coordsDistMat, _ := DistanceMx("euclidean", coords)
	distMatRows, distMatCols := distMat.Dims()
	MAX := 0.0
	for i := 0; i < distMatRows; i++ {
		for j := i; j < distMatCols; j++ {
			if distMat.At(i, j) > MAX {
				MAX = distMat.At(i, j)
			}
		}
	}
	MUL := 20.0
	OFF := 10.0
	svg := fmt.Sprintf("<h1>%s</h1>\n", title)
	svg += fmt.Sprintf("<svg width=\"%f\" height=\"%f\">\n", float64(coordsDims[1])*MUL+2*OFF, float64(coordsDims[0])*MUL+2*OFF)
	rows, _ := coords.Dims()
	scale := func(x float64) float64 { return MUL*x + OFF }
	for row := 0; row < rows; row++ {
		coord := coords.RowView(row)
		mu := mUnits.RowView(row)
		avgDistance := 0.0
		allRowsInRadius := AllRowsInRadius(row, math.Sqrt2*1.01, coordsDistMat)
		for _, rwd := range allRowsInRadius {
			if rwd.Dist > 0.0 {
				otherMu := mUnits.RowView(rwd.Row)
				muDist, _ := Distance("euclidean", mu, otherMu)
				avgDistance += muDist
			}
		}
		avgDistance /= float64(len(allRowsInRadius) - 1)
		color := int((1.0 - avgDistance/MAX) * 255.0)
		polygonCoords := ""
		x := scale(coord.At(0, 0))
		y := scale(coord.At(1, 0))
		xOffset := 0.5 * MUL
		yOffset := 0.5 * MUL
		if coordsType == "hexagon" {
			yOffset = math.Sqrt(0.75) / 2.0 * MUL
		}
		polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y-yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y-yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x-xOffset, y+yOffset)
		polygonCoords += fmt.Sprintf("%f,%f ", x+xOffset, y+yOffset)
		svg += fmt.Sprintf("<polygon points=\"%s\" style=\"fill:rgb(%d,%d,%d);stroke:black;stroke-width:1\" />\n", polygonCoords, color, color, color)
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
