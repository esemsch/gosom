package som

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/gonum/matrix/mat64"
)

func CreateSVG(coords, mUnits *mat64.Dense, coordsDims []int, title string, appnd bool) {
	distMat, _ := DistanceMx("euclidean", mUnits)
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
		for _, rwd := range AllRowsInRadius(coord, math.Sqrt2*1.01, coords) {
			if rwd.Dist > 0.0 {
				coord2 := coords.RowView(rwd.Row)
				otherMu := mUnits.RowView(rwd.Row)
				muDist, _ := Distance("euclidean", mu, otherMu)
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
