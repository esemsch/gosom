package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/milosgajdos83/gosom/pkg/dataset"
	"github.com/milosgajdos83/gosom/pkg/utils"
	"github.com/milosgajdos83/gosom/som"
)

var (
	// path to input data set
	input string
	// feature scaling flag
	scale bool
	// map dimensions: 2D only [for now]
	dims string
	// map grid type: planar
	grid string
	// map unit shape type: hexagon, rectangle
	ushape string
	// initial SOM unit neihbourhood radius
	radius int
	// radius decay strategy: lin, exp
	rdecay string
	// neighbourhood func: gaussian, bubble, mexican
	neighb string
	// initial SOM learning rate
	lrate int
	// learning rate decay strategy: lin, exp
	ldecay string
)

func init() {
	flag.StringVar(&input, "input", "", "Path to input data set")
	flag.BoolVar(&scale, "scale", false, "Request data scaling")
	flag.StringVar(&dims, "dims", "", "comma-separated SOM dimensions")
	flag.StringVar(&grid, "grid", "planar", "SOM grid")
	flag.StringVar(&ushape, "ushape", "hexagon", "SOM map unit shape")
	flag.IntVar(&radius, "radius", 0, "SOM neihbourhood starting radius")
	flag.StringVar(&rdecay, "rdecay", "lin", "Radius decay strategy")
	flag.StringVar(&neighb, "neighb", "gaussian", "SOM neighbourhood function")
	flag.IntVar(&lrate, "lrate", 0, "SOM initial learning rate")
	flag.StringVar(&ldecay, "ldecay", "lin", "Learning rate decay strategy")
}

func parseCliFlags() error {
	// parse cli flags
	flag.Parse()
	// path to input data is mandatory
	if input == "" {
		return fmt.Errorf("Invalid path to input data: %s\n", input)
	}
	return nil
}

func main0() {
	// parse cli flags; exit with non-zero return code on error
	if err := parseCliFlags(); err != nil {
		fmt.Printf("Error parsing cli flags: %s\n", err)
		os.Exit(1)
	}
	// parse SOM grid dimensions
	mdims, err := utils.ParseDims(dims)
	if err != nil {
		fmt.Printf("Error parsing grid dimensions: %s\n", err)
		os.Exit(1)
	}
	// load data set from a file in provided path
	ds, err := dataset.New(input)
	if err != nil {
		fmt.Printf("Unable to load Data Set: %s\n", err)
		os.Exit(1)
	}
	// scale input data if requested
	data := ds.Data()
	if scale {
		data = ds.Scale()
	}
	// SOM configuration
	config := &som.Config{
		Dims:     mdims,
		InitFunc: som.RandInit,
		Grid:     grid,
		UShape:   ushape,
		Radius:   radius,
		RDecay:   rdecay,
		NeighbFn: neighb,
		LRate:    lrate,
		LDecay:   ldecay,
	}
	// create new SOM map
	smap, err := som.NewMap(config, data)
	if err != nil {
		fmt.Printf("Failed to create new SOM: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Hello Go SOM: %v\n", smap)
}
