#!/bin/bash
set -u
set -e

# build PyCall; needed since this example currently requires PyPlot / matplotlib
julia -e 'ENV["PYTHON"] = ""; using Pkg; Pkg.update(); Pkg.build("PyCall")'

julia -e 'dir=ENV["GEN_EXAMPLE"]; include("$dir/run_all.jl")'
