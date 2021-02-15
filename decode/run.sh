#!/bin/bash
set -u
set -e

julia -e 'using Pkg; Pkg.update(); dir=ENV["GEN_EXAMPLE"]; include("$dir/run_all.jl")'
