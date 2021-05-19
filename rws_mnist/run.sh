#!/bin/bash
set -u
set -e


cd ${GEN_EXAMPLE}
if [ ! -f mnist_salakhutdinov.jld ]; then
    wget https://gen-examples.s3.us-east-2.amazonaws.com/mnist_salakhutdinov.jld
fi
cd ..
julia -e 'using Pkg; Pkg.update(); dir=ENV["GEN_EXAMPLE"]; include("$dir/run.jl")'
