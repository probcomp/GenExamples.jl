#!/bin/bash
set -u
set -e

julia -e 'dir=ENV["GEN_EXAMPLE"]; include("$dir/run_all.jl")'
