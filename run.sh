#!/bin/bash
set -u
set -e

julia -e 'include("${GEN_EXAMPLE}/run_all.jl")'
