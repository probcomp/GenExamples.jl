# GenExamples.jl

[![Build Status](https://travis-ci.com/probcomp/GenExamples.jl.svg?token=bxXxGvmE2n2G9iCjKFwG&branch=main)](https://travis-ci.com/probcomp/GenExamples.jl)

A repository containing Gen examples with a Travis CI build that tests that
they run.

In the future, this repo we may also include automated tests for the
approximate correctness of the inferences in each example.

The examples are divided into directories.
Each directory has a `run_all.jl` script that runs all of the examples in the directory.
Each directory contains a Julia environment that suffices to run all of the examples in the directory.

NOTE: These examples take substantially longer to run than Gen's unit tests.
