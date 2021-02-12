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

## Examples

### Variants of Bayesian polynomial regression

`regression/`

### Variants of Bayesian inference over the structure of Gaussian process covariance function

`gp_structure/`

### Minimal example of [involutive MCMC](https://arxiv.org/abs/2007.09871)

`involutive_mcmc/`

### Decoding a substitution cipher using a bigram model and parallel tempering MCMC

`decode/`

### Minimal example of maximum likelihood estimation

`mle/`
