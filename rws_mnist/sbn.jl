# sigmoid belief networks

import Gen

sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))

mutable struct SBNTrace <: Gen.Trace
    parents::Vector{Bool}
    W::Matrix{Float32}
    b::Vector{Float32}
    outputs::Vector{Bool}
    probs::Vector{Float32}
    log_prob::Float64
end

Gen.get_args(trace::SBNTrace) = (trace.W, trace.b, trace.parents)
Gen.get_score(trace::SBNTrace) = trace.log_prob
Gen.get_retval(trace::SBNTrace) = trace.outputs
Gen.get_choices(trace::SBNTrace) = Gen.choicemap((:outputs, trace.outputs))
Gen.project(trace::SBNTrace, ::Gen.EmptySelection) = 0.0

struct SigmoidBeliefNetwork <: Gen.GenerativeFunction{Vector{Bool},SBNTrace} end

Gen.get_gen_fn(trace::SBNTrace) = SigmoidBeliefNetwork()

"""
    sigmoid_belief_network(parents::Vector{Bool}, b::Vector{Float32}, W::Matrix{Float32})

Samples an output vector of binary units given an input vector of binary units
"""
const sigmoid_belief_network = SigmoidBeliefNetwork()

#@fastmath
function sbn_compute_probs(W::Matrix{Float32}, b::Vector{Float32}, parents)
    n = length(b)
    m = length(parents)
    probs::Vector{Float32} = copy(b)
    @inbounds @simd for i in 1:n # TODO compare against switching iteration order.. so that W is accessed column-major
        @simd for j in 1:m
            if parents[j]
                probs[i] += W[i,j]
            end
        end
        probs[i] = sigmoid(probs[i])
    end
    return probs
end

#@fastmath
@inbounds function sbn_logpdf_sum(probs, outputs)
    total = 0.0
    n = length(probs)
    for i in 1:n
        if outputs[i]
            total += log(probs[i])
        else
            total += log(1.0f0 - probs[i])
        end
    end
    return total
end

#@fastmath
@inbounds function sbn_W_grad!(W_grad, outputs, probs::Vector{Float32}, parents)
    num_outputs = length(outputs)
    num_parents = length(parents)
    @assert size(W_grad) == (num_outputs, num_parents)
    @simd for col in 1:num_parents
        if parents[col]
            @simd for row in 1:num_outputs
                prob = probs[row]
                W_grad[row, col] = (outputs[row] ? 1.0f0 - prob : -prob)
            end
        else
            @simd for row in 1:num_outputs
                W_grad[row, col] = 0.0f0
            end
        end
    end
    return W_grad
end


function sbn_sample(probs::Vector{Float32})
    n = length(probs)
    outputs = Vector{Bool}(undef, n)
    @inbounds @simd for i in 1:n
        outputs[i] = rand(Float32) < probs[i]
    end
    return outputs
end

Gen.accepts_output_grad(::SigmoidBeliefNetwork) = false
Gen.has_argument_grads(::SigmoidBeliefNetwork) = (false, true, true)

function Gen.simulate(::SigmoidBeliefNetwork, args::Tuple)
    (parents, b, W) = args
    probs = sbn_compute_probs(W, b, parents)
    outputs = sbn_sample(probs)
    log_prob = sbn_logpdf_sum(probs, outputs)
    trace = SBNTrace(parents, W, b, outputs, probs, log_prob)
    return trace
end

function Gen.generate(::SigmoidBeliefNetwork, args::Tuple, choices::Gen.ChoiceMap)
    (parents, b, W) = args
    probs = sbn_compute_probs(W, b, parents)
    if Gen.has_value(choices, :outputs)
        outputs = choices[:outputs]
        log_prob = sbn_logpdf_sum(probs, outputs)
        log_weight = log_prob
    else
        outputs = sbn_sample(probs)
        log_prob = sbn_logpdf_sum(probs, outputs)
        log_weight = 0.0
    end
    trace = SBNTrace(parents, W, b, outputs, probs, log_prob)
    return (trace, log_weight)
end

function Gen.accumulate_param_gradients!(trace::SBNTrace, retval_grad::Nothing, scale_factor)
    parents = trace.parents
    outputs = trace.outputs
    probs = trace.probs
    b_grad = sum(outputs .* (1.0f0 .- probs), dims=2) - sum( (.!outputs) .* probs, dims=2)
    W = trace.W
    W_grad = Matrix{Float32}(undef, size(W)[1], size(W)[2])
    W_grad = sbn_W_grad!(W_grad, outputs, probs, parents)
    return (nothing, b_grad, W_grad)
end

###########################################
# test logpdf_grad via finite differences #
###########################################

function finite_diff_arr(f::Function, args::Tuple, i::Int, idx, dx::Real)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][idx] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][idx] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2.0f0 * dx)
end

function test_logpdf_grad()

    n = 5
    m = 3

    W = randn(Float32, n, m)
    b = randn(Float32, n)

    parents = Vector{Bool}(rand(m) .< 0.5)
    outputs = Vector{Bool}(rand(n) .< 0.5)

    (trace, log_weight) = Gen.generate(sigmoid_belief_network, (parents, b, W), Gen.choicemap((:outputs, outputs)))
    @assert isapprox(log_weight, Gen.get_score(trace))
    (_, b_grad, W_grad) = Gen.accumulate_param_gradients!(trace, nothing, NaN)

    @assert size(b_grad) == (n,)
    @assert size(W_grad) == (n, m)

    f = (b, W) -> Gen.generate(sigmoid_belief_network, (parents, b, W), Gen.choicemap((:outputs, outputs)))[2]
    dx = 1f-4

    # check gradients with respect to b
    b_grad_expected = Vector{Float32}(undef, n)
    for i in 1:n
        b_grad_expected[i] = finite_diff_arr(f, (b, W), 1, i, dx)
    end
    @assert isapprox(b_grad_expected, b_grad, rtol=2e-2)

    # check gradients with respect to W
    W_grad_flat_expected = Vector{Float32}(undef, n*m)
    for i in 1:(n*m)
        W_grad_flat_expected[i] = finite_diff_arr(f, (b, W), 2, i, dx)
    end
    @assert isapprox(W_grad_flat_expected, W_grad[:], rtol=2e-2)
end

test_logpdf_grad()
