using Gen: @gen, @param, @load_generated_functions

@gen (static) function p()

    # prior on the fourth hidden layer
    @param h3_b::Vector{Float32}

    # third hidden layer
    @param h2_W::Matrix{Float32}
    @param h2_b::Vector{Float32}

    # third hidden layer
    @param h1_W::Matrix{Float32}
    @param h1_b::Vector{Float32}

    # visible layer
    @param x_W::Matrix{Float32}
    @param x_b::Vector{Float32}

    # TODO issue -- Gen wraps h3_n in a ReverseDiff, which causes a failure 
    h3_n = 10
    #h3_n = length(h3_b)

    #h2_n = length(h2_b)
    #h1_n = length(h1_b)
    #h1_n = 200
    #x_n = length(x_b)

    # sample third hidden layer (10 units)
    h3 ~ sigmoid_belief_network([false], h3_b, zeros(Float32, h3_n, 1))

    # sample second hidden layer (200 units)
    #@assert size(h2_W) == (h2_n, h3_n) # TODO @assert is not supported in static modeling language
    h2 ~ sigmoid_belief_network(h3, h2_b, h2_W)

    # sample first hidden layer (200 units)
    #@assert size(h1_W) == (h1_n, h2_n)
    h1 ~ sigmoid_belief_network(h2, h1_b, h1_W)
    #h1 ~ sigmoid_belief_network([false], h1_b, zeros(Float32, h1_n, 1))

    # sample visible layer (28x28=784 units)
    #@assert size(x_W) == (x_n, h1_n)
    x ~ sigmoid_belief_network(h1, x_b, x_W)

    return nothing
end

@gen (static) function q(x::Vector{Bool})
    #@assert sum(x) > 0

    # first hidden layer
    @param h1_W::Matrix{Float32}
    @param h1_b::Vector{Float32}

    # second hidden layer
    @param h2_W::Matrix{Float32}
    @param h2_b::Vector{Float32}

    # third hidden layer
    @param h3_W::Matrix{Float32}
    @param h3_b::Vector{Float32}

    #x_n = length(x)
    #h1_n = length(h1_b)
    #h2_n = length(h2_b)
    #h3_n = length(h3_b)
    
    # sample first hidden layer (200 units)
    #@assert size(h1_W) == (h1_n, x_n)
    h1 ~ sigmoid_belief_network(x, h1_b, h1_W)

    # sample second hidden layer (200 units)
    #@assert size(h2_W) == (h2_n, h1_n)
    h2 ~ sigmoid_belief_network(h1, h2_b, h2_W)

    # sample third hidden layer (10 units)
    #@assert size(h3_W) == (h3_n, h2_n)
    h3 ~ sigmoid_belief_network(h2, h3_b, h3_W)

    return nothing
end

@load_generated_functions()

import Random
using Gen: apply_update!, init_parameter!, get_parameter_value

# see https://github.com/jbornschein/reweighted-ws/blob/e96414719d09ab4941dc77bab4cf4847acc6a8e7/learning/models/sbn.py#L94
# and https://github.com/jbornschein/reweighted-ws/blob/e96414719d09ab4941dc77bab4cf4847acc6a8e7/learning/model.py#L22
function make_W(n_output, n_input)
    scale = Float32(sqrt(6f0) / sqrt(n_input + n_output))
    return scale * ((2*rand(Float32, n_output, n_input)) .- 1.0f0) / n_input
end

# see https://github.com/jbornschein/reweighted-ws/blob/e96414719d09ab4941dc77bab4cf4847acc6a8e7/learning/models/sbn.py#L94
make_b(n) = -ones(Float32, n)

function initialize_p_params!(; x_n=28*28, h1_n=200, h2_n=200, h3_n=10)
    init_parameter!((p, :h3_b), make_b(h3_n))

    init_parameter!((p, :h2_W), make_W(h2_n, h3_n))
    init_parameter!((p, :h2_b), make_b(h2_n))

    init_parameter!((p, :h1_W), make_W(h1_n, h2_n))
    init_parameter!((p, :h1_b), make_b(h1_n))

    init_parameter!((p, :x_W), make_W(x_n, h1_n))
    init_parameter!((p, :x_b), make_b(x_n))
end

function initialize_q_params!(; x_n=28*28, h1_n=200, h2_n=200, h3_n=10)
    init_parameter!((q, :h1_W), make_W(h1_n, x_n))
    init_parameter!((q, :h1_b), make_b(h1_n))

    init_parameter!((q, :h2_W), make_W(h2_n, h1_n))
    init_parameter!((q, :h2_b), make_b(h2_n))

    init_parameter!((q, :h3_W), make_W(h3_n, h2_n))
    init_parameter!((q, :h3_b), make_b(h3_n))
end

import Serialization

function save_params(p, q, metadata, filename)
    println("saving params to $filename")
    p_params_dict = Dict(
        name => Gen.get_parameter_value((p, name))
        for (_, name) in Gen.get_parameters(p, Gen.default_parameter_context)[Gen.default_julia_parameter_store])
    q_params_dict = Dict(
        name => Gen.get_parameter_value((q, name))
        for (_, name) in Gen.get_parameters(q, Gen.default_parameter_context)[Gen.default_julia_parameter_store])
    data = Dict(
        "metadata" => metadata,
        "q_params" => q_params_dict,
        "p_params" => p_params_dict)
    Serialization.serialize(filename, data)
    return nothing
end

function load_params!(p, q, filename)
    println("loading params from $filename")
    data = Serialization.deserialize(filename)
    println("got metadata: $(data["metadata"])")
    for name in keys(data["p_params"])
        init_parameter!((p, name), data["p_params"][name])
        println("$name: $(size(Gen.get_parameter_value(p, name)))")
    end
    for name in keys(data["q_params"])
        init_parameter!((q, name), data["q_params"][name])
        println("$name: $(size(Gen.get_parameter_value(q, name)))")
    end
    return data["metadata"]
end
