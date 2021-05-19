include("$(@__DIR__)/sbn.jl")
include("$(@__DIR__)/model.jl")
include("$(@__DIR__)/rws.jl")
include("$(@__DIR__)/sgd_momentum.jl")
include("$(@__DIR__)/mnist.jl")

import Random

function get_minibatch(data::Vector{Vector{Bool}}, minibatch_size::Int)
    n = length(data)
    idx = Random.randperm(n)[1:minibatch_size]
    return data[idx]
end

import Gen

function datum_to_choicemap(datum::Vector{Bool})
    choices = Gen.choicemap()
    choices[:x => :outputs] = datum
    return choices
end

function estimate_log_marginal_likelihood(data::Vector{Vector{Bool}}, est_particles)
    n = length(data)
    log_ml_estimates = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
         datum = data[i]
        (_, log_ml_estimates[i]) = Gen.importance_resampling(
            p, (), datum_to_choicemap(datum), q,
            (datum,), est_particles)
    end
    return sum(log_ml_estimates) / n
end

function do_checkpoint(train_data, test_data, est_size, est_particles, save_to, iter)
    train_est_batch = get_minibatch(train_data, est_size)
    train_lml_est = estimate_log_marginal_likelihood(train_est_batch, est_particles)
    println("iter $iter, train set LML estimate: $train_lml_est")
    if !isnothing(save_to)
        metadata = Dict()
        save_params(p, q, metadata, save_to)
    end
end

function train(
        train_data, test_data, iters::Int;
        est_size=256, est_particles=10000,
        save_to=nothing,
        checkpoint_period=10,
        num_particles=5,
        minibatch_size=24,
        momentum_beta=0.95f0,
        learning_rate=0.001f0)

    # training config
    conf = SGDWithMomentumConf(momentum_beta, learning_rate)
    p_update = Gen.init_optimizer(conf, p)
    q_update = Gen.init_optimizer(conf, q)

    # do training using the whole data set as a batch
    for iter in 1:iters

        # checkpoint
        if checkpoint_period != Inf && (((iter-1) % checkpoint_period == 0) || (iter == iters))
            println("checkpointing..")
            do_checkpoint(train_data, test_data, est_size, est_particles, save_to, iter)
        end

        # select a minibatch
        minibatch = get_minibatch(train_data, minibatch_size)

        # NOTE be sure to set environment variable OPENBLAS_NUM_THREADS=1
        # and set number of threads with e.g. JULIA_NUM_THREADS=8
        Threads.@threads for datum in minibatch

            # compute stochastic gradient estimates
            reweighted_wake_sleep_gradients!(
                p, (), q, (), datum;
                scale_gradient=1.0f0/minibatch_size,
                data_to_choicemap=datum_to_choicemap,
                num_particles=num_particles,
                do_q_wake_phase=true,
                do_q_sleep_phase=true,
                get_data=(p_trace) -> p_trace[:x => :outputs]::Vector{Bool},
                get_latent_choices=(p_trace) -> get_choices(p_trace))

        end

        # apply updates and reset the gradient estimates
        apply_update!(p_update)
        apply_update!(q_update)
    end
end

function do_training()

    Random.seed!(9)
    
    (train_data, test_data) = get_mnist_data()
    
    initialize_p_params!()
    initialize_q_params!()
    
    max_epochs = 1 # one epoch is about 2000 iterations of SGD 
    iters_per_epoch = Int(round(50000 / 24))
    checkpoint_period = Int(round(10 * iters_per_epoch)) # every 10 epochs
    iters = Int(round((iters_per_epoch * max_epochs)))
    println("iters_per_epoch: $iters_per_epoch, running for $iters iters...")
    train(train_data, test_data, iters;
        save_to="params.jls",
        num_particles=5,
        checkpoint_period=1000,#checkpoint_period,
        est_particles=100,
        momentum_beta=0.95f0,
        learning_rate=1f-3,
        minibatch_size=32)

end

do_training()
