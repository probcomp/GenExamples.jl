import Gen
using Gen: choicemap, ChoiceMap, simulate, generate, get_score, get_choices
using Gen: accumulate_param_gradients!

function reweighted_wake_sleep_gradients!(
        p, p_args, q, q_args, data;
        scale_gradient=1.0,
        data_to_choicemap=(data::ChoiceMap) -> data,
        num_particles=5,
        do_q_wake_phase=true,
        do_q_sleep_phase=true,
        get_data=nothing,
        get_latent_choices=nothing)

    if do_q_sleep_phase && (isnothing(get_data) || isnothing(get_latent_choices))
        error("do_q_sleep_phase was set but either get_data or get_latent_choices was not provided")
    end

    data_choicemap = data_to_choicemap(data)

    # run q multiple times
    q_traces = Vector{Any}(undef, num_particles)
    p_traces = Vector{Any}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    Threads.@threads for i in 1:num_particles
        q_traces[i] = simulate(q, (q_args..., data))
        (p_traces[i], should_be_score) = generate(
            p, p_args, merge(data_choicemap, get_choices(q_traces[i])))
        @assert isapprox(should_be_score, get_score(p_traces[i]))
        log_weights[i] = get_score(p_traces[i]) - get_score(q_traces[i])
    end
    @assert !any(isnan.(log_weights))

    # *** synchronization barrier ***

    # normalize weights
    (_, log_normalized_weights) = Gen.normalize_weights(log_weights)
    normalized_weights = exp.(log_normalized_weights)
    @assert isapprox(sum(normalized_weights), 1.0)

    @sync begin

    # wake-phase update of p
    for i in 1:num_particles
        Threads.@spawn accumulate_param_gradients!(
            p_traces[i], nothing, Float32(normalized_weights[i] * scale_gradient))
    end
    
    # wake-phase update of q
    if do_q_wake_phase
        for i in 1:num_particles
            Threads.@spawn accumulate_param_gradients!(
                q_traces[i], nothing, Float32(normalized_weights[i] * scale_gradient))
        end
    end

    # sleep-phase update of q
    Threads.@spawn if do_q_sleep_phase
        p_trace = simulate(p, p_args)
        (q_trace, should_be_score) = generate(
            q, (q_args..., get_data(p_trace)), get_latent_choices(p_trace))
        @assert isapprox(should_be_score, get_score(q_trace))
        accumulate_param_gradients!(q_trace, nothing, Float32(scale_gradient))
    end

    # *** synchronization barrier ***

    end # @sync
end

# TODO we need to add an optional check to generate that all constraints were visited
