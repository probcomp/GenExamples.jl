# SGD with momentum
# grad = beta * prev_grad + (1-beta) * new_grad
# params = params + alpha * grad

import Gen

struct SGDWithMomentumConf{T}
    beta::T
    learning_rate::T
end

struct SGDWithMomentumJulia
    conf::SGDWithMomentumConf
    store::Gen.JuliaParameterStore
    parameter_ids::Vector{Tuple{Gen.GenerativeFunction,Symbol}}
    prev_updates::Vector{Any}
end

function Gen.init_optimizer(
        conf::SGDWithMomentumConf,
        parameter_ids::Vector,
        store::Gen.JuliaParameterStore=Gen.julia_default_parameter_store)

    # initialize gradients with momentum to zero
    prev_updates = Vector{Any}(undef, length(parameter_ids))
    for i in 1:length(parameter_ids)
        (gen_fn, name) = parameter_ids[i]::Tuple{Gen.GenerativeFunction,Symbol}
        prev_updates[i] = zero(Gen.get_parameter_value((gen_fn, name), store))
    end

    return SGDWithMomentumJulia(conf, store, parameter_ids, prev_updates)
end

function Gen.apply_update!(state::SGDWithMomentumJulia)
    beta = state.conf.beta
    learning_rate = state.conf.learning_rate
    for i in 1:length(state.prev_updates)
        id = state.parameter_ids[i]
        value = get_parameter_value(id, state.store)
        grad = Gen.get_gradient(id, state.store)
        update = beta * state.prev_updates[i] + (1f0 - beta) * learning_rate * grad
        new_value = Gen.in_place_add!(value, update)
        Gen.set_parameter_value!(id, new_value, state.store)
        Gen.reset_gradient!(id, state.store)
        state.prev_updates[i] = update
    end
    return nothing
end
