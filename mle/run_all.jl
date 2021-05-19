using Gen

@gen (static) function foo()
    @param mu::Float64
    y = @trace(normal(mu, 1), :y)
    return y
end

load_generated_functions()

init_parameter!((foo, :mu), -1)

trace, = generate(foo, (), choicemap((:y, 3)))
step_size = 0.01
for iter=1:1000
    accumulate_param_gradients!(trace, 0.)
    gradient = get_gradient((foo, :mu))
    value = get_parameter_value((foo, :mu))
    init_parameter!((foo, :mu), value + step_size * gradient)
end

@assert abs(get_parameter_value((foo, :mu)) - 3) < 1e-2
