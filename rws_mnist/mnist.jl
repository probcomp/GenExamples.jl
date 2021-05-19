import JLD

function to_bool_vectors(int_mat::Matrix)
    data = Vector{Vector{Bool}}()
    num_data = size(int_mat)[1]
    @assert size(int_mat)[2] == 28 * 28
    for i in 1:num_data
        push!(data, Vector{Bool}(int_mat[i,:] .== 1))
    end
    return data
end

function get_mnist_data()

    data = JLD.jldopen("$(@__DIR__)/mnist_salakhutdinov.jld")

    train_x = to_bool_vectors(read(data, "train_x"))
    test_x = to_bool_vectors(read(data, "test_x"))
    valid_x = to_bool_vectors(read(data, "valid_x"))

    train_data = train_x
    test_data = vcat(test_x, valid_x)

    return (train_data, test_data)
end
