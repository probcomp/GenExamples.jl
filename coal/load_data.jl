import CSV
import DataFrames

function load_data_set()
    df = CSV.read("$(@__DIR__)/coal.csv", DataFrames.DataFrame)
    dates = df[!,1]
    dates = dates .- minimum(dates)
    dates * 365.25 # convert years to days
end

