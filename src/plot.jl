using CSV, DataFrames, Plots

df = CSV.read("results/sumexp.csv", DataFrame; header=12)

filter!(:name => n -> occursin("/", n), df)
transform!(df, :name => ByRow(n -> begin
    parts = split(n, "/", limit=2)
    return (method=parts[1], size=parse(Int, parts[2]))
end) => AsTable)
sort!(df, [:method, :size])

p = plot(
    df.size,
    df.cpu_time,
    group=df.method,
    markershape=:circle,
    markersize=4,
    linewidth=1,
    xscale=:log10,
    yscale=:log10,
    xlabel="Size",
    ylabel="CPU Time",
    legend=:topleft,
    size=(700, 500),
    # minorgrid = true,
)

# 保存图片
savefig(p, "results/benchmark_sumexp.svg")