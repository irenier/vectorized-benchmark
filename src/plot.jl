using CSV, DataFrames, CairoMakie

function plot_bench(df::DataFrame, output::String)
    filter!(:name => n -> occursin("/", n), df)
    transform!(df, :name => ByRow(n -> begin
        parts = split(n, "/", limit=2)
        return (method=parts[1], size=parse(Int, parts[2]))
    end) => AsTable)
    sort!(df, [:method, :size])

    methods = unique(df.method)
    colors = Makie.wong_colors()
    markers = [:circle, :rect, :diamond, :utriangle, :dtriangle,
        :ltriangle, :rtriangle, :pentagon, :hexagon, :cross]

    fig = Figure(padding=5)
    ax = Axis(fig[1, 1],
        xlabel="Size",
        ylabel="CPU Time",
        xscale=log10,
        yscale=log10,
    )

    for (i, method) in enumerate(methods)
        sub = filter(:method => m -> m == method, df)
        scatterlines!(ax, sub.size, sub.cpu_time,
            color=colors[mod1(i, length(colors))],
            marker=markers[mod1(i, length(markers))],
            label=method,
            alpha=0.6,
        )
    end

    axislegend(ax, position=:lt)
    save(output, fig)
    return fig
end

df = CSV.read("results/sumexp.csv", DataFrame; header=11)
plot_bench(df, "results/benchmark_sumexp.svg")

df = CSV.read("results/nrm2.csv", DataFrame; header=11)
plot_bench(df, "results/benchmark_nrm2.svg")

df = CSV.read("results/dot.csv", DataFrame; header=11)
plot_bench(df, "results/benchmark_dot.svg")

df = CSV.read("results/axpy.csv", DataFrame; header=11)
plot_bench(df, "results/benchmark_axpy.svg")