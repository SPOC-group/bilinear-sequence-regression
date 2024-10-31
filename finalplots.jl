include("main.jl")

function nofix(delta, betas, rhos, alphas)
    filename = string("plots/delta_", delta, "_beta_", betas, "_rho_", rhos)
    filename_csv = filename * ".csv"
    filename_pdf = filename * ".pdf"
    data = []

    if isfile(filename_csv)
        data = CSV.read(filename_csv, DataFrame)
    else
        for (rho, beta, alpha) in Iterators.product(rhos, betas, alphas)
            alpha_strong = rho < 1 ? rho/beta * (1+beta-rho) : 1

            q = if delta == 0 && alpha > alpha_strong
                1.
            else
                iterate_q(alpha, beta, rho, delta, init = 0.5)
            end
    
            push!(data, (
                rho = rho,
                delta = delta,
                beta = beta,
                alpha = alpha,
                mmse = 1 - q,
                mse_ridge_opt = ridge(alpha, delta, delta)[1],
                alpha_LR = alpha * beta / (1+beta) / rho
            ))
    
            println((rho, beta, alpha))
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end

    plt = plot(xlabel = "alpha = n / dL", ylabel = "mmse", title = string("beta = ", betas, ", delta = ", delta, ", rho = ", rhos))
    plot!(plt, ylim = (0,1))
    @df data plot!(plt, :alpha, :mmse, group = (:rho, :beta))
    # @df data plot!(plt, :alpha, :mse_ridge_opt, group = :rho, color = :black, label = :none)
    # display(plt)
    # savefig(plt, filename_pdf)
end

############### plot fix beta, vary rho, vs alpha

function fixbeta(delta, beta, rhos, alphas)
    filename = string("plots/delta_", delta, "_beta_", beta, "_rho_", rhos)
    filename_csv = filename * ".csv"
    filename_pdf = filename * ".pdf"
    data = []

    if isfile(filename_csv)
        data = CSV.read(filename_csv, DataFrame)
    else
        for (rho, alpha) in Iterators.product(rhos, alphas)
            alpha_strong = rho < 1 ? rho/beta * (1+beta-rho) : 1

            q = if delta == 0 && alpha > alpha_strong
                1.
            else
                iterate_q(alpha, beta, rho, delta, init = 0.5)
            end
    
            push!(data, (
                rho = rho,
                delta = delta,
                beta = beta,
                alpha = alpha,
                mmse = 1 - q,
                mse_ridge_opt = ridge(alpha, delta, delta)[1],
                alpha_LR = alpha * beta / (1+beta) / rho
            ))
    
            println((rho, beta, alpha))
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end

    plt = plot(xlabel = "alpha = n / dL", ylabel = "mmse", title = string("beta = ", beta, ", delta = ", delta, ", rho = ", rhos))
    plot!(plt, ylim = (0,1))
    @df data plot!(plt, :alpha, :mmse, group = :rho)
    @df data plot!(plt, :alpha, :mse_ridge_opt, group = :rho, color = :black, label = :none)
    # display(plt)
    # savefig(plt, filename_pdf)
end

function fixbetainfinity(delta, rhos, alphas)
    plt = plot(xlabel = "alpha = n / dL", ylabel = "mmse", title = string("beta = inf, delta = ", delta, ", rho = ", rhos))
    plot!(plt, ylim = (0,1))

    data = []

    filename = string("plots/delta_", delta, "_beta_inf_rho_", rhos)
    filename_pdf = filename * ".pdf"
    filename_csv = filename * ".csv"

    for rho in rhos
        alpha_strong = rho < 1 ? rho : 1
        mses = map(a -> if delta == 0 && a > alpha_strong
                0.
            else
                1 - iterate_q_larbebeta(a, rho, delta)
            end, alphas
        )

        for (alpha, mse) in zip(alphas, mses)
            push!(data, (
                rho = rho,
                delta = delta,
                beta = Inf,
                alpha = alpha,
                mmse = mse,
                mse_ridge_opt = ridge(alpha, delta, delta)[1],
                alpha_LR = alpha / rho
            ))
        end

        plot!(plt, alphas, mses, label = rho)

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end

    # display(plt)
    # savefig(plt, filename_pdf)
end

############### plot fix beta, vary rho, vs alpha

function fixrho(delta, betas, rho, alphas)
    filename = string("plots/delta_", delta, "_beta_", betas, "_rho_", rho)
    filename_csv = filename * ".csv"
    filename_pdf = filename * ".pdf"
    data = []

    if isfile(filename_csv)
        data = CSV.read(filename_csv, DataFrame)
    else

        for (beta, alpha) in Iterators.product(betas, alphas)
            alpha_strong = rho < 1 ? rho/beta * (1+beta-rho) : 1

            q = if delta == 0 && alpha > alpha_strong
                1.
            else
                if beta != Inf
                    iterate_q(alpha, beta, rho, delta, init = 0.5)
                else
                    iterate_q_larbebeta(alpha, rho, delta)
                end
            end
    
            push!(data, (
                rho = rho,
                delta = delta,
                beta = beta,
                alpha = alpha,
                mmse = 1 - q,
                mse_ridge_opt = ridge(alpha, delta, delta)[1],
                alpha_LR = beta != Inf ? alpha * beta / (1+beta) / rho : alpha / rho
            ))
    
            println((rho, beta, alpha))
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end

    plt = plot(xlabel = "alpha", ylabel = "mmse", title = string("beta = ",betas, ", delta = ", delta, ", rho = ", rho))
    plot!(plt, ylim = (0,1))
    @df data plot!(plt, :alpha, :mmse, group = :beta)
    
    as = data.alpha |> unique |> sort
    
    # plot ridge
    mseridge = map(a -> ridge(a, delta, delta), as)
    plot!(plt, as, mseridge, label = "ridge", color = :black, linestyle = :dash)

    # plot large beta
    mselargebeta = map(a -> 1 - iterate_q_larbebeta(a, rho, delta), as)
    plot!(plt, as, mselargebeta, label = "beta = inf", color = :black, linestyle = :solid)

    # display(plt)
    # savefig(plt, filename_pdf)
end

############## plot fix beta, vary rho, vs alpha in the low rank scaling

function fixbetaLR(delta, beta, rhos, alphas)
    filename = string("plots/LR_delta_", delta, "_beta_", beta, "_rho_", rhos, "_alphaLR_", alphas)
    filename_csv = filename * ".csv"
    filename_pdf = filename * ".pdf"
    data = []

    if isfile(filename_csv)
        data = CSV.read(filename_csv, DataFrame)
    else
        rhos_no_zero = filter(x -> x != 0, rhos)

        for (rho, alpha) in Iterators.product(rhos_no_zero, alphas)
            strong = rho <= 1 ? (1 - rho / (1 + beta)) : beta / rho / (1+beta)
            alpha_true = alpha * rho * (1+beta) / beta
            alpha_LR = alpha
            q = (delta == 0 && alpha_LR > strong) ? 1. : iterate_q(alpha_true, beta, rho, delta, init = 0.5)

            push!(data, (
                rho = rho,
                delta = delta,
                beta = beta,
                alpha = alpha_true,
                mmse = 1 - q,
                mse_ridge_opt = ridge(alpha_true, delta, delta)[1],
                alpha_LR = alpha_LR
            ))

            println((rho, beta, alpha))
        end

        if 0 in rhos
            for alpha in alphas
                weak = (1+delta) * sqrt(beta) / (1+beta)
                q = alpha < weak ? 0. : low_rank_solution(beta, alpha, delta + 1e-8; init = [0.001, 0.001, 0.001], tol = 1e-6)[1]
        
                println((0., beta, alpha))
        
                push!(data, (
                    rho = 0.,
                    delta = delta,
                    beta = beta,
                    alpha = NaN,
                    mmse = 1 - q,
                    mse_ridge_opt = NaN,
                    alpha_LR = alpha
                ))
        
            end
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end

    plt = plot(xlabel = "alpha_LR = n / r(d+L)", ylabel = "mmse", title = string("beta = ", beta, ", delta = ", delta, ", rho = ", rhos))
    plot!(plt, ylim = (0,1), legend = :outerright)
    @df data plot!(plt, :alpha_LR, :mmse, group = :rho)
    @df data plot!(plt, :alpha_LR, :mse_ridge_opt, group = :rho, color = :black, label = :none)
    # display(plt)
    # savefig(plt, filename_pdf)
end

function fixbetainfinityLR(delta, rhos, alphas)
    plt = plot(xlabel = "alpha_LR = n / dL", ylabel = "mmse", title = string("beta = inf, delta = ", delta, ", rho = ", rhos))
    plot!(plt, ylim = (0,1), legend = :outerright)

    filename = string("plots/LR_delta_", delta, "_beta_inf_rho_", rhos)
    filename_pdf = filename * ".pdf"
    filename_csv = filename * ".csv"

    rhos_no_zero = filter(x -> x != 0, rhos)
    data = []

    for rho in rhos_no_zero
        strong = rho <= 1 ? 1 : 1 / rho
        alpha_true = alphas * rho
        alpha_LR = alphas

        mses = map(a -> if delta == 0 && a > strong
                0.
            else
                1 - iterate_q_larbebeta(a * rho, rho, delta)
            end, alpha_LR
        )
        plot!(plt, alphas, mses, label = rho)

        for (alpha, mse) in zip(alpha_true, mses)
            push!(data, (
                rho = rho,
                delta = delta,
                beta = Inf,
                alpha = alpha,
                mmse = mse,
                mse_ridge_opt = ridge(alpha, delta, delta)[1],
                alpha_LR = alpha / rho
            ))
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end

    if 0 in rhos
        rho = 0

        strong = rho <= 1 ? 1 : 1 / rho
        alpha_true = alphas * rho
        alpha_LR = alphas

        mses = map(a -> if delta == 0 && a > strong
                0.
            else
                1-low_rank_solution_largebeta(a, delta)
            end, alpha_LR
        )
        plot!(plt, alphas, mses, label = rho)

        for (alpha, mse) in zip(alpha_LR, mses)
            push!(data, (
                rho = rho,
                delta = delta,
                beta = Inf,
                alpha = alpha * rho,
                mmse = mse,
                mse_ridge_opt = ridge(alpha, delta, delta)[1],
                alpha_LR = alpha
            ))
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)

    end

    # display(plt)
    # savefig(plt, filename_pdf)
end

###### plot MNNE strong threshold

function MNNE_data(betas, rhos)
    filename = string("plots/MNNE_beta_", betas, "_rho_", rhos)
    filename_csv = filename * ".csv"
    data = []

    if isfile(filename_csv)
        data = CSV.read(filename_csv, DataFrame)
    else
        for (rho, beta) in Iterators.product(rhos, betas)
        
            if rho == 0
                alpha_MNNE = 0.
                alpha_BO = 0.
                alpha_LR_BO = 1
                alpha_LR_MNNE = 2 * (1+sqrt(beta)/(1+beta))
            elseif rho == 1
                alpha_MNNE, alpha_BO = (1, 1)
                alpha_LR_BO = alpha_BO * beta / (1+beta) / rho
                alpha_LR_MNNE = alpha_MNNE * beta / (1+beta) / rho
            else
                alpha_MNNE, alpha_BO = MNNE_equation(rho, beta)
                alpha_LR_BO = alpha_BO * beta / (1+beta) / rho
                alpha_LR_MNNE = alpha_MNNE * beta / (1+beta) / rho
            end
    
            push!(data, (
                rho = rho,
                beta = beta,
                alpha_BO = alpha_BO,
                alpha_MNNE = alpha_MNNE,
                alpha_LR_BO = alpha_LR_BO,
                alpha_LR_MNNE = alpha_LR_MNNE
            ))
    
            println((rho, beta))
        end

        data = DataFrame(data)
        CSV.write(filename_csv, data)
    end
end

##### plot free entropy

function plot_free_entropy(beta, rho, Delta, alphas)
    qs = vcat(1 .- 10 .^ range(start = -4, stop = log10(0.9), length = 50), 0)

    plt = plot(title = string("beta = ", beta, ", rho = ", rho), xscale = :log10, legend = :bottomright, xlabel = "log(1-q)", ylabel = "free entropy")

    data = []

    for (i,alpha) in enumerate(alphas)
        color = i

        fes = map(q -> free_entropy_vs_overlap(alpha, beta, rho, Delta, q), qs)
        q_star = iterate_q(alpha, beta, rho, Delta)
        plot!(plt, 1 .- qs, fes, color = color, label = string("alpha = ", alpha))
        scatter!(plt, [1 - q_star], [free_entropy_vs_overlap(alpha, beta, rho, Delta, q_star)], color = color, label = :none)

        for (q,fe) in zip(qs, fes)
            push!(data, (alpha = alpha, q = q, fe = fe))
        end


    end

    CSV.write("plots/hard_beta_$(beta)_rho_$(rho)_alphas_$(alphas).csv", DataFrame(data))
    # savefig(plt, "plots/hard_beta_$(beta)_rho_$(rho)_alphas_$(alphas).pdf")
    # display(plt)
end

######

# fig 1 left
fixbeta(0., 1., [0.05], 0:0.001:1)
fixbeta(0., 1., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)
fixbeta(0., 5., [0.05], 0:0.001:1)
fixbeta(0., 5., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)
fixbetainfinity(0., [0.05], 0:0.001:1)
fixbetainfinity(0., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)

# fig 1 right
fixbetaLR(0., 1., [0., 0.05], 0:0.001:1)
fixbetaLR(0., 1., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)
fixbetaLR(0., 5., [0., 0.05], 0:0.001:1)
fixbetaLR(0., 5., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)
fixbetainfinityLR(0., [0., 0.05], 0:0.001:1)
fixbetainfinityLR(0., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)

# fig 2 
fixrho(0., [1, 2, 5, 10, 100, 1000, Inf], 0.05, 0:0.001:1)
fixrho(0., [1, 2, 5, 10, 100, 1000, Inf], 0.1, 0:0.005:1)
fixrho(0., [1, 2, 5, 10, 100, 1000, Inf], 0.2, 0:0.005:1)
fixrho(0., [1, 2, 5, 10, 100, 1000, Inf], 0.5, 0:0.005:1)
fixrho(0., [1, 2, 5, 10, 100, 1000, Inf], 1, 0:0.005:1)
fixrho(0., [1, 2, 5, 10, 100, 1000, Inf], 2, 0:0.005:1)

# fig 4
MNNE_data([1, 2, 5, 10, 100], 0:0.001:1)

# fig 7
plot_free_entropy(1, 1, 0., [0.5, 0.95, 0.99])
plot_free_entropy(1, 0.5, 0., [0.35, 0.7, 0.74])
plot_free_entropy(2, 1, 0., [0.5, 0.95, 0.99])
plot_free_entropy(2, 0.5, 0., [0.2, 0.575, 0.615])

# fig 9 noisy left
fixbeta(0.1, 1., [0.05], 0:0.001:1)
fixbeta(0.1, 1., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)
fixbeta(0.1, 5., [0.05], 0:0.001:1)
fixbeta(0.1, 5., [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)
fixbetainfinity(0.1, [0.05], 0:0.001:1)
fixbetainfinity(0.1, [0.1, 0.2, 0.5, 1, 2], 0:0.005:1)

# fig 9 noisy right
fixbetaLR(0.1, 1., [0., 0.05], 0:0.004:1)
fixbetaLR(0.1, 1., [0.1, 0.2, 0.5, 1, 2], 0:0.02:1)
fixbetaLR(0.1, 5., [0., 0.05], 0:0.004:1)
fixbetaLR(0.1, 5., [0.1, 0.2, 0.5, 1, 2], 0:0.02:1)
fixbetainfinityLR(0.1, [0., 0.05], 0:0.04:1)
fixbetainfinityLR(0.1, [0.1, 0.2, 0.5, 1, 2], 0:0.02:1)

# fig 10 noisy
fixrho(0.1, [1, 2, 5, 10, 100, 1000, Inf], 0.05, 0:0.004:1)
fixrho(0.1, [1, 2, 5, 10, 100, 1000, Inf], 0.1, 0:0.02:1)
fixrho(0.1, [1, 2, 5, 10, 100, 1000, Inf], 0.2, 0:0.02:1)
fixrho(0.1, [1, 2, 5, 10, 100, 1000, Inf], 0.5, 0:0.02:1)
fixrho(0.1, [1, 2, 5, 10, 100, 1000, Inf], 1, 0:0.02:1)
fixrho(0.1, [1, 2, 5, 10, 100, 1000, Inf], 2, 0:0.02:1)