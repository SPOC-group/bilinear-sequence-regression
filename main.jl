using Random, Polynomials, LinearAlgebra, QuadGK, NLsolve, Roots, Statistics, HCubature
using Plots, DataFrames, CSV, StatsPlots

# normalised as Y = (A B / sqrt(r) + sqrt(noise) Z) / sqrt[4](d1 * d2)
# for Z d1 x d2 and A d1 x r and B r x d2, all iid std Gaussians
# R1 = max(d1,d2) / min(d1,d2)
# R2 = r / min(d1,d2)
function stieltjes(R1, R2, R_noise, x; eps = 1e-8)
    phi = R2 / R1
    psi = R2
    eta = (1+R_noise) * sqrt(R1)
    zeta = sqrt(R1)
    z = x - eps * im

    # here already we have z^2!
    a0 = -psi^3
    a1 = psi * ( zeta * (psi - phi) + psi * ( eta * (phi - psi) + psi * z^2 ) )
    a2 = - zeta^2 * (phi - psi)^2 + zeta * ( eta * (phi - psi)^2 + psi * z^2 * (2 * phi - psi) ) - eta * psi^2 * z^2 * phi
    a3 = - zeta * z^2 * phi * ( 2 * zeta * psi - 2 * zeta * phi - 2 * eta * psi + 2 * eta * phi + psi * z^2 )
    a4 = zeta * z^4 * phi^2 * (eta - zeta)
    
    poly = Polynomial([a0, a1, a2, a3, a4])

    rts = z * roots(poly) 

    i = argmax(imag.(rts)) 
    G = rts[i]
    pdf = imag(G) / pi
    velocity = real(G)
    (pdf, velocity, G)
end

function pdf_edges(R1, R2, R_noise)
    phi = R2 / R1
    psi = R2
    eta = (1+R_noise) * sqrt(R1)
    zeta = sqrt(R1)

    a0 = phi^2*(phi - psi)^4*psi^6*(eta - zeta)^2*zeta^2*(eta^2*phi^2 + 2*eta*phi*zeta + (1 - 4*phi)*zeta^2)*
    (eta^2*psi^2 + 2*eta*psi*zeta + (1 - 4*psi)*zeta^2)

    a1 = 2*phi^2*(phi - psi)^2*psi^7*(eta - zeta)*zeta*
    (2*eta^5*phi^3*psi^3 + 3*eta^4*phi^2*psi^2*(phi + psi)*zeta - eta^3*phi*psi*(psi^2 + 2*phi*psi*(-3 + 4*psi) + phi^2*(1 + 8*psi))*
      zeta^2 + eta^2*(-34*phi^2*psi^2 - 2*psi^3 + 9*phi*psi^3 + phi^3*(-2 + 9*psi))*zeta^3 + 
     eta*(phi^3*(1 - 4*psi) + phi*(7 - 4*psi)*psi^2 + (-3 + psi)*psi^2 + phi^2*(-1 + 5*psi)*(3 + 8*psi))*zeta^4 - 
     (phi + psi - 5*psi^2 + 2*phi*psi*(-3 + 8*psi) + phi^2*(-5 + 16*psi))*zeta^5)
 
    a2 = phi^2*psi^8*zeta*(-8*eta^5*phi^3*psi^3*(phi + psi) + eta^4*phi^2*psi^2*(-23*psi^2 + 2*phi*psi*(-9 + 4*psi) + phi^2*(-23 + 8*psi))*
      zeta + 2*eta^3*phi*psi*(-5*psi^3 + phi^2*psi*(-19 + 10*psi) + phi*psi^2*(-19 + 43*psi) + phi^3*(-5 + 43*psi))*zeta^2 - 
     2*eta^2*(-3*psi^4 + phi*psi^3*(13 + 5*psi) + phi^2*psi^2*(12 + 31*(-3 + psi)*psi) + phi^3*psi*(13 + psi*(-93 + 2*psi)) + 
       phi^4*(-3 + psi*(5 + 31*psi)))*zeta^3 + 2*eta*(-3*(-1 + psi)*psi^3 + phi^4*(-3 + 11*psi) + phi^3*(3 + (23 - 139*psi)*psi) + 
       phi^2*psi*(-7 + (56 - 139*psi)*psi) + phi*psi^2*(-7 + psi*(23 + 11*psi)))*zeta^4 + 
     (phi^4*(1 - 4*psi) + psi^2*(1 + (-8 + psi)*psi) + 2*phi^3*(-4 - 9*psi + 66*psi^2) - 2*phi*psi*(1 + psi*(-8 + psi*(9 + 2*psi))) + 
       phi^2*(1 + 2*psi*(8 + psi*(-47 + 66*psi))))*zeta^5)
 
    a3 = 2*phi^2*psi^9*zeta*(2*eta^4*phi^3*psi^3 + 2*eta^3*phi^2*psi^2*(-(phi*(-6 + psi)) + 6*psi)*zeta - 
     eta^2*phi*psi*(phi + psi)*(-11*psi + phi*(-11 + 49*psi))*zeta^2 + 
     2*eta*(phi*(7 - 2*psi)*psi^2 - psi^3 + phi^2*psi*(-1 + 3*psi)*(-7 + 6*psi) + phi^3*(-1 + 2*psi*(-1 + 9*psi)))*zeta^3 - 
     (-((-1 + psi)*psi^2) + phi^3*(-1 + 3*psi) + phi^2*(1 + 2*(8 - 15*psi)*psi) + phi*psi*(-4 + psi*(16 + 3*psi)))*zeta^4)
 
    a4 = phi^2*psi^10*zeta^2*(-8*eta^2*phi^2*psi^2 + 4*eta*phi*psi*(-4*psi + phi*(-4 + 9*psi))*zeta + 
     (psi^2 + 2*phi*psi*(-5 + 3*psi) + phi^2*(1 + 3*(2 - 9*psi)*psi))*zeta^2)
 
    a5 = 4*phi^3*psi^12*zeta^3

    poly = Polynomial([a0, a1, a2, a3, a4, a5])
    rts = roots(poly)
    res = filter(x -> imag(x) == 0 && real(x) >= 0, rts) .|> v -> sqrt(real(v)) 
    res = unique(sort(res))
    if length(res) == 2
        [[res[1], res[2]]]
    elseif length(res) == 3
        [[0., res[1]], [res[2], res[3]]]
    elseif length(res) == 4
        [[res[1], res[2]], [res[3], res[4]]]
    elseif length(res) == 4
        [[0., res[1]], [res[2], res[3]], [res[4], res[5]]]
    else
        error("Too many edges of the bulk!")
    end
end

function eq_q(alpha, beta, rho, Delta, q)
    m20 = 1
    hq_normalised = alpha / (Delta + m20 - q)   # hq * rho * (1+beta)

    R1 = beta    # max(1,beta) / min(1,beta)
    R2 = rho   # rho / min(1,beta)
    R_noise = 1 / hq_normalised

    mu(x) = stieltjes(R1, R2, R_noise, x)[1]
    bulks = pdf_edges(R1, R2, R_noise)

    int1 = R1 == 1 ? 0. : sum( 2 * quadgk(x -> mu(x)/x^2, bulk[1], bulk[2])[1] for bulk in bulks)
    int2 = sum( 2 * quadgk(x -> mu(x)^3, bulk[1], bulk[2])[1] for bulk in bulks)

    m20 - R_noise + R_noise^2 / R1^(3/2) * ( (R1-1)^2 * int1 + 4 * pi^2 / 3 * int2)
end

function iterate_q(alpha, beta, rho, Delta; init = 1e-3, m = 5, b = 0.01, ftol = 1e-8, iters = 100)
    if alpha == 0
        return 0.
    end

    function updater!(F, x)
        F[1] = eq_q(alpha, beta, rho, Delta, x[1])
    end
    r = fixedpoint(updater!, [init], iterations = iters, ftol=ftol, m=m, beta=b).zero[1]
    r
end

function free_entropy(alpha, beta, rho, Delta, hq)

    R1 = beta    # max(1,beta) / min(1,beta)
    R2 = rho   # rho / min(1,beta)
    R_noise = 1 / hq
    mu(x) = stieltjes(R1, R2, R_noise, x)[1]

    a, b = if R1 == 1
        (1e-10, 50)
    else
        edges = pdf_edges(R1, R2, R_noise)
        (minimum(edges), maximum(edges))
    end

    int1 = 2 * hcubature(x -> mu(x[1]) * mu(x[2]) * (abs(x[1] - x[2]) < 1e-8 ? 0. : log(abs(x[1]^2 - x[2]^2))), [a, a], [b, b], maxevals = 20000)[1]

    int2 = if beta == 1
        0.
    else
        2 * quadgk(x -> mu(x) * (x < 1e-8 ? 0. : log(x)), a, b)[1]
    end
   
    (alpha - 1) /2 * log(hq) - hq * Delta / 2 - 1/beta * int1 - (beta - 1)/beta * int2
end

function free_entropy_vs_overlap(alpha, beta, rho, Delta, q)

    hq = alpha / (Delta + 1 - q)

    R1 = beta    # max(1,beta) / min(1,beta)
    R2 = rho   # rho / min(1,beta)
    R_noise = 1 / hq
    mu(x) = stieltjes(R1, R2, R_noise, x)[1]

    a, b = if R1 == 1
        (1e-10, 50)
    else
        edges = pdf_edges(R1, R2, R_noise)
        # println(edges)
        (minimum(vcat(edges...)), maximum(vcat(edges...)))
    end

    int1 = 2 * hcubature(x -> mu(x[1]) * mu(x[2]) * (abs(x[1] - x[2]) < 1e-8 ? 0. : log(abs(x[1]^2 - x[2]^2))), [a, a], [b, b], maxevals = 10000)[1]

    int2 = if beta == 1
        0.
    else
        2 * quadgk(x -> mu(x) * (x < 1e-8 ? 0. : log(x)), a, b)[1]
    end
   
    (alpha - 1) /2 * log(hq) - hq * Delta / 2 - 1/beta * int1 - (beta - 1)/beta * int2
end

function ridge(gamma, lambda, delta)

    if abs(lambda - delta) < 1e-8
        mse = 1 - (2 * gamma) / (sqrt(2 * (gamma + 1) * delta + (gamma - 1)^2 + delta^2) + gamma + delta + 1) 
        return mse
    else
        p = gamma + lambda
        c = gamma - lambda
        t = sqrt((p - 1)^2 + 4 * lambda)

        m = 2 * gamma / (p + t + 1) 
        q = 4 * gamma * (gamma * (p + t - 3) + (1 + delta) * (p + t + 1)) / (p + t + 1) / (p^2 - 2 * c + t^2 + 2 * t * (p + 1) + 1)
        mse = 1 - 2*m + q
        return mse
    end
end

function low_rank_solution(beta, alpha, delta; init = [0.5, 0.5, 0.5], tol = 1e-10, maxsteps = 1000, damping = 0.5)
    ga = init[1]
    gb = init[2]
    hq = init[3]

    error = 200

    for i in 1:maxsteps
        newhq = (1-damping) * alpha / (delta + 1 - ga * gb) + damping * hq

        newga = ((beta+1)^2 * newhq^2 - beta)/((beta+1) * newhq * (beta * newhq + newhq + 1))
        newgb = ((beta+1)^2 * newhq^2 - beta)/((beta+1) * newhq * (beta * newhq + newhq + beta))

        error = norm([ga, gb, hq] .- [newhq, newga, newgb])

        ga = newga 
        gb = newgb
        hq = newhq

        if error < tol
            break
        end
    end

    [ga * gb, error]
end

function eqs_larbebeta(q, alpha, rho, Delta)
    d = (1+Delta-q) / alpha
    -q + 1 - d + d/2 * (
        sqrt(
            1 + rho * ((d+2)*rho*d + 2d + rho - 2)
        ) 
        - (rho-1)
        - d*rho
    )
end

function iterate_q_larbebeta(alpha, rho, Delta)
    if alpha ==0 
        return 0.
    end
    find_zero(q -> eqs_larbebeta(q[1], alpha, rho, Delta), 0.5)
end

function low_rank_solution_largebeta(alpha, delta)
    0.5 * (1 + delta + alpha - sqrt((alpha-1)^2 +2 * (alpha+1) * delta + delta^2))
end

function MNNE_integralP(gamma, x, k)
    gamma_plus = (1 + sqrt(gamma))^2
    gamma_minus = (1 - sqrt(gamma))^2


    f(t) = gamma_minus < t < gamma_plus ? t^(k-1) * sqrt((gamma_plus - t) * (t - gamma_minus)) : 0
    1/(2pi * gamma) * quadgk(f, x, gamma_plus)[1]
end

function MNNE_equation(rho, beta)
    beta = 1/beta
    gamma = beta * (1-rho) / (1 - rho * beta)
    gamma_plus = (1 + sqrt(gamma))^2
    
    f(lambda) = rho + beta * rho - beta * rho^2 + (1-rho * beta) * (
                    rho * lambda^2 + (1 - rho) * (
                        MNNE_integralP(gamma, lambda^2, 1) 
                        - 2 * lambda * MNNE_integralP(gamma, lambda^2, 1/2) 
                        + lambda^2 * MNNE_integralP(gamma, lambda^2, 0)
                    )
                )


    g(lambda) = MNNE_integralP(gamma, lambda^2, 1/2) - lambda *  MNNE_integralP(gamma, lambda^2, 0) - lambda * rho / (1-rho)

    myrho = rho
    mybeta = 1/beta

    bo = min(1, myrho/mybeta * (1 + mybeta - myrho))
    # bo = min(1, rho/mybeta * (1 + mybeta - rho))

    # try
        # println((g(sqrt(gamma_minus)), g(sqrt(gamma_plus))))
        sol = find_zero(g, (0,gamma_plus))
        # println([rho, beta, f(sol), rho/beta * (1 + beta - rho)])
        [f(sol), bo]
    # catch
    #     [0, bo]
    # end
end
