using Random, Plots, Polynomials, LinearAlgebra, QuadGK, NLsolve, Roots, Statistics, HCubature
using DataFrames, CSV, StatsPlots

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

# same as above
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

function se_eq_q(beta, rho, hq)

    R1 = beta    # max(1,beta) / min(1,beta)
    R2 = rho   # rho / min(1,beta)
    R_noise = 1 / hq

    mu(x) = stieltjes(R1, R2, R_noise, x)[1]
    bulks = pdf_edges(R1, R2, R_noise)

    int1 = R1 == 1 ? 0. : sum( 2 * quadgk(x -> mu(x)/x^2, bulk[1], bulk[2])[1] for bulk in bulks)
    int2 = sum( 2 * quadgk(x -> mu(x)^3, bulk[1], bulk[2])[1] for bulk in bulks)

    1 - R_noise + R_noise^2 / R1^(3/2) * ( (R1-1)^2 * int1 + 4 * pi^2 / 3 * int2)
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

# normalised MMSE ||Shat - S*||^2 / sqrt(d1 * d2) for the BO denoiser
function denoising_MMSE(R1, R2, R_noise)
    mu(x) = stieltjes(R1, R2, R_noise, x)[1]
    
    bulks = pdf_edges(R1, R2, R_noise)

    int1 = R1 == 1 ? 0. : sum( 2 * quadgk(x -> mu(x)/x^2, bulk[1], bulk[2])[1] for bulk in bulks)
    int2 = sum( 2 * quadgk(x -> mu(x)^3, bulk[1], bulk[2])[1] for bulk in bulks)
    
    R_noise - R_noise^2 / sqrt(R1) * ((R1-1)^2/R1 * int1 + 4 * pi^2/3/R1 * int2)
end


# denoiser. Takes Y = (A B / sqrt(r) + sqrt(noise) Z) / sqrt[4](d1 * d2) as above and denoises it to get back S = A B / sqrt(r) / sqrt[4](d1 * d2)
# R2 = r / min(d1,d2)
function denoiser(Y, R2, R_noise)
    d1, d2 = size(Y)
    R1 = max(d1,d2)/min(d1,d2)
    vel(x) = stieltjes(R1, R2, R_noise, x)[2]
    
    U, S, V = svd(Y) # such that Y ≈ U * Diagonal(S) * V'
    newS = map(s -> s - 2 * R_noise / sqrt(R1) * ((R1 - 1)/(2s) + vel(s)), S)
    U * Diagonal(newS) * V'
end

function datamodel_sensing(d1, beta, rho, alpha, delta; rng = Random.default_rng())
    d2 = round(beta * d1) |> Int
    r = round(rho * d1) |> Int
    N = round(alpha * (d1*d2)) |> Int
    
    X = randn(rng, (N, d1, d2))
    S = randn(rng, (d1,r)) * randn(rng, (r,d2)) / sqrt(r) 
    h = [ sum(X[mu, i, j] * S[i, j] for i in 1:d1, j in 1:d2) for mu in 1:N] / sqrt(d1 * d2)
    y = h + sqrt(delta) * randn(rng, N)
    (X, y, S)
end

function AMP(X, y, Strue, rho, delta; rng = Random.default_rng(), maxT = 50, tol = 1e-4, scale = 5., init_q = 0., damping = 0., reg_delta = 0., verbose = false)
 
    # set dimensions
    n, d, L = size(X)
    r = round(d * rho) |> Int

    # adapt instance
    Strue = Strue / (d*L)^(1/4) # normalise to the S_ convention

    # generate initialisation
    Srnd = randn(rng, (d,r)) * randn(rng, (r,L)) / sqrt(r) / (d*L)^(1/4)
    Sinit = (1-init_q) * Srnd + init_q * Strue

    # generate init for AMP parameters
    uX = X / (d*L)^(1/4)
    hatS    = Sinit
    hatC    = scale
    omega   = scale * ones(n)
    V       = scale  

    iter_error = Inf
    mse_error = Inf
    iter = maxT

    # define AMP functions
    gOut(y, w, V) = (y-w)/(delta + reg_delta + V)
    del_gOut(y, w, V) = -1/(delta + reg_delta + V)

    # iterate AMP and state evolution
    for t in 1:maxT
        # update V, omega
        newV = hatC
        newOmega = [ sum(uX[mu, i, j] * hatS[i, j] for i in 1:d, j in 1:L) - gOut(y[mu], omega[mu], V) * newV for mu in 1:n ]
        
        # update V, omega with damping
        V = newV * (1-damping) + V * damping
        omega = newOmega * (1-damping) + omega * damping
        
        # update A, R
        A = sum( gOut(y[mu], omega[mu], V)^2 for mu in 1:n) * alpha / n
        # A = - sum( del_gOut(y[mu], omega[mu], V) for mu in 1:n) * alpha / n
        R = hatS + 1 / (A * sqrt(d * L))  * sum(gOut(y[mu], omega[mu], V) * uX[mu,:,:] for mu in 1:n)

        # update S, C
        newhatS = denoiser(R, rho, 1/A)
        hatC = denoising_MMSE(beta, rho, 1/A) 
        
        # compute iteration error
        iter_error = norm(hatS - newhatS)^2 / sqrt(d * L)
        hatS = newhatS

        # compute mse error
        mse_error = norm(hatS - Strue)^2 / sqrt(d * L) 

        verbose ? println(
        (
            t = t,
            iter_error = iter_error,
            mse_iter = mse_error,
        )
        ) : nothing

        if iter_error < tol
            iter = t
            break
        end
    end

    return (hatS * (d * L)^(1 / 4), error, iter)
end


# Assert that the number of arguments is correct
if length(ARGS) != 7
    println("Usage: julia main.jl delta beta rho alpha d1")
    exit(1)
end

# Read from command line
delta = parse(Float64, ARGS[1])
beta = parse(Float64, ARGS[2])
rho = parse(Float64, ARGS[3])
alpha = parse(Float64, ARGS[4])
d = parse(Int, ARGS[5])
tol = parse(Float64, ARGS[6])
damping = parse(Float64, ARGS[7])


samples = 1
L = round(beta * d) |> Int
r = round(rho * d) |> Int
n = round(alpha * (d * L)) |> Int

println("Running with parameters: delta = ", delta, ", beta = ", beta, ", rho = ", rho, ", alpha = ", alpha, ", d = ", d, ", L = ", L, ", r = ", r, ", n = ", n)
println("Theory MMSE is ", 1 - iterate_q(alpha, beta, rho, delta))

# Repeat the experiment "samples" times
mse_amp = zeros(samples)

for i in 1:samples
    X, y, Strue = datamodel_sensing(d, beta, rho, alpha, delta)
    hatS, convergence, iter = AMP(X, y, Strue, rho, delta, maxT = 500, tol = tol, scale = 20., init_q = 0., damping = damping, reg_delta = 0., verbose = true)
    mse_amp[i] = norm(hatS - Strue)^2 / (d * L)

    println("Sample ", i, " converged in ", iter, " iterations to iteration error ", convergence)
    println("MSE achieved is ", mse_amp[i])
    println()
end

# Save to file named AMP_rho_beta_delta_alpha_d1.csv
filename = join([
        "AMP_paper_Ema/AMP",
        string(rho),
        string(beta),
        string(delta),
        string(alpha),
        string(d)
    ], "_") * ".csv"


df = DataFrame(mse_amp=mse_amp)
CSV.write(filename, df)

println("Results saved to ", filename)


