using RadiiPolynomial, MAT, IntervalArithmetic, LinearAlgebra, MATLAB, JLD2

# Versions:
# Julia: v1.10.4

# Libraries:
# RadiiPolynomial v0.8.24
# MAT v0.10.7
# IntervalArithmetic v0.22.36
# MATLAB v0.9.0
# JLD2 v0.5.15

include("list_functions.jl") # Necessary functions for the proof
include("matproducts.jl") # Functions for matrix computations using Intlab (MATLAB)
# Ensure to use the correct path in matproducts.jl to Intlab

#########################################################################################################
##  Part of this code is based on the code in https://github.com/matthieucadiot/LocalizedPatternSH.jl  ##
#########################################################################################################

@time begin # Measure the computation time

    #################### Initialization ####################
    N01 = 100 # Sequence truncation dimension, N0>=N
    N02 = 60 

    # Initialize parameters to define how large the finite domain is
    d1 = interval(pi) / interval(0.06)
    d2 = interval(pi) / interval(0.2)

    N1 = 60# Operator truncation dimension
    N2 = 60 
    nu1 = interval(1.1) # Used for computing aliasing error
    nu2 = interval(1.1) 
    Nfft1 = 512 # Should be powers of 2
    Nfft2 = 512 
    c = interval(1.3) # Wave speed

    #################### Newton ####################

    d1float = π / 0.06 # No need to work with intervals for improving the solution using Newton's method
    d2float = π / 0.2
    cfloat = 1.3

    fourier0float = CosFourier(N01, π / d1float) ⊗ CosFourier(N02, π / d2float)
    # Import 101x61 approximate solution in Fourier coefficients for SB
    u = load("useq_c13.jld2", "uNewton")

    ucoeff = reshape(coefficients(u), 101, 61)

    # u = ucoeff[1:N01+1, 1:N02+1] # If you want to make approximation smaller
    # If you want to pad approximation with zeros, uncomment three lines below
    upad = zeros(N01+1,N02+1) 
    upad[1:size(ucoeff, 1), 1:size(ucoeff, 2)] = ucoeff
    u = upad

    US = Sequence(fourier0float, vec(u))

    FN = F(US, N01, N02, Nfft1, Nfft2, d1float, d2float, cfloat)
    DFN = DF(US, N01, N02, Nfft1, Nfft2, d1float, d2float, cfloat)

    uNewton, successN = newton(uN -> (F(uN, N01, N02, Nfft1, Nfft2, d1float, d2float, cfloat), DF(uN, N01, N02, Nfft1, Nfft2, d1float, d2float, cfloat)), US, maxiter=10)
    display("Did Newton converge?")
    display(successN)

    ## Plot the approximation (using MATLAB)
    PlotCoeffs2D(uNewton,-d1float,d1float,-d2float,d2float)

    ## Include interval arithmetic for converged approximation
    fourier = CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2) #N is truncation dimension operators
    fourier0 = CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2) #N0>=N is truncation dimension sequences

    UC = coefficients(uNewton)
    UC = reshape(UC, N01 + 1, N02 + 1)
    UC = coefficients(Sequence(fourier0, vec(UC)))

    ∂1 = project(Derivative((2, 0)), fourier0, fourier0, Interval{Float64})
    ∂2 = project(Derivative((0, 2)), fourier0, fourier0, Interval{Float64})
    Δ = copy(∂1)
    radd!(Δ, ∂2)

    L = LinearOperator(fourier0, fourier0, Diagonal(diag(coefficients(Δ .^ 2 + c^2 * ∂1))))
    Δ = Nothing
    L = convert(Vector{Interval{Float64}}, diag(coefficients(I + L)))
    Li = ones((N01 + 1) * (N02 + 1)) ./ L

    ##### Compute the projection into zero trace functions
    setprecision(80)
    Li_trace = interval.(big.(mid.(Li) .^ 2))
    S = trace(N01, N02, 80)
    C = S'
    W = UC - (Li_trace .* C) * solve_linear(S * (Li_trace .* C), S * UC, 80)

    W = interval.(Float64.(inf.(W), RoundDown), Float64.(sup.(W), RoundUp))
    US = Sequence(fourier0, vec(W))

    S = Nothing
    C = Nothing
    Li_trace = Nothing

    #################### Computation of the constants needed for computing the aliasing error in the bounds ####################

    rhobar1 = log(nu1)
    rhobar2 = log(nu2)

    delta1 = interval(Float64.(inf.(0), RoundDown), Float64.(sup.(π / Nfft1), RoundUp))
    delta2 = interval(Float64.(inf.(0), RoundDown), Float64.(sup.(π / Nfft2), RoundUp))

    # Create the grid for n₁ and n₂
    n1a = -Nfft1:(Nfft1-1)
    n2a = -Nfft2:(Nfft2-1)

    # Create ndgrid equivalent
    n1a_grid = repeat(n1a, 1, length(n2a))  # n₁ repeated across rows
    n2a_grid = repeat(n2a', length(n1a), 1)  # n₂ repeated across columns

    power = n1a_grid .* rhobar1 .+ n2a_grid .* rhobar2 .+ im .* (n1a_grid .* delta1 .+ n2a_grid .* delta2)

    upad = interval.(zeros(Nfft1 + 1, Nfft2 + 1))

    # Place original Fourier coefficients in the top-left corner of upad
    UC = reshape(coefficients(US), (N01 + 1, N02 + 1)) # Extract coefficients
    upad[1:size(UC, 1), 1:size(UC, 2)] = UC

    ## Making the cosine approximation symmetric
    # Extract submatrices
    sub1 = upad[2:end, 2:end]
    sub2 = upad[2:end, 1:end-1]
    sub3 = upad[1:end-1, 2:end]
    sub4 = upad[1:end-1, 1:end-1]

    # Apply transformations
    rotated_sub1 = rot180(sub1)
    flipped_sub2 = reverse(sub2, dims=1)
    flipped_sub3 = reverse(sub3, dims=2)

    # Combine the submatrices
    u_sympad = [rotated_sub1 flipped_sub2; flipped_sub3 sub4]

    ubar_delta = u_sympad .* exp.(power)

    # Inverse Fourier transform
    U_delta = rifft!(ubar_delta, CosFourier(Nfft1, interval(pi) / d1) ⊗ CosFourier(Nfft2, interval(pi) / d2))

    # For Y0 we only use C, but Cprime and Cdoubleprime will be used for the other bounds
    C = 1 / (4 * Nfft1 * Nfft2) * sum(abs.(exp.(U_delta) .- U_delta .- 1))
    Cprime = 1 / (4 * Nfft1 * Nfft2) * sum(abs.(exp.(U_delta) .- 1))
    Cdoubleprime = 1 / (4 * Nfft1 * Nfft2) * sum(abs.(exp.(U_delta)))

    #################### Computing the bounds ####################

    #################### Y0 bound ####################

    # Construct BN
    ∂1 = project(Derivative((2, 0)), fourier, fourier, Interval{Float64})
    ∂2 = project(Derivative((0, 2)), fourier, fourier, Interval{Float64})
    Δ = copy(∂1)
    radd!(Δ, ∂2)

    L = LinearOperator(fourier, fourier, Diagonal(diag(coefficients(Δ .^ 2 + c^2 * ∂1))))
    Δ = Nothing
    L = convert(Vector{Interval{Float64}}, diag(coefficients(I + L)))
    Li = ones((N1 + 1) * (N2 + 1)) ./ L

    Ufft = fft(US, (Nfft1, Nfft2)) # Apply FFT

    # First derivative of G
    Gfftder = exp.(Ufft) .- 1
    DG = rifft!(Gfftder, CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2))
    DG = project(Multiplication(DG), fourier, fourier, Interval{Float64})

    # Construction of B^N, the approximate inverse of (the finite projection) DF(U0)
    BN = interval.(inv(I + mid.(DG) .* mid.(Li)'))

    ## Construct F (finitely many terms)
    F0 = F(US, N01, N02, Nfft1, Nfft2, d1, d2, c)
    FN = project(F0, fourier)

    ## First term Y0 bound
    # Y1 including aliasing: 
    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N1, n2 in 0:N2]
    Y1 = norm(BN * FN + C * Sequence(fourier, interval.(-inf.(vec(ε_vals)), sup.(vec(ε_vals)))), 2)^2

    ## Second term Y0 bound
    # Y2 including aliasing:
    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N01, n2 in 0:N02]
    ε_valsSeq0 = Sequence(fourier0, interval.(-inf.(vec(ε_vals)), sup.(vec(ε_vals))))
    ε_valsSeq = project(ε_valsSeq0, fourier)
    AuxSeq = F0 - project(F0, fourier)
    Y2 = norm(AuxSeq + C * (ε_valsSeq0 - ε_valsSeq), 2)^2

    ## Third term Y0 bound
    Y3 = C^2 * ((2 * nu1^(-2 * N01 - 2) * (1 + nu2^(-2)) +
                 2 * nu2^(-2 * N02 - 2) * (1 + nu1^(-2)) -
                 4 * nu1^(-2 * N01 - 2) * nu2^(-2 * N02 - 2)) /
                ((1 - nu1^(-2)) * (1 - nu2^(-2))))

    ## Total Y0 bound
    #Y0 = (d1 + d2) * sup(sqrt(Y1 + Y2 + Y3))
    Y0 = sqrt(2*d1) * sup(sqrt(Y1 + Y2 + Y3))
    display("The value of Y0 is $Y0")


    #################### Z2 bound ####################

    # Construct B matrix and its norm
    # Construction of the conversion operators from cosine to exponential and vice versa. In particular D_1 has terms (√α_n)_n on the diagonal
    # and D2 has terms  (1/√α_n)_n on the diagonal. 
    D1 = convert(Vector{Interval{Float64}}, exp2cos(N1,N2))
    D2 = ones((N1 + 1) * (N2 + 1)) ./ D1

    # Computation of the norm of B
    Bconj = LinearOperator(fourier, fourier, coefficients(BN)') # Construction of the adjoint
    norm_B = maximum([1 (spec_norm(coefficients(D1 .* BN .* D2')))])

    ## Finite part of the 1-norm of the exponential (similar to computation Y3)
    expUfft = exp.(Ufft)
    expUlarge = rifft!(expUfft, CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2))

    expUcoeff = reshape(coefficients(expUlarge), (N01 + 1, N02 + 1))

    # Compute norm including defect
    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N01, n2 in 0:N02]
    expUfinite = norm(expUlarge + Cdoubleprime * Sequence(fourier0, interval.(-inf.(vec(ε_vals)), sup.(vec(ε_vals)))), 1)

    ## Infinite part of the 1-norm of the exponential (similar to Y3, but no squares)
    expUinfinite = Cdoubleprime * ((2 * nu1^(-N01 - 1) * (1 + nu2^(-1)) +
                                    2 * nu2^(-N02 - 1) * (1 + nu1^(-1)) -
                                    4 * nu1^(-N01 - 1) * nu2^(-N02 - 1)) /
                                   ((1 - nu1^(-1)) * (1 - nu2^(-1))))

    # Total 1-norm of exponential
    expUnorm = expUfinite + expUinfinite

    kappa1 = 1 / (1 - c^4 / 4)

    expUfft = exp.(Ufft)
    expUN0 = rifft!(expUfft, CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2))

    expUfft = exp.(Ufft)
    expUN = rifft!(expUfft, CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2))

    # US --> exp.(Ubar)
    expMUN = project(Multiplication(expUN), CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2), CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2))
    term1 = spec_norm((_matprod(D1 .* coefficients(expMUN),coefficients(Bconj) .* D2')))

    ## 1-norm of e^{Ubar} - e^{U^N_0}
    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N01, n2 in 0:N02]
    ε_valsSeq0 = Sequence(fourier0, vec(ε_vals))
    ε_valsSeq = project(ε_valsSeq0, fourier)
    ε_valsSeq0 = interval.(-inf.(ε_valsSeq0), sup.(ε_valsSeq0))
    ε_valsSeq = interval.(-inf.(ε_valsSeq), sup.(ε_valsSeq))
    AuxSeq = expUN0 - expUN
    expUnormfinite = norm(AuxSeq + Cprime * (ε_valsSeq0 - ε_valsSeq), 1)

    term2 = norm_B * expUnormfinite

    maxnorm_Bexp = maximum([term1 + term2 1])

    kappa2 = compute_kappa2(0.1, 20, 20)

    rtest = 1e-4 # Test value for r with which we compute the preliminary Z2 bound

    Z2test = sup((kappa1 * (exp.(kappa2 * rtest) - 1) / rtest) * maxnorm_Bexp)
    display("For rtest = $rtest, the value of Z2 is $Z2test")


    #################### 𝒵1 bound ####################
    ## 1-norm of Vbar-Vbarᴺ 
    Gfftder = exp.(Ufft) .- 1 # Note that we have to redefine Gfftder as rifft! changes the argument
    DG = rifft!(Gfftder, CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2)) # Create DG as vector instead of linear operator
    Gfftder = exp.(Ufft) .- 1 # Note that we have to redefine Gfftder as rifft! changes the argument
    DG_N0 = rifft!(Gfftder, CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2))

    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N01, n2 in 0:N02]
    ε_valsSeq0 = Sequence(fourier0, vec(ε_vals))
    ε_valsSeq = project(ε_valsSeq0, fourier)
    ε_valsSeq0 = interval.(-inf.(ε_valsSeq0), sup.(ε_valsSeq0))
    ε_valsSeq = interval.(-inf.(ε_valsSeq), sup.(ε_valsSeq))
    AuxSeq = DG_N0 - DG
    VbarNnormfinite = norm(AuxSeq + Cprime * (ε_valsSeq0 - ε_valsSeq), 1)

    # Include infinite parts
    VbarN0norminfinite = Cprime * ((2 * nu1^(-N01 - 1) * (1 + nu2^(-1)) +
                                  2 * nu2^(-N02 - 1) * (1 + nu1^(-1)) -
                                  4 * nu1^(-N01 - 1) * nu2^(-N02 - 1)) /
                                 ((1 - nu1^(-1)) * (1 - nu2^(-1))))

    VbarNnorminfinite = Cprime * ((2 * nu1^(-N1 - 1) * (1 + nu2^(-1)) +
                                 2 * nu2^(-N2 - 1) * (1 + nu1^(-1)) -
                                 4 * nu1^(-N1 - 1) * nu2^(-N2 - 1)) /
                                ((1 - nu1^(-1)) * (1 - nu2^(-1))))

    VbarNnorm = VbarNnormfinite + (VbarN0norminfinite - VbarNnorminfinite)


    ## Z1 bound
    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N1, n2 in 0:N2]

    Ni1 = interval(N1 + 1) * interval(pi) / d1
    Ni2 = interval(N2 + 1) * interval(pi) / d2
    # The formula below is valid if N2+1>d2*c/(pi*sqrt(2)), which is fine if N2 is big enough
    lN = interval(1) / maximum([(Ni1)^4 - c^2 * Ni1^2 + interval(1) (Ni2)^4 + interval(1)])

    # Vbar = e^ubar - I + defect
    Gfftder = exp.(Ufft) .- 1 # Note that we have to redefine Gfftder as rifft! changes the argument
    Vbar = rifft!(Gfftder, CosFourier(N1, interval(pi) / d1) ⊗ CosFourier(N2, interval(pi) / d2)) + Sequence(fourier, vec(Cprime * interval.(-inf.(ε_vals), sup.(ε_vals))))

    M1 = coefficients(project(Multiplication(Vbar * Vbar), fourier, fourier, Interval{Float64})) - _matprod(coefficients(project(Multiplication(Vbar), fourier, fourier, Interval{Float64})), coefficients(project(Multiplication(Vbar), fourier, fourier, Interval{Float64})))
    n2 = spec_norm((lN^2 * D1 .* _matprod(_matprod(coefficients(BN),M1),coefficients(Bconj)) .* D2'))
    n1 = spec_norm(((I - D1 .* _matprod(coefficients(BN), coefficients(I + project(Multiplication(Vbar), fourier, fourier, Interval{Float64}) .* Li')) .* D2')))

    Z1 = sqrt(n1^2 + n2 + lN^2 * (norm(Vbar, 1) + VbarNnorminfinite)^2 + spec_norm(((D1 .* Li) .* M1 .* (Li .* D2)')))

    display("The value of Z1 is $Z1")

    ## 𝒵ᵤ bound
    Zu = computation_Zu(Vbar, N1, N2)

    ########################################

    ## Total 𝒵₁ bound
    tM1 = interval(N01) * interval(pi) / interval(d1)
    tM2 = interval(N02 + 1) * interval(pi) / interval(d2)
    C0 = interval(1) / ((tM2 + interval(1))^4)

    𝒵1 = Z1 + norm_B * Zu + norm_B * C0 * VbarNnorm

    display("The value of Z1 is $Z1")
    display("The value of 𝒵u is $Zu")
    display("The value of 𝒵1 is $𝒵1")


    #################### Proof ####################

    ######### Choice for r0 and definition of Z2
    rmin = (1 - sup(𝒵1) - sqrt((1 - sup(𝒵1))^2 - 2 * sup(Y0) * Z2test)) / Z2test
    r0 = interval(rmin) # Obtain a candidate value for r0
    Z2 = sup((kappa1 * (exp.(kappa2 * r0) - 1) / r0) * maxnorm_Bexp)  # We can compute Z2(r0) once r0 is fixed
    display("For r0 = $r0, the value of Z2 is $Z2")

    Z2 = interval.(Float64.(inf.(Z2), RoundDown), Float64.(sup.(Z2), RoundUp))
    Z2 = sup(Z2)
    
    𝒵1 = interval.(Float64.(inf.(𝒵1), RoundDown), Float64.(sup.(𝒵1), RoundUp))
    𝒵1 = sup(𝒵1)
    r0 = Float64(sup(r0))
    Y0 = sup(Y0)

    ## Validation of the proof  
    if 𝒵1 + Z2 * r0 < 1
        if 1 / 2 * Z2 * r0^2 - (1 - 𝒵1) + Y0 < 0
            display("The proof was successful for r0 = ")
            display(r0)
        else
            display("Failure: discriminant is negative")
        end
    else
        display("Failure: linear term is positive")
    end

    ## Stability
    
    δ0 = interval(0.01) ; δ0big = interval(big(0.01))

    #defect with one norm                                 

    ## this is the defect with the finite truncation of e^{U}

    err_Vbar = VbarNnorminfinite
    λmin = -exp(kappa2*r0)*(norm(Vbar-interval(1),1) + err_Vbar) - interval(0.25)*c^4

    t = -λmin + interval(0.0001)
    t = interval.(Float64.(inf.(t), RoundDown), Float64.(sup.(t), RoundUp))
    t = sup(t)

    ##### computation of Zu

    𝒵u1, 𝒵u2 = computation_Zu_stability(Vbar, δ0, λmin ,N1, N2)

    𝒵u1 = interval(2)*𝒵u1 + err_Vbar/(interval(1)-δ0-interval(0.25)*c^4) ; 𝒵u2 = interval(2)*𝒵u2 + err_Vbar/(interval(1)-δ0-interval(0.25)*c^4)  #### the Zu bound for stability is essentially 2 times the one for the existence proof 

    stability(US,Vbar,Ufft,c,d1,d2,𝒵1,𝒵u1,𝒵u2,Z2,r0,t,δ0)

end

