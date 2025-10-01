# Function F
function F(uN, N01, N02, Nfft1, Nfft2, d1, d2, c)
    fourier0 = CosFourier(N01, pi / d1) ⊗ CosFourier(N02, pi / d2) # N0>=N is truncation dimension sequences

    ##### Linear part
    ∂1 = project(Derivative((2, 0)), fourier0, fourier0)
    ∂2 = project(Derivative((0, 2)), fourier0, fourier0)
    Δ = copy(∂1)
    radd!(Δ, ∂2)

    L = LinearOperator(fourier0, fourier0, Diagonal(diag(coefficients(Δ .^ 2 + c^2 * ∂1))))
    Δ = Nothing
    L = convert(Vector, diag(coefficients(I + L)))

    ##### Nonlinear part
    Ufft = fft(uN, (Nfft1, Nfft2)) # Apply FFT
    Gfft = exp.(Ufft) .- Ufft .- 1
    G = rifft!(Gfft, CosFourier(N01, pi / d1) ⊗ CosFourier(N02, pi / d2))

    ##### Construct F (finitely many terms)
    F0 = L .* uN + G
    return project(F0, fourier0)
end

# Function DF
function DF(uN, N01, N02, Nfft1, Nfft2, d1, d2, c)
    fourier0 = CosFourier(N01, pi / d1) ⊗ CosFourier(N02, pi / d2) # N0>=N is truncation dimension sequences

    ##### Linear part
    ∂1 = project(Derivative((2, 0)), fourier0, fourier0)
    ∂2 = project(Derivative((0, 2)), fourier0, fourier0)
    Δ = copy(∂1)
    radd!(Δ, ∂2)

    L = LinearOperator(fourier0, fourier0, Diagonal(diag(coefficients(I + Δ .^ 2 + c^2 * ∂1))))
    Δ = Nothing

    ##### Nonlinear part
    Ufft = fft(uN, (Nfft1, Nfft2)) # Apply FFT
    Gfft = exp.(Ufft) .- 1
    G = rifft!(Gfft, CosFourier(N01, pi / d1) ⊗ CosFourier(N02, pi / d2))

    ##### Construct and return DF (finitely many terms)
    return L + project(Multiplication(G), fourier0, fourier0)
end

# Function epsilon
function ε(n1, n2, nu1, nu2, Nfft1, Nfft2)
    numerator = 2 * nu1^abs(n1) * nu2^abs(n2) * (nu1^(-2 * Nfft1) + nu2^(-2 * Nfft1))
    denominator = (1 - nu1^(-2 * Nfft1)) * (1 - nu2^(-2 * Nfft2))
    return numerator / denominator
end

# Function to plot the solution (on a finite domain) using MATLAB
function PlotCoeffs2D(U0, a, b, c, d)
    # U0 is a sequence in 2D
    # a,b,c,d are the endpoints of the interval [a,b] × [c,d]

    y1 = a:0.1:b
    y2 = c:0.1:d

    m1 = length(y1)
    m2 = length(y2)

    U = complex(zeros(m1, m2))

    for b₁ = 1:m1
        for b₂ = 1:m2
            U[b₁, b₂] = complex((U0(y1[b₁], y2[b₂])))[1]
        end
    end

    U = real.(U')
    # Plotting in MATLAB:
    mat" 

    h = surf($y1,$y2,$U)
    view([-10.5 13])
    xlabel('x1')
    ylabel('x2')
    zlabel('U')

    set(h,'LineStyle','none')"
end

# Function to compute the projection into trace zero functions
function trace(N1, N2, p) # Adjusted one-dimensional function from https://github.com/matthieucadiot/LocalizedPatternSH.jl to two dimensions
    setprecision(p)
    M = (N1 + 1) * (N2 + 1)
    D10 = interval.(big.(zeros(N2 + 1, M)))
    D12 = interval.(big.(zeros(N2 + 1, M)))
    w = interval.(big.(0:N1))
    Y = w .^ 2
    X = interval.(big.((-1) .^ (0:N1)))
    k = 0

    for n2 in 0:N2
        D10[n2+1, (n2*(N1+1)+1):((n2+1)*(N1+1))] = interval(big(2)) * X
        D10[n2+1, n2*(N1+1)+1] = interval.(big.(1))
        D12[n2+1, (n2*(N1+1)+1):((n2+1)*(N1+1))] = interval(big(2)) * Y .* X
    end

    f = [D10;
        D12]
    return f
end

# Function for constructing conversion operators cosine to exponential and other way around
function exp2cos(N1,N2) # Adjusted one-dimensional function from https://github.com/matthieucadiot/LocalizedPatternSH.jl to two dimensions
    d = interval(2)*(interval.(ones((N1+1)*(N2+1))))

    d[1] = interval(1)
    for n1 = 1:N1
        d[n1+1] = sqrt(interval(2))
    end

    for n2 = 1:N2
        d[n2*(N1+1)+1] = sqrt(interval(2))
    end

    return d
end

# Function for constructing conversion operators
function exp2sin_cos(N1,N2)
    d = interval(2)*(interval.(ones((N2+1)*N1)))
    for n1 = 1:N1
        d[n1] = sqrt(interval(2))
    end
    return d
end

# Function for constructing conversion operators
function exp2cos_sin(N1,N2)
    d = interval(2)*(interval.(ones((N1+1)*N2)))
    for n2 = 1:N2
        d[(n2-1)*(N1+1)+1] = sqrt(interval(2))
    end
    return d
end

# Function to compute linear part
function l(i1, i2, d1, d2, c)
    i1 = interval(i1) * pi / d1
    i2 = interval(i2) * pi / d2
    return (i1^2 + i2^2)^2 - c^2 * i1^2 + interval(1)
end

# Function to compute integral 1/l_{n_2} using Riemann summations, necessary for computation κ₂ in Z2-bound
function int_ln(n2, c, dx, L)
    I = interval(0)
    for x = 0:dx:sup(c^2*interval(0.5)-interval(n2)*pi/d2-dx)
        I = I + interval(dx)/l((x+dx)*d1*2,n2,d1,d2,c)^2
    end
    I = I + interval(2)*dx/l((c^2*interval(0.5)-interval(n2)*pi/d2)*d1/pi,n2,d1,d2,c)^2
    for x = inf(c^2*interval(0.5)-interval(n2)*pi/d2+dx):dx:L 
        I = I + interval(dx)/l((x)*d1*2,n2,d1,d2,c)^2
    end

    return I + interval(16)/(interval(7)*(interval(2)*pi)^8*(n2*pi/d2 + L)^7)
end

# Function to compute κ₂, needed for computation Z2-bound
function compute_kappa2(dx, L, N)
    # dx: discretization in space for the Riemann summations
    # L: size of the domain on which the integration is done
    # N: truncation finite summation
    I = interval(0)
    for n2 = 1:N 
        I = I + int_ln(n2,c,dx,L)
    end 
    I = int_ln(0,c,dx,L) + interval(2)*I 
    
    # Sum of the rest of the terms 
    I = I + interval(5)/((interval(2)*pi)^7*interval(8)*interval(6)*interval(N)^6)*(interval(2)*d2)^7

    return sqrt(I)
end

# Function used for computing the projection into trace zero functions
function solve_linear(M, b, precis)
    setprecision(precis)

    x = interval.(big.(mid.(M) \ mid.(b)))
    Minv = interval.((inv((mid.(M)))))
    N = size(b)[1]
    Id = interval.(big.(1.0 * I[1:N, 1:N]))

    Z1 = opnorm(LinearOperator(Id - M * Minv), Inf)

    if inf(interval(1) - Z1) > 0
        Y0 = norm(Minv * (M * x - b), Inf)
        rmin = big.(sup(Y0 / inf(1 - Z1)))

        return interval.(inf.(x) .- rmin, sup.(x) .+ rmin)
    else
        return NaN
    end
end

# Function to compute Zᵤ-bound
function computation_Zu(Vbar, N1, N2)

    fourierE = CosFourier(2 * N1, interval(pi) / d1) ⊗ CosFourier(2 * N2, interval(pi) / d2)
    E0 = Sequence(fourierE, interval.((zeros((2 * N1 + 1)*(2 * N2 + 1)))))
    m1 = 2

    for n1 = 0:m1*N1
        for n2 = 0:m1*N2
            an2 = sqrt(interval(4) * (interval(1) + (interval(n2) * interval(pi) / d2)^2 * c^2) - c^4) / (interval(2) * sqrt(c^2 - 2 * (interval(n2) * interval(pi) / d2)^2 + sqrt((c^2 - 2 * (interval(n2) * interval(pi) / d2)^2)^2 + interval(4) * (interval(1) + c^2 * (interval(n2) * interval(pi) / d2)^2) - c^4)))
            bn2 = interval(0.5) * (interval(2) * sqrt(c^2 - 2 * (interval(n2) * interval(pi) / d2)^2 + sqrt((c^2 - 2 * (interval(n2) * interval(pi) / d2)^2)^2 + interval(4) * (interval(1) + c^2 * (interval(n2) * interval(pi) / d2)^2) - c^4)))
            Cn2 = interval(1) / (interval(2) * sqrt(an2^2 + bn2^2) * interval(pi) * sqrt(interval(4) * (interval(1) + (interval(n2) * interval(pi) / d2)^2 * c^2) - c^4))

            E0[(n1, n2)] = Cn2^2 * an2 * interval((-1)^(n1)) * (interval(1) - exp(-interval(4) * an2 * d1)) / (d1 * (interval(4) * an2^2 + (interval(2) * interval(pi) * (interval(n1) * interval(pi) / d1)^2)))

        end
    end

    D12 = convert(Vector{Interval{Float64}}, exp2cos(N1,N2))

    EVbar = project(E0 * Vbar, CosFourier(N1,interval(pi)/d1) ⊗ CosFourier(N2,interval(pi)/d2))
    nVbar = abs(coefficients(D12 .* EVbar)' * coefficients(D12 .* Vbar))

    Zu1 = interval(2) * d1 * nVbar

    # Computation of Zu2, starting with computation of a and C(d1)
    a = interval(10000)
    for n2 = 0:m1*N2
        an2 = sqrt(interval(4) * (interval(1) + (interval(n2) * interval(pi) / d2)^2 * c^2) - c^4) / (interval(2) * sqrt(c^2 - 2 * (interval(n2) * interval(pi) / d2)^2 + sqrt((c^2 - 2 * (interval(n2) * interval(pi) / d2)^2)^2 + interval(4) * (interval(1) + c^2 * (interval(n2) * interval(pi) / d2)^2) - c^4)))
        a = minimum([a an2])
    end
    Cd1 = interval(4) * d1 + interval(4) * exp(-a * d1) / (a * (interval(1) - exp(-interval(1.5) * a * d1))) + interval(2) / (a * (interval(1) - exp(-interval(2) * a * d1)))

    Zu2 = Zu1 + interval(2) * d1 * Cd1 * exp(-interval(2) * a * d1) * nVbar

    return sqrt(Zu1 + Zu2)
end

# Function to compute Zᵤ-bound in the stability case
function computation_Zu_stability(Vbar, δ0, λmin ,N1, N2)
   
    fourierE = CosFourier(2 * N1, interval(pi) / d1) ⊗ CosFourier(2 * N2, interval(pi) / d2)
    E0 = Sequence(fourierE, interval.((zeros((2 * N1 + 1)*(2 * N2 + 1)))))
    m1 = 2
    
    # After studying the variations of an2 and Cn2 with μ, one can prove that the worst case scenario is obtained for μ = δ0. This makes sense as this is the worst exponential decay for the function f_n2,μ in Lemma 6.4

    for n2 = 0:m1*N2
        an2_min = sqrt(interval(4) * (interval(1) + (interval(n2) * interval(pi) / d2)^2 * c^2 - δ0) - c^4) / (interval(2) * sqrt(c^2 - 2 * (interval(n2) * interval(pi) / d2)^2 + sqrt((c^2 - 2 * (interval(n2) * interval(pi) / d2)^2)^2 + interval(4) * (interval(1) + c^2 * (interval(n2) * interval(pi) / d2)^2 - δ0) - c^4)))

        bn2_min = interval(0.5) * (interval(2) * sqrt(c^2 - 2 * (interval(n2) * interval(pi) / d2)^2 + sqrt((c^2 - 2 * (interval(n2) * interval(pi) / d2)^2)^2 + interval(4) * (interval(1) + c^2(interval(n2) * interval(pi) / d2)^2 - δ0) - c^4)))

        Cn2 = interval(1) / (interval(2) * sqrt(an2_min^2 + bn2_min^2) * interval(pi) * sqrt(interval(4) * (interval(1) + (interval(n2) * interval(pi) / d2)^2 * c^2 - δ0) - c^4))

        for n1 = 0:m1*N1
            E0[(n1, n2)] = Cn2^2 * an2_min * interval((-1)^(n1)) * (interval(1) - exp(-interval(4) * an2_min * d1)) / (d1 * (interval(4) * an2_min^2 + (interval(2) * interval(pi) * (interval(n1) * interval(pi) / d1)^2)))
        end
    end

    D12 = convert(Vector{Interval{Float64}}, exp2cos(N1,N2))

    EVbar = project(E0 * Vbar, CosFourier(N1,interval(pi)/d1) ⊗ CosFourier(N2,interval(pi)/d2))
    nVbar = abs(coefficients(D12 .* EVbar)' * coefficients(D12 .* Vbar))

    Zu1 = interval(2) * d1 * nVbar

    # Computation of Zu2, starting with computation of a and C(d1)
    a = interval(10000)
    for n2 = 0:m1*N2
        an2 = sqrt(interval(4) * (interval(1) + (interval(n2) * interval(pi) / d2)^2 * c^2 - δ0) - c^4) / (interval(2) * sqrt(c^2 - 2 * (interval(n2) * interval(pi) / d2)^2 + sqrt((c^2 - 2 * (interval(n2) * interval(pi) / d2)^2)^2 + interval(4) * (interval(1) + c^2 * (interval(n2) * interval(pi) / d2- δ0)^2) - c^4)))
        a = minimum([a an2])
    end
    Cd1 = interval(4) * d1 + interval(4) * exp(-a * d1) / (a * (interval(1) - exp(-interval(1.5) * a * d1))) + interval(2) / (a * (interval(1) - exp(-interval(2) * a * d1)))

    Zu2 = Zu1 + interval(2) * d1 * Cd1 * exp(-interval(2) * a * d1) * nVbar

    return sqrt(Zu1), sqrt(Zu2)
end

# Function to deal with the sine-cosine basis
function mult_sin(V,SS)
    N1,N2 = order(SS)
    
    R0 = interval.(zeros(N1,N1))
    for j = 1:N1
        R0[j,N1-j+1] = -interval(1)
    end

    R = LinearOperator(SinFourier(N1, π/d1)⊗CosFourier(N2,π/d2),SinFourier(N1, π/d1)⊗CosFourier(N2,π/d2),interval.(zeros((N1)*(N2+1),(N1)*(N2+1))))

    for j = 0:N2
        R[(:,j),(:,j)] = R0
    end

    Vexp = Sequence(Fourier(N1, π/d1)⊗CosFourier(N2,π/d2),interval.(zeros((2*N1+1)*(N2+1))))

    for n1 = -N1:N1 
        for n2 = 0:N2
            if n1<0
                Vexp[(n1,n2)] = V[(-n1,n2)]
            else
                Vexp[(n1,n2)] = V[(n1,n2)]
            end 
        end 
    end

    Mexp = project(Multiplication(Vexp),Fourier(N1, π/d1)⊗CosFourier(N2,π/d2), Fourier(N1, π/d1)⊗CosFourier(N2,π/d2),Interval{Float64})

    C = Mexp[(1:N1, :), (-N1:-1, :)]
    D = Mexp[(1:N1, :), (1:N1, :)]

    return  LinearOperator(SS,SS, _matprod(C,coefficients(R)) + D)
end 

# Function to compute the bounds for the stability analysis
function computation_bounds(Vbar,r0,δ0,t,d1,d2,c,SS,D1,D2,𝒵u1,𝒵u2,N1,N2)
    # Construction of the operator L
    ∂1 = project(Derivative((2,0)), SS, SS,Interval{Float64})
    ∂2 = project(Derivative((0,2)), SS, SS,Interval{Float64})
    Δ = copy(∂1)
    radd!(Δ,∂2)

    L = LinearOperator(SS, SS, Diagonal(diag(coefficients(Δ .^ 2 + c^2 * ∂1)))) + I
    Δ = Nothing
    
    ∂1 = Nothing
    ∂2 = Nothing

    if SS == SinFourier(N1, π/d1)⊗CosFourier(N2, π/d2) # Workaround for the sine-cosine basis
        DG = mult_sin(Vbar,SS)
    else
        DG = project(Multiplication(Vbar),SS,SS,Interval{Float64})
    end
    
    DF0 = L +  DG

    D,P =  eigen(coefficients(mid.(D1).*mid.(DF0).*mid.(D2)'))

    # We compute an approximate inverse Pinv for P. Computing the defect nP, we can use a Neumann series argument to compute the real inverse of P as P^{-1} = sum_k (I - Pinv*P)^k Pinv. We propagate the error given by nP in the bounds below. In practice, nP is very small and will not affect the quality of the bounds 
    P = interval.(mid.(D2).*P.*mid.(D1)')
    Pinv = interval.(inv(mid.(P)))
    nP = opnorm(LinearOperator(I - D1.*_matprod(Pinv,P).*D2'),2)
    norm_P = spec_norm(D1.*P.*D2')
    norm_Pinv = spec_norm(D1.*Pinv.*D2')

    D = _matprod(_matprod(Pinv,coefficients(DF0)),P)
    DF = Nothing
    S = diag(D)
    R = D - Diagonal(S)

    St = S .+ t
    Stinv = interval(1) ./ St

    # Computation of bounds Z1i 
    # Diagonal part of DG(U0) is given by V0[(0,0)]

    tN = interval(N1+1)*π/d1
    Z12 = interval(1)/(abs(Vbar[(0,0)] + tN^4 - c^2*tN + interval(1)  -δ0))*(err_Vbar+ norm(Vbar-Vbar[(0,0)],1))

    Z13 = spec_norm((D1.*Stinv).*R.*D2')*(interval(1)+ nP*norm_Pinv/(interval(1)-nP))

    Vbar2 = Vbar*Vbar

    if SS == SinFourier(N1, π/d1)⊗CosFourier(N2, π/d2) # Workaround for the sine-cosine basis
        M1 = coefficients(mult_sin(Vbar2,SS)) - _matprod(coefficients(DG),coefficients(DG))
    else
        M1 = coefficients(project(Multiplication(Vbar2),SS, SS, Interval{Float64})) - _matprod(coefficients(DG),coefficients(DG))
    end
     
    Z14 = sqrt(spec_norm((D1.*Stinv).*_matprod(_matprod(Pinv,M1),Pinv').*((Stinv.*D2)')))*(interval(1)+ nP*norm_Pinv/(interval(1)-nP))

    Z11 = interval(1)/(abs(tN^4 - c^2*tN + interval(1)-δ0))*sqrt(spec_norm(D1.*_matprod(_matprod(P',M1),P).*D2'))

    # Computation of the bounds 𝒞1*r0 an 𝒞2*r0
    norm_SPL = spec_norm((D1.*Stinv).*Pinv.*(diag(coefficients(L)-δ0*I).*D2)')*(interval(1)+ nP*norm_Pinv/(interval(1)-nP))

    κ2 = compute_kappa2(0.1, 20, 20)

    𝒞1 = interval(1)/(interval(1)-interval(0.25)*c^4-δ0)*(err_Vbar+ norm(Vbar-interval(1),1))*(exp(κ2*r0)-interval(1))
    𝒞2 = norm_SPL/(interval(1)-interval(0.25)*c^4-δ0)*(err_Vbar + norm(Vbar-interval(1),1))*(exp(κ2*r0)-interval(1))

    # Computation of the bound 𝒵u3
    𝒵u3 = norm_SPL*𝒵u2

    # Computation of ϵ
    if sup(𝒞1)<1
        κ1 = (𝒵u1 + 𝒞1)/(interval(1)-𝒞1)
        if sup(Z12 + 𝒵u2 + sqrt(interval(1) + κ1^2)*𝒞1) < 1
            if sup(Z12 + 𝒵u2) < 1
                κ2 = (Z11 + (𝒵u2 + sqrt(interval(1) + κ1^2)*𝒞1)*norm_P)/(interval(1) - (Z12 + 𝒵u2 + sqrt(interval(1) + κ1^2)*𝒞1))
                ϵq = Z13 + Z14*(Z11 + interval(2)*𝒵u2*norm_P)/(interval(1) - Z12 - interval(2)*𝒵u2) + interval(2)*𝒵u3*(norm_P + (Z11 + interval(2)*𝒵u2*norm_P)/(interval(1) - Z12 - interval(2)*𝒵u2))
                ϵ = Z13 + Z14*κ2 + (𝒵u3 + 𝒞2*sqrt(interval(1) + κ1^2))*(norm_P + κ2)
                return maximum([ϵ ϵq]),S
            else 
                display("third condition not respected")
                return Nan
            end
        else 
            display("second condition not respected")
            return Nan 
        end 
    else 
        display("first condition not respected")
        return Nan 
    end
end

# Function to study the stability of the solution
function stability(U,Vbar,Ufft,c,d1,d2,𝒵1,𝒵u1,𝒵u2,𝒵2,r0,t,δ0)
    # The sequence U has already been projected in zero trace functions
    S = space(U)
    N01, N02 = order(S)

    fourier = space(Vbar)
    N1, N2 = order(fourier)

    D1 = convert(Vector{Interval{Float64}},exp2cos(N1,N2))
    D2 = interval.(ones((N1+1)*(N2+1)))./D1

    SS = CosFourier(N1, π/d1)⊗CosFourier(N2,π/d2)
    ϵ,S = computation_bounds(Vbar,r0,δ0,t,d1,d2,c,SS,D1,D2,𝒵u1,𝒵u2,N1,N2)

    norm_DF_inv = maximum((interval(1) ./ (abs.(S .- ϵ*abs.(t .+ S))) ))
    norm_DF_inv2 = maximum((interval(1) ./ (abs.(S .+ ϵ*abs.(t .+ S))) ))
    norm_DF_inv = maximum([norm_DF_inv norm_DF_inv2]) # norm of ||DF^{-1}||_2 

    k = 1
    nDF_cos = 0
    ##### counting the number of negative eigenvalues
    while sup(S[k] + ϵ*abs(t+S[k]))<0
        k = k+1
        nDF_cos = nDF_cos + 1
    end

    ##### verifying that the rest of the disks is in the positive part of the real line
    for n = k:length(S)
        if inf(S[n] - ϵ*abs(t+S[n]))<0
            display("At least one disk is intersecting the negative part of the real line")
            display("Stability proof is inconclusive")
            break
        end
    end

    display("Amount of negative eigenvalues is $nDF_cos for the cosine-cosine part")

    D1 = convert(Vector{Interval{Float64}},exp2sin_cos(N1,N2))
    D2 = interval.(ones((N1)*(N2+1)))./D1

    SS = SinFourier(N1, π/d1)⊗CosFourier(N2,π/d2)
    ϵ,S = computation_bounds(Vbar,r0,δ0,t,d1,d2,c,SS,D1,D2,𝒵u1,𝒵u2,N1,N2)



    k = 1
    nDF_sin = 0
    ##### counting the number of negative eigenvalues
    while sup(S[k] + ϵ*abs(t+S[k]))<0
        k = k+1
        nDF_sin = nDF_sin + 1
    end

    #### we expect an eigenvalue at 0, we save the enclosure of the disk containing 0
    zero_inf = S[k] - ϵ*abs(t+S[k])
    zero_sup = S[k] + ϵ*abs(t+S[k])

    ###### verifying that the negative disks are disjoint from the disk containing 0
    if (sup(S[k-1] + ϵ*abs(t+S[k-1])) >= inf(zero_inf))
        display("The negative disks intersect the disk containing 0")
        display("Stability proof is inconclusive")
        return Nan
    end


    ##### verifying that the rest of the disks is in the positive part of the real line and do not intersect the disk containing 0
    for n = k+1:length(S)
        if inf(S[n] - ϵ*abs(t+S[n]))<= inf(zero_sup)
            display("At least one disk is intersecting the disk containing 0")
            display("Stability proof is inconclusive")
            break  
        end
    end

    display("Amount of negative eigenvalues is $nDF_sin for the sine-cosine part")

    nDF = nDF_cos + nDF_sin
    display("DF(ũ) contains exactly $nDF negative eigenvalue(s)")

    # Computation of θ 
    D1 = convert(Vector{Interval{Float64}},exp2cos(N1,N2))
    D2 = interval.(ones((N1+1)*(N2+1)))./D1

    SS = CosFourier(N01, π/d1)⊗CosFourier(N02,π/d2)

    # Construction of the operator L
    ∂1 = project(Derivative((2,0)), SS, SS,Interval{Float64})
    ∂2 = project(Derivative((0,2)), SS, SS,Interval{Float64})
    Δ = copy(∂1)
    radd!(Δ,∂2)

    L = LinearOperator(SS, SS, Diagonal(diag(coefficients(Δ .^ 2 + c^2 * ∂1)))) + I
    
    Δ = Nothing

    ε_vals = [ε(n1, n2, nu1, nu2, Nfft1, Nfft2) for n1 in 0:N01, n2 in 0:N02]
    Gfftder = exp.(Ufft) .- 1 # Note that we have to redefine Gfftder as rifft! changes the argument
    Vbar = rifft!(Gfftder, CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2)) + Sequence(CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2), vec(Cprime * interval.(-inf.(ε_vals), sup.(ε_vals))))

    DG = project(Multiplication(Vbar),SS, SS,Interval{Float64})
    DF0 = L +  DG

    Wbar = mid.(DF0)\(mid.(∂1)*mid.(US))  # computation of the approximation of Wbar
    Wbar = interval.(-2*mid(c)*Wbar)

    # Now we project into the trace zero subspace
    setprecision(80)
    L = convert(Vector{Interval{Float64}}, diag(coefficients(L)))
    Li = ones((N01+1)*(N02+1))./L
    Li_trace = interval.(big.(mid.(Li).^4))
    S = trace(N01,N02,80)
    C = S'
    W = coefficients(Wbar)
    W = W - (Li_trace.*C)*solve_linear(S*(Li_trace.*C),S*W,80)   

    W = interval.(Float64.(inf.(W),RoundDown),Float64.(sup.(W),RoundUp))
    Wbar = Sequence(fourier0,vec(W))

    S = Nothing
    C = Nothing
    Li_trace = Nothing

    θ0 = coefficients(US + interval(2)*c*Wbar)'*coefficients(∂1*US)

    Gfftder = exp.(Ufft)  # Note that we have to redefine Gfftder as rifft! changes the argument
    exp_U = rifft!(Gfftder, CosFourier(N01, interval(pi) / d1) ⊗ CosFourier(N02, interval(pi) / d2)) + Sequence(fourier0, vec(Cprime * interval.(-inf.(ε_vals), sup.(ε_vals))))

    norm_exp = sqrt(interval(2)*d1)*(norm(exp_U*Wbar,2) + err_Vbar*norm(Wbar,1)) # norm of ||e^{U}*W||_2

    Z1 = 𝒵1
    Z2 = 𝒵2

    κ1 = 1 / (1 - c^4 / 4) # Recompute κ1 and κ2
    κ2 = compute_kappa2(0.1, 20, 20)
    ϵ0 = κ1*r0 + interval(4)*c^2*norm_DF_inv/(interval(1)-Z1-Z2*r0)*(sqrt(interval(2)*d1)*norm(∂1*US + interval(1)/(interval(2)*c)*DF0*Wbar,2) +r0/(interval(2)-c^2) + interval(0.5)/c*abs(interval(1)-exp(κ2*r0))*norm_exp )
    ϵ = sqrt(interval(2)*d1)*norm(US + interval(2)*c*Wbar,2)*r0/(interval(2)-c^2) + (κ1*r0 + interval(2)*c*ϵ0)*(sqrt(interval(2)*d1)norm(∂2*US,2) + r0/(interval(2)-c^2))

    θ = interval(inf(θ0-ϵ),sup(θ0+ϵ))

    if nDF == 0 || (inf(θ) > 0 && nDF == 1)
        display("Solution is orbitally stable")
    else
        display("Solution is orbitally unstable")
    end

end