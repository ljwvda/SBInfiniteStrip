using MATLAB
mat"addpath('/Users/lindsey/Code/Intlab_V12')"  #Here, add the path to your intlab.
mat"startintlab"
function _matprod(A,B) #For Real matrix products
    #Ensure to have started Intlab
    Ai = inf.(A)
    As = sup.(A)
    Bi = inf.(B)
    Bs = sup.(B)
    d = mat"infsup($Ai,$As)*infsup($Bi,$Bs)"
    return interval.(d["inf"],d["sup"])
end


function spec_norm(A) #For Real matrix spectral norm
    #Ensure to have started Intlab
    Ai = inf.(A)
    As = sup.(A)
    d = mat"norm(infsup($Ai,$As),2)"
    return interval.(d["inf"],d["sup"])
end

function _cmatprod(A,B) #For complex matrix products
    #Ensure to have started Intlab
    Air = inf.(real(A))
    Aii = inf.(imag(A))
    Asr = sup.(real(A))
    Asi = sup.(imag(A))
    Bir = inf.(real(B))
    Bii = inf.(imag(B))
    Bsr = sup.(real(B))
    Bsi = sup.(imag(B))
    dr = mat"infsup($Air,$Asr)*infsup($Bir,$Bsr) - infsup($Aii,$Asi)*infsup($Bii,$Bsi)"
    di = mat"infsup($Aii,$Asi)*infsup($Bir,$Bsr) + infsup($Air,$Asr)*infsup($Bii,$Bsi)"
    return interval.(dr["inf"],dr["sup"]) + interval.(1im*di["inf"],1im*di["sup"])
end

function _cmatprod2(A,B) #For products of LinearOperators in RadiiPolynomial.jl
    dom = domain(A)
    codom = codomain(B)
    A = coefficients(A)
    B = coefficients(B)
    #Ensure to have started Intlab
    Air = inf.(real(A))
    Aii = inf.(imag(A))
    Asr = sup.(real(A))
    Asi = sup.(imag(A))
    Bir = inf.(real(B))
    Bii = inf.(imag(B))
    Bsr = sup.(real(B))
    Bsi = sup.(imag(B))
    dr = mat"infsup($Air,$Asr)*infsup($Bir,$Bsr) - infsup($Aii,$Asi)*infsup($Bii,$Bsi)"
    di = mat"infsup($Aii,$Asi)*infsup($Bir,$Bsr) + infsup($Air,$Asr)*infsup($Bii,$Bsi)"
    return LinearOperator(dom,codom,interval.(dr["inf"],dr["sup"]) + interval.(1im*di["inf"],1im*di["sup"]))
end






