# Existence and orbital stability proofs of traveling wave solutions on an infinite strip for the suspension bridge equation

This repository contains the code corresponding to the paper *Existence and orbital stability proofs of traveling wave solutions on an infinite strip for the suspension bridge equation* by Matthieu Cadiot and Lindsey van der Aalst ([arXiv:2509.16693](https://arxiv.org/abs/2509.16693)).

---

## How to run the proofs

Run one of the following files to reproduce the results in the paper:

- [`SBinfinite_proof_c12.jl`](SBinfinite_proof_c12.jl) — corresponds to **Theorem 6.1**.
- [`SBinfinite_proof_c13.jl`](SBinfinite_proof_c13.jl) — corresponds to **Theorem 6.2**.
- [`SBinfinite_proof_c08.jl`](SBinfinite_proof_c08.jl) — corresponds to **Theorem 6.3**.

Each proof script contains both the **existence** and **orbital (in)stability** proofs.

---

## Requirements

To obtain rigorous proofs, [INTLAB](https://www.tuhh.de/ti3/rump/) is required.  
You can add the path to your local INTLAB folder at the top of `matproducts.jl`.

### Versions

- Julia: **v1.10.4**

### Libraries

- RadiiPolynomial v0.8.24  
- MAT v0.10.7  
- IntervalArithmetic v0.22.36  
- MATLAB v0.9.0  
- JLD2 v0.5.15  

---

## Contact

For questions about the code, please contact:  
**Lindsey van der Aalst** — l.j.w.van.der.aalst@vu.nl
