"""Symbolic derivation of the normalized-splatting backward pass.

Differentiates the 3x3 projection Jacobian J -- including the normalization row
[t_x, t_y, t_z] / ||t|| that distinguishes normalized splatting from vanilla
EWA splatting -- with respect to t_x, t_y and t_z, and prints dL/dt_{x,y,z}.

The printed expressions are what `computeCov2DCUDA` in
submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu implements by
hand. Re-run this if that Jacobian ever changes.

    pip install sympy
    python docs/derivations/gen_derivatives.py
"""
from sympy import symbols, sqrt, diff, Matrix

# Define symbols
t_x, t_y, t_z, f_x, f_y = symbols('t_x t_y t_z f_x f_y')
dL_dJ00, dL_dJ01, dL_dJ02, dL_dJ10, dL_dJ11, dL_dJ12, dL_dJ20, dL_dJ21, dL_dJ22 = symbols('dL_dJ00 dL_dJ01 dL_dJ02 dL_dJ10 dL_dJ11 dL_dJ12 dL_dJ20 dL_dJ21 dL_dJ22')

# Original expressions for the elements of J
txsq = t_x**2
tysq = t_y**2
tzsq = t_z**2
tnorm = sqrt(txsq + tysq + tzsq)

J = Matrix([
    [f_x / t_z, 0, -(f_x *t_x) / tzsq],
    [0, f_y / t_z, -(f_y *t_y) / tzsq],
    [t_x/tnorm, t_y/tnorm, t_z/tnorm]
])


dJ00_dt_x = diff(J[0, 0], t_x)
dJ00_dt_y = diff(J[0, 0], t_y)
dJ00_dt_z = diff(J[0, 0], t_z)
dJ01_dt_x = diff(J[0, 1], t_x)
dJ01_dt_y = diff(J[0, 1], t_y)
dJ01_dt_z = diff(J[0, 1], t_z)
dJ02_dt_x = diff(J[0, 2], t_x)
dJ02_dt_y = diff(J[0, 2], t_y)
dJ02_dt_z = diff(J[0, 2], t_z)
dJ10_dt_x = diff(J[1, 0], t_x)
dJ10_dt_y = diff(J[1, 0], t_y)
dJ10_dt_z = diff(J[1, 0], t_z)
dJ11_dt_x = diff(J[1, 1], t_x)
dJ11_dt_y = diff(J[1, 1], t_y)
dJ11_dt_z = diff(J[1, 1], t_z)
dJ12_dt_x = diff(J[1, 2], t_x)
dJ12_dt_y = diff(J[1, 2], t_y)
dJ12_dt_z = diff(J[1, 2], t_z)
dJ20_dt_x = diff(J[2, 0], t_x)
dJ20_dt_y = diff(J[2, 0], t_y)
dJ20_dt_z = diff(J[2, 0], t_z)
dJ21_dt_x = diff(J[2, 1], t_x)
dJ21_dt_y = diff(J[2, 1], t_y)
dJ21_dt_z = diff(J[2, 1], t_z)
dJ22_dt_x = diff(J[2, 2], t_x)
dJ22_dt_y = diff(J[2, 2], t_y)
dJ22_dt_z = diff(J[2, 2], t_z)

partial_derivatives = {
    'dJ00_dt_x': dJ00_dt_x,
    'dJ00_dt_y': dJ00_dt_y,
    'dJ00_dt_z': dJ00_dt_z,
    'dJ01_dt_x': dJ01_dt_x,
    'dJ01_dt_y': dJ01_dt_y,
    'dJ01_dt_z': dJ01_dt_z,
    'dJ02_dt_x': dJ02_dt_x,
    'dJ02_dt_y': dJ02_dt_y,
    'dJ02_dt_z': dJ02_dt_z,
    'dJ10_dt_x': dJ10_dt_x,
    'dJ10_dt_y': dJ10_dt_y,
    'dJ10_dt_z': dJ10_dt_z,
    'dJ11_dt_x': dJ11_dt_x,
    'dJ11_dt_y': dJ11_dt_y,
    'dJ11_dt_z': dJ11_dt_z,
    'dJ12_dt_x': dJ12_dt_x,
    'dJ12_dt_y': dJ12_dt_y,
    'dJ12_dt_z': dJ12_dt_z,
    'dJ20_dt_x': dJ20_dt_x,
    'dJ20_dt_y': dJ20_dt_y,
    'dJ20_dt_z': dJ20_dt_z,
    'dJ21_dt_x': dJ21_dt_x,
    'dJ21_dt_y': dJ21_dt_y,
    'dJ21_dt_z': dJ21_dt_z,
    'dJ22_dt_x': dJ22_dt_x,
    'dJ22_dt_y': dJ22_dt_y,
    'dJ22_dt_z': dJ22_dt_z,
}

# Calculate dL/dt_x, dL/dt_y, and dL/dt_z by multiplying each partial derivative by the corresponding dL/dJ and summing
dL_dt_x = sum([
    partial_derivatives['dJ00_dt_x']*dL_dJ00, partial_derivatives['dJ01_dt_x']*dL_dJ01, partial_derivatives['dJ02_dt_x']*dL_dJ02,
    partial_derivatives['dJ10_dt_x']*dL_dJ10, partial_derivatives['dJ11_dt_x']*dL_dJ11, partial_derivatives['dJ12_dt_x']*dL_dJ12,
    partial_derivatives['dJ20_dt_x']*dL_dJ20, partial_derivatives['dJ21_dt_x']*dL_dJ21, partial_derivatives['dJ22_dt_x']*dL_dJ22
])

dL_dt_y = sum([
    partial_derivatives['dJ00_dt_y']*dL_dJ00, partial_derivatives['dJ01_dt_y']*dL_dJ01, partial_derivatives['dJ02_dt_y']*dL_dJ02,
    partial_derivatives['dJ10_dt_y']*dL_dJ10, partial_derivatives['dJ11_dt_y']*dL_dJ11, partial_derivatives['dJ12_dt_y']*dL_dJ12,
    partial_derivatives['dJ20_dt_y']*dL_dJ20, partial_derivatives['dJ21_dt_y']*dL_dJ21, partial_derivatives['dJ22_dt_y']*dL_dJ22
])

dL_dt_z = sum([
    partial_derivatives['dJ00_dt_z']*dL_dJ00, partial_derivatives['dJ01_dt_z']*dL_dJ01, partial_derivatives['dJ02_dt_z']*dL_dJ02,
    partial_derivatives['dJ10_dt_z']*dL_dJ10, partial_derivatives['dJ11_dt_z']*dL_dJ11, partial_derivatives['dJ12_dt_z']*dL_dJ12,
    partial_derivatives['dJ20_dt_z']*dL_dJ20, partial_derivatives['dJ21_dt_z']*dL_dJ21, partial_derivatives['dJ22_dt_z']*dL_dJ22
])

# Simplify the expressions
dL_dt_x_simplified = dL_dt_x.simplify()
dL_dt_y_simplified = dL_dt_y.simplify()
dL_dt_z_simplified = dL_dt_z.simplify()

print(dL_dt_x_simplified)
print(dL_dt_y_simplified)
print(dL_dt_z_simplified)

