#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore

Used to explore initial data
"""
from run import params, initial_data
from evolver.initialize import unpack
from math import pi
from evolver.eoms import compute_hartree, compute_rho, compute_deltarho2

unpacked_data = unpack(initial_data, params.total_wavenumbers)
a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data
H = adot/a

phi2pt, phi2ptdt, phi2ptgrad = compute_hartree(phiA, phidotA, phiB, phidotB, params)

rho = compute_rho(phi0, phi0dot, params.model)
deltarho2 = compute_deltarho2(a, phi0, phi2pt, phi2ptdt, phi2ptgrad, params.model)

print("rho:", rho)
print("deltarho2:", deltarho2)
print("deltarho2/rho:", deltarho2/rho)
print("kappa^4/(2 pi^2):", params.kappa**4/(2*pi**2))
print("deltarho2 / (kappa^4/(2 pi^2)):", deltarho2/(params.kappa**4/(2*pi**2)))
print()
print("phi2pt:", phi2pt)
print("kappa^2/(4 pi^2):", params.kappa**2/4/pi**2)
print("phi2pt / (kappa^2/(4 pi^2)):", phi2pt / (params.kappa**2/4/pi**2))
