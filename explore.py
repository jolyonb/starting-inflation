"""
explore

Used to explore initial data
"""
from run import params, initial_data
from evolver.initialize import unpack
from evolver.eoms import compute_hartree, compute_rho, compute_deltarho2

unpacked_data = unpack(initial_data, params.total_wavenumbers)
a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data
H = adot/a

phi2pt, phi2ptdt, phi2ptgrad = compute_hartree(phiA, phidotA, phiB, phidotB, params)

rho = compute_rho(phi0, phi0dot, params.model)
deltarho2 = compute_deltarho2(a, phi0, phi2pt, phi2ptdt, phi2ptgrad, params.model)

print("rho:", rho)
print("deltarho2:", deltarho2)
