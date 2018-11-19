#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the equations of motion associated with a model
"""
from math import pi
import numpy as np
from math import sqrt

def eoms(unpacked_data, params, time=None):
    """
    Workhorse to compute all equations of motion.

    Arguments:
        * unpacked_data: Tuple containing all data:
                         (a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB)
        * params: The parameters class associated with the data
        * time: The current time (shouldn't be needed for EOMs, but helpful for debug)

    Returns a tuple of second derivatives:
        * (addot, phi0ddot, phiddotA, psidotA, phiddotB, psidotB)
    """
    # Initialization
    a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data
    H = adot/a

    if params.hartree:
        # Compute Hartree corrections
        hpotential, hgradient, hkinetic = compute_hartree(phiA, phidotA, phiB, phidotB, params)
    else:
        hpotential, hgradient, hkinetic = (0, 0, 0)

    # Debug output
    # print(time, hpotential, hgradient, hkinetic)

    # Background fields
    Hdot = compute_hubbledot(a, phi0dot, hkinetic, hgradient, params)
    addot = a*(Hdot + H*H)
    phi0ddot = compute_phi0ddot(phi0, phi0dot, H, hpotential, params)

    # Perturbative modes
    psidotA = compute_perturb_psidot(phi0dot, H, psiA, phiA, params)
    psidotB = compute_perturb_psidot(phi0dot, H, psiB, phiB, params)
    phiddotA = compute_perturb_phiddot(phi0, phi0dot, a, H, phiA, phidotA, psiA, psidotA, hpotential, params)
    phiddotB = compute_perturb_phiddot(phi0, phi0dot, a, H, phiB, phidotB, psiB, psidotB, hpotential, params)

    # Return results
    return addot, phi0ddot, phiddotA, psidotA, phiddotB, psidotB

def compute_hubble(a, phi0, phi0dot, hpotential, hkinetic, hgradient, params):
    """
    Computes the Hubble factor.

    Arguments:
        * a, phi0, phi0dot, hpotential, hkinetic, hgradient: Initial values
        * params: Parameters class

    Returns H == adot/a
    """
    return sqrt((compute_rho(phi0, phi0dot, params)
                 + compute_deltarho2(a, phi0, hkinetic, hgradient, hpotential, params)
                 )/3)

def compute_rho(phi0, phi0dot, params):
    """
    Computes the background energy density.

    Arguments:
        * phi0, phi0dot: Initial values
        * params: Parameters class

    Returns rho
    """
    return phi0dot * phi0dot / 2 + potential(phi0, params)

def compute_deltarho2(a, phi0, hkinetic, hgradient, hpotential, params):
    """
    Computes the average perturbed energy density.

    Arguments:
        * phi0, hkinetic, hgradient, hpotential: Current values
        * params: Parameters class

    Returns deltarho2
    """
    result = hkinetic + hgradient/(a*a) + hpotential*ddpotential(phi0, params)
    result /= 2
    return result

def compute_hubbledot(a, phi0dot, hkinetic, hgradient, params):
    """
    Computes the derivative of the Hubble factor.

    Arguments:
        * a, phi0dot, hkinetic, hgradient: Current values
        * params: Parameters class

    Returns H == adot/a
    """
    return - 0.5 * (phi0dot * phi0dot + hkinetic + hgradient/(a*a)/3)

def compute_phi0ddot(phi0, phi0dot, H, hpotential, params):
    """
    Computes the second derivative of phi0.

    Arguments:
        * phi0, phi0dot, H, hpotential: Current values
        * params: Parameters class

    Returns phiddot
    """
    return (- 3 * H * phi0dot
            - dpotential(phi0, params)
            - 0.5 * dddpotential(phi0, params) * hpotential)

def compute_perturb_psidot(phi0dot, H, psi, phi, params):
    """
    Computes the time derivative of psi.

    Arguments:
        * phi0dot, H, psi, phi: Current values
        * params: Parameters class

    Returns psidot
    """
    return -H*psi + 0.5*phi0dot*phi

def compute_perturb_phiddot(phi0, phi0dot, a, H, phi, phidot, psi, psidot, hpotential, params):
    """
    Computes the time derivative of delta phi.

    Arguments:
        * phi0, phi0dot, a, H, phi, phidot, psi, psidot, hpotential: Current values
        * params: Parameters class

    Returns phiddot
    """
    k2 = params.all_wavenumbers2
    return (-3*H*phidot
            - (k2/(a*a) + ddpotential(phi0, params) + 0.5*ddddpotential(phi0, params)*hpotential)*phi
            - (2*dpotential(phi0, params) + dddpotential(phi0, params)*hpotential)*psi
            + 4*phi0dot*psidot
            )

def compute_initial_psi(a, adot, phi0, phi0dot, phiA, phidotA, phiB, phidotB,
                        hkinetic, hgradient, hpotential, params):
    """
    Compute initial values of psiA and psiB

    Arguments:
        * a, adot, phi0, phi0dot, phiA, phidotA, phiB, phidotB,
          hkinetic, hgradient, hpotential: Current values
        * params: Parameters class

    Returns a tuple:
        * (psiA, psiB)
    """
    # Compute background quantities we need
    H = adot/a
    Hdot = compute_hubbledot(a, phi0dot, hkinetic, hgradient, params)
    phi0ddot = compute_phi0ddot(phi0, phi0dot, H, hpotential, params)

    # Compute psi
    k2 = params.all_wavenumbers2
    factor = Hdot + k2/(a*a)
    psiA = 0.5*(phi0ddot*phiA - phi0dot*phidotA)/factor
    psiB = 0.5*(phi0ddot*phiB - phi0dot*phidotB)/factor

    # Return the results
    return psiA, psiB

def compute_psi_constraint_viol(a, adot, phi0, phi0dot, phiA, phidotA, phiB, phidotB,
                                psiA, psiB, hkinetic, hgradient, hpotential, params):
    """
    Compute the constraint violation in psi.

    Arguments:
        * a, adot, phi0, phi0dot, phiA, phidotA, phiB, phidotB,
          hkinetic, hgradient, hpotential: Current values
        * params: Parameters class

    Returns a tuple:
        * (psiA, psiB)
    """
    # Construct the psi modes
    psi_0, psi_1 = construct_full_modes(psiA, psiB, params)

    # Construct the psi modes based on the constraint
    result = compute_initial_psi(a, adot, phi0, phi0dot, phiA, phidotA, phiB, phidotB,
                                 hkinetic, hgradient, hpotential, params)
    psi_constraint_0, psi_constraint_1 = construct_full_modes(result[0], result[1], params)

    # Concatenate all psi modes into one big long vector
    psi = np.concatenate((psi_0, psi_1[0], psi_1[1], psi_1[2]))
    psi_constraint = np.concatenate((psi_constraint_0,
                                     psi_constraint_1[0],
                                     psi_constraint_1[1],
                                     psi_constraint_1[2]))

    # Compute the violation
    violation = psi - psi_constraint

    return violation

def compute_hubble_constraint_viol(a, adot, phi0, phi0dot, hpotential, hkinetic, hgradient, params):
    """
    Compute the constraint violation in Hubble.

    Arguments:
        * a, adot, phi0, phi0dot, hpotential, hkinetic, hgradient: Current values
        * params: Parameters class

    Returns a tuple:
        * (psiA, psiB)
    """
    # Compute the current Hubble
    H = adot / a

    # Compute the expected Hubble
    Hexpect = compute_hubble(a, phi0, phi0dot, hpotential, hkinetic, hgradient, params)

    # Compute the violation
    violation = H - Hexpect

    return violation

def potential(phi0, params):
    """
    Computes the potential.

    Arguments:
        * phi0: The field value.
        * params: Parameters class, which may store values associated with the potential.

    Returns:
        * The value of the potential.
    """
    # We use a lambda phi^4/4 potential
    return params.lamda * phi0**4 / 4

def dpotential(phi0, params):
    """
    Computes the derivative of the potential.

    Arguments:
        * phi0: The field value.
        * params: Parameters class, which may store values associated with the potential.

    Returns:
        * The value of the derivative of the potential.
    """
    # We use a lambda phi^4/4 potential
    return params.lamda * phi0**3

def ddpotential(phi0, params):
    """
    Computes the second derivative of the potential.

    Arguments:
        * phi0: The field value.
        * params: Parameters class, which may store values associated with the potential.

    Returns:
        * The value of the second derivative of the potential.
    """
    # We use a lambda phi^4/4 potential
    return params.lamda * 3 * phi0**2

def dddpotential(phi0, params):
    """
    Computes the third derivative of the potential.

    Arguments:
        * phi0: The field value.
        * params: Parameters class, which may store values associated with the potential.

    Returns:
        * The value of the third derivative of the potential.
    """
    # We use a lambda phi^4/4 potential
    return params.lamda * 6 * phi0

def ddddpotential(phi0, params):
    """
    Computes the fourth derivative of the potential.

    Arguments:
        * phi0: The field value.
        * params: Parameters class, which may store values associated with the potential.

    Returns:
        * The value of the fourth derivative of the potential.
    """
    # We use a lambda phi^4/4 potential
    return params.lamda * 6

def construct_full_modes(phiA, phiB, params):
    """
    Based on phiA and phiB, construct the full version of the fields.
    Note: Also works for psi!

    Arguments:
        * phiA, phiB: The field values.
        * params: Parameters class

    Returns: A list of arrays of the full (complex) modes for each k
        * [phi0, phi1[0:2]]
    """
    # Start by splitting the data into l=0 and l=1 modes
    phiA0 = phiA[:params.k_modes]
    phiA1 = phiA[params.k_modes:]
    phiB0 = phiB[:params.k_modes]
    phiB1 = phiB[params.k_modes:]

    phi0 = phiA0 * params.poscoeffs[0] + phiB0 * params.velcoeffs[0]
    phi1 = [phiA1 * params.poscoeffs[1][i] + phiB1 * params.velcoeffs[1][i] for i in range(3)]

    return phi0, phi1

def compute_hartree(phiA, phidotA, phiB, phidotB, params):
    """
    Based on fullphi and fullphidot, construct the Hartree corrections.

    Arguments:
        * phiA, phidotA, phiB, phidotB: The field values including coefficients, split
                                        into lists by l values
        * params: Parameters class

    Returns: A tuple
        * (hpotential, hgradient, hkinetic)
    """
    # Compute the fields, with associated coefficients
    fullphi0, fullphi1 = construct_full_modes(phiA, phiB, params)
    fullphidot0, fullphidot1 = construct_full_modes(phidotA, phidotB, params)

    # Compute Hartree corrections
    hpotential = hartree_potential(fullphi0, params)
    hgradient = hartree_gradient(fullphi1, params)
    hkinetic = hartree_kinetic(fullphidot0, params)

    # Return results
    return hpotential, hgradient, hkinetic

def hartree_kinetic(fullphidot0, params):
    """
    Computes the Hartree correction <(delta dot{hat{phi}})^2>

    Arguments:
        * fullphidot0: Array of all phidot_k values for ell = 0
        * params: Parameters class

    Returns:
        * The value of <(delta dot{hat{phi}})^2>
    """
    squares = np.real(fullphidot0 * np.conj(fullphidot0))
    result = np.sum(params.gaussian_profile[0] * params.k2_grids[0] * squares) / (2*pi*params.Rmax)
    return result

def hartree_potential(fullphi0, params):
    """
    Computes the Hartree correction <(delta hat{phi})^2>

    Arguments:
        * fullphi0: Array of all phi_k values for ell = 0
        * params: Parameters class

    Returns:
        * The value of <(delta hat{phi})^2>
    """
    squares = np.real(fullphi0 * np.conj(fullphi0))
    result = np.sum(params.gaussian_profile[0] * params.k2_grids[0] * squares) / (2*pi*params.Rmax)
    return result

def hartree_gradient(fullphi1, params):
    """
    Computes the Hartree correction h^{ij}/a^2*<partial_i delta hat{phi} partial_j delta hat{phi}>

    Arguments:
        * fullphi1: Array of all phi_k values for ell = 1
        * params: Parameters class

    Returns:
        * The value of h^{ij}/a^2*<partial_i delta hat{phi} partial_j delta hat{phi}>
    """
    result = 0
    sumsquare = np.real(sum(fullphi1[m] * np.conj(fullphi1[m]) for m in range(3)))
    result = np.sum(params.gaussian_profile[1] * params.denom_fac[1] * params.k2_grids[1] * sumsquare)
    result /= (6*pi*params.Rmax**3)
    return result
