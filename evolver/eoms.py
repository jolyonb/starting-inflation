# -*- coding: utf-8 -*-
"""
eoms.py

Computes the equations of motion associated with a model
"""
from math import pi, sqrt
import numpy as np

class EOMParameters(object):
    """
    A class to store all the parameters that are needed for computing EOMs,
    and also for plotting
    """
    def __init__(self,
                 Rmax,
                 kappa,
                 k_modes,
                 total_wavenumbers,
                 hartree,
                 infmodel,
                 k_grids,
                 k2_grids,
                 all_wavenumbers,
                 all_wavenumbers2,
                 denom_fac,
                 gaussian_profile,
                 poscoeffs,
                 velcoeffs):
        """Save parameters for later computations"""
        self.Rmax = Rmax
        self.kappa = kappa
        self.k_modes = k_modes
        self.total_wavenumbers = total_wavenumbers
        self.hartree = hartree
        self.model = infmodel
        self.k_grids = k_grids
        self.k2_grids = k2_grids
        self.all_wavenumbers = all_wavenumbers
        self.all_wavenumbers2 = all_wavenumbers2
        self.denom_fac = denom_fac
        self.gaussian_profile = gaussian_profile
        self.poscoeffs = poscoeffs
        self.velcoeffs = velcoeffs


def compute_all(unpacked_data, params):
    """
    Routine to compute all quantities of interest

    Arguments:
        * unpacked_data: Tuple containing all data:
                         (a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB)
        * params: EOMParameters object
    """
    # Initialization
    a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data

    # Compute energies
    rho = compute_rho(phi0, phi0dot, params.model)
    phi2pt, phi2ptdt, phi2ptgrad = compute_hartree(phiA, phidotA, phiB, phidotB, params)
    deltarho2 = compute_deltarho2(a, phi0, phi2pt, phi2ptdt, phi2ptgrad, params.model)

    # Compute quantities that depend on Hartree on or off
    if params.hartree:
        H = compute_hubble(rho, deltarho2)
        Hdot = compute_hubbledot(a, phi0dot, phi2ptdt, phi2ptgrad)
        phi0ddot = compute_phi0ddot(phi0, phi0dot, H, phi2pt, params.model)
    else:
        H = compute_hubble(rho, 0)
        Hdot = compute_hubbledot(a, phi0dot, 0, 0)
        phi0ddot = compute_phi0ddot(phi0, phi0dot, H, 0, params.model)

    # Compute some derivative quantities
    adot = a * H
    addot = a * (Hdot + H * H)
    epsilon = slow_roll_epsilon(H, Hdot)

    # Return results
    return rho, deltarho2, H, adot, Hdot, addot, epsilon, phi0ddot, phi2pt, phi2ptdt, phi2ptgrad

def eoms(unpacked_data, params, time=None):
    """
    Workhorse to compute all equations of motion.

    Arguments:
        * unpacked_data: Tuple containing all data:
                         (a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB)
        * params: EOMParameters object
        * time: The current time (shouldn't be needed for EOMs, but helpful for debug)

    Returns a tuple of second derivatives:
        * (adot, phi0ddot, phiddotA, psidotA, phiddotB, psidotB)
    """
    # Initialization
    a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data

    # Compute background quantities
    (rho, deltarho2, H, adot, Hdot, addot,
     epsilon, phi0ddot, phi2pt, phi2ptdt, phi2ptgrad) = compute_all(unpacked_data, params)

    # Turn off phi2pt for the following if needed
    if not params.hartree:
        phi2pt = 0

    # Perturbative modes
    psidotA = compute_perturb_psidot(phi0dot, H, psiA, phiA)
    psidotB = compute_perturb_psidot(phi0dot, H, psiB, phiB)
    phiddotA = compute_perturb_phiddot(phi0, phi0dot, a, H, phiA, phidotA, psiA,
                                       psidotA, phi2pt, params)
    phiddotB = compute_perturb_phiddot(phi0, phi0dot, a, H, phiB, phidotB, psiB,
                                       psidotB, phi2pt, params)

    # Return results
    return (adot, epsilon, phi0dot, phi0ddot,
            phidotA, phiddotA, psidotA, phidotB, phiddotB, psidotB)

def slow_roll_epsilon(H, Hdot):
    """Computes the slow roll parameter epsilon from H and Hdot"""
    return - Hdot / (H * H)

def N_efolds(a):
    """Computes the number of efolds that have passed given the current scalefactor"""
    return np.log(a)

def compute_hubble(rho, deltarho2):
    """
    Computes the Hubble factor.

    Arguments:
        * rho: Background energy density
        * deltarho2: Perturbed energy density

    Returns H == \dot{a}/a
    """
    return sqrt((rho + deltarho2)/3)

def compute_rho(phi0, phi0dot, model):
    """
    Computes the background energy density.

    Arguments:
        * phi0, phi0dot: Background values
        * model: InflationModel class

    Returns rho == \dot{\phi}_0^2/2 + V(\phi_0)
    """
    return phi0dot * phi0dot / 2 + model.potential(phi0)

def compute_deltarho2(a, phi0, phi2pt, phi2ptdt, phi2ptgrad, model):
    """
    Computes the average perturbed energy density.

    Arguments:
        * a, phi0: Background values
        * phi2pt, phi2ptdt, phi2ptgrad: various two-point functions
        * model: InflationModel class

    Returns deltarho2 == 1/2 (2-pt \dot{\phi} + 2-pt \grad(\phi)/a^2 + V''(phi_0) * 2-pt \phi + 0.25 * V''''(phi_0) * 2-pt \phi * 2-pt \phi)
    """
    return (phi2ptdt + phi2ptgrad/(a*a) + phi2pt*model.ddpotential(phi0) + (phi2pt*phi2pt*model.ddddpotential(phi0)/4))/2

def compute_hubbledot(a, phi0dot, phi2ptdt, phi2ptgrad):
    """
    Computes the derivative of the Hubble factor.

    Arguments:
        * a, phi0dot: Background values
        * phi2ptdt, phi2ptgrad: Perturbed values

    Returns H == -1/(2 M_{pl}^2) [\dot{\phi}_0^2 + 2-pt \dot{\phi} + 2-pt \grad(\phi)/3a^2]
    """
    return - 0.5 * (phi0dot * phi0dot + phi2ptdt + phi2ptgrad/(3*a*a))

def compute_phi0ddot(phi0, phi0dot, H, phi2pt, model):
    """
    Computes the second derivative of phi0.

    Arguments:
        * phi0, phi0dot, H: Background values
        * phi2pt: Perturbed values
        * model: InflationModel class

    Returns phiddot == -3 H \dot{\phi}_0 - V'(\phi_0) - 1/2 V'''(phi_0) * <\phi^2>
    """
    return (- 3 * H * phi0dot
            - model.dpotential(phi0)
            - 0.5 * model.dddpotential(phi0) * phi2pt)

def compute_perturb_psidot(phi0dot, H, psi, phi):
    """
    Computes the time derivative of psi.

    Arguments:
        * phi0dot, H: Background values
        * psi, phi: Perturbed values

    Returns psidot == - H \psi + 1/(2 M_{pl}^2) \dot{\phi}_0 \delta \phi
    """
    return -H*psi + 0.5*phi0dot*phi

def compute_perturb_phiddot(phi0, phi0dot, a, H, phi, phidot, psi, psidot, phi2pt, params):
    """
    Computes the time derivative of delta phi.

    Arguments:
        * phi0, phi0dot, a, H: Background values
        * phi, phidot, psi, psidot, phi2pt: Perturbed values
        * params: EOMParameters object

    Returns phiddot == -3H \dot{\phi} - [k^2/a^2 + V''(\phi_0) + 1/2 V''''(\phi_0) <\phi^2>] \phi
                       - 2 [V'(\phi_0) + 1/2 V'''(\phi_0) <\phi^2>] psi + 4 \dot{\phi}_0 \dot{\psi}
    """
    model = params.model
    k2 = params.all_wavenumbers2
    return (-3*H*phidot
            - (k2/(a*a) + model.ddpotential(phi0) + 0.5*model.ddddpotential(phi0)*phi2pt)*phi
            - 2 * (model.dpotential(phi0) + 0.5*model.dddpotential(phi0)*phi2pt)*psi
            + 4*phi0dot*psidot)

def compute_initial_psi(a, adot, phi0, phi0dot,
                        phiA, phidotA, phiB, phidotB,
                        phi2pt, phi2ptdt, phi2ptgrad, params):
    r"""
    Compute initial values of psiA and psiB

    Arguments:
        * a, adot, phi0, phi0dot: Background values
        * phiA, phidotA, phiB, phidotB: Perturbed values
        * phi2pt, phi2ptdt, phi2ptgrad: 2-point values
        * params: EOMParameters object

    Returns a tuple:
        * (psiA, psiB)

    Formula:
    (\dot{H} + 2/(3 Mp^2 a^2) phi2ptgrad + k^2/a^2) \psi = 1/(2 M_{pl}^2) [\ddot{\phi}_0 \phi - \dot{\phi}_0 \dot{\phi}]
    """
    # Compute background quantities we need
    H = adot/a
    Hdot = compute_hubbledot(a, phi0dot, phi2ptdt, phi2ptgrad)
    phi0ddot = compute_phi0ddot(phi0, phi0dot, H, phi2pt, params.model)

    # Compute psi
    factor = Hdot + 2/(3*a*a) * phi2ptgrad + params.all_wavenumbers2/(a*a)
    psiA = 0.5*(phi0ddot*phiA - phi0dot*phidotA)/factor
    psiB = 0.5*(phi0ddot*phiB - phi0dot*phidotB)/factor

    # Return the results
    return psiA, psiB

def construct_full_modes(fieldA, fieldB, params):
    """
    Based on fieldA and fieldB, construct the full version of the fields, using the
    coefficients in params. Works for phi/phidot/psi.

    Arguments:
        * fieldA, fieldB: The field values
        * params: EOMParameters object

    Returns: A list of arrays of the full (complex) modes for each k
        * [field0, field1[0:2]]
    """
    # Start by splitting the data into l=0 and l=1 modes
    fieldA0 = fieldA[:params.k_modes]
    fieldA1 = fieldA[params.k_modes:]
    fieldB0 = fieldB[:params.k_modes]
    fieldB1 = fieldB[params.k_modes:]

    field0 = fieldA0 * params.poscoeffs[0] + fieldB0 * params.velcoeffs[0]
    field1 = [fieldA1 * params.poscoeffs[1][i] + fieldB1 * params.velcoeffs[1][i] for i in range(3)]

    return field0, field1

def compute_hartree(phiA, phidotA, phiB, phidotB, params):
    """
    Based on phi and phidot, construct the Hartree corrections.

    Arguments:
        * phiA, phidotA, phiB, phidotB: The field values including coefficients, split
                                        into lists by l values
        * params: EOMParameters object

    Returns: A tuple
        * (phi2pt, phi2ptdt, phi2ptgrad)
    """
    # Compute the fields, with associated coefficients
    fullphi0, fullphi1 = construct_full_modes(phiA, phiB, params)
    fullphidot0, fullphidot1 = construct_full_modes(phidotA, phidotB, params)

    # Compute the 2-point functions
    phi2pt = compute_2pt(fullphi0, params)
    phi2ptdt = compute_2ptdt(fullphidot0, params)
    phi2ptgrad = compute_2ptgrad(fullphi1, params)

    # Return results
    return (phi2pt, phi2ptdt, phi2ptgrad)

def compute_2ptpsi(psiA, psiB, params):
    """
    Based on psi and psidot, construct the two points function for psi.

    Arguments:
        * psiA, psiB: The field values including coefficients, split
                      into lists by l values
        * params: EOMParameters object

    Returns: psi2pt
    """
    # Compute the fields, with associated coefficients
    fullpsi0, fullpsi1 = construct_full_modes(psiA, psiB, params)

    # Compute the 2-point functions
    psi2pt = compute_2pt(fullpsi0, params)

    # Return results
    return psi2pt

def compute_2ptdt(fullphidot0, params):
    r"""
    Computes the two point function <(\delta \dot{\hat{\phi}})^2>

    Arguments:
        * fullphidot0: Array of all phidot_k values for ell = 0
        * params: EOMParameters object

    Returns <(\delta \dot{\hat{\phi}})^2>
    == \pi/(2 R^3) \sum_n n^2 |\dot{\phi}|^2 e^{-k^2/(2 \kappa^2)}
    == 1/(2 \pi R) \sum_k k^2 |\dot{\phi}|^2 e^{-k^2/(2 \kappa^2)}
    """
    squares = np.real(fullphidot0 * np.conj(fullphidot0))
    result = np.sum(params.k2_grids[0] * squares * params.gaussian_profile[0])
    result /= (2*pi*params.Rmax)
    return result

def compute_2pt(fullphi0, params):
    r"""
    Computes the two point function <(\delta \hat{\phi})^2>

    Arguments:
        * fullphi0: Array of all phi_k values for ell = 0
        * params: EOMParameters object

    Returns <(\delta \hat{\phi})^2>
    == \pi/(2 R^3) \sum_n n^2 |\phi|^2 e^{-k^2/(2 \kappa^2)}
    == 1/(2 \pi R) \sum_k k^2 |\phi|^2 e^{-k^2/(2 \kappa^2)}
    """
    squares = np.real(fullphi0 * np.conj(fullphi0))
    result = np.sum(params.k2_grids[0] * squares * params.gaussian_profile[0])
    result /= (2*pi*params.Rmax)
    return result

def compute_2ptgrad(fullphi1, params):
    r"""
    Computes the two point function <(\grad \delta \hat{\phi})^2>
    == h^{ij} <\partial_i \delta \hat{\phi} \partial_j \delta \hat{\phi}>

    Arguments:
        * fullphi1: Array of all phi_k values for ell = 1
        * params: EOMParameters object

    Returns <(\grad \delta \hat{\phi})^2>
    == 1/(6 \pi R^3) \sum_k \sum_m k^2 / |j_2(kR)|^2 * |\phi|^2 * e^{-k^2/(2 \kappa^2)}

    Note - no 1/a^2 included here
    """
    # Do the m sum
    sumsquare = np.real(sum(fullphi1[m] * np.conj(fullphi1[m]) for m in range(3)))
    # Now do the k sum
    result = np.sum(params.denom_fac[1] * params.k2_grids[1] * sumsquare
                    * params.gaussian_profile[1])
    result /= (6*pi*params.Rmax**3)
    return result
