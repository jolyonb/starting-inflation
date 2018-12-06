# -*- coding: utf-8 -*-
"""
initialize

Initializes parameters for a run
"""
import numpy as np
from scipy.special import spherical_jn
from evolver.besselroots import get_jn_roots
from evolver.utilities import pack
from evolver.eoms import (compute_hubble, compute_initial_psi, compute_hartree,
                          compute_rho, compute_deltarho2, EOMParameters)

def create_package(phi0,
                   phi0dot,
                   infmodel,
                   end_time,
                   basefilename,
                   num_k_modes=40,
                   hartree=True,
                   Rmaxfactor=2,
                   kappafactor=20,
                   l1modeson=True,
                   perform_run=True,
                   **kwargs):
    """
    Packages all the settings that need to be set for a run into a dictionary.
    This package can then be initialized using create_parameters.

    A package can be used multiple times, and can be easily modified to
    provide different initial conditions.

    Some arguments are mandatory, while others are optional. Any extra
    package settings can be specified by keyword arguments.

    Arguments:
        * phi0: Starting value for the background field
        * phi0dot: Starting value for the time derivative of the background field
        * infmodel: Instance of an InflationModel class
        * end_time: Maximum time to evolve to
        * basefilename: The base name of the desired output file
                        (different extensions will be added)
        * num_k_modes: This is the number of k modes we will use for ell = 0
        * hartree: Whether or not to compute Hartree corrections
        * Rmaxfactor: The factor by which to increase Rmax from the initial Hubble radius
        * kappafactor: Sets the scale of the regulator in terms of Hubble
        * l1modeson: If set to False, initializes all l=1 modes with zero coefficients
        * perform_run: Do we evolve, pr just set everything up?
    """
    package = {
        'phi0': phi0,
        'phi0dot': phi0dot,
        'infmodel': infmodel,
        'end_time': end_time,
        'basefilename': basefilename,
        'num_k_modes': num_k_modes,
        'hartree': hartree,
        'Rmaxfactor': Rmaxfactor,
        'kappafactor': kappafactor,
        'l1modeson': l1modeson,
        'perform_run': perform_run,
        **kwargs
    }
    return package

def create_parameters(package):
    """
    Takes in a package and generates the parameters dictionary required by
    a Model. This involves constructing grids and initial conditions. Once this
    dictionary has been generated, it should not be modified, as doing so
    may cause inconsistencies.
    """
    # Start by copying everything in the package
    parameters = {**package}

    #########################
    # Background quantities #
    #########################
    phi0 = parameters['phi0']
    phi0dot = parameters['phi0dot']
    infmodel = parameters['infmodel']
    rho = compute_rho(phi0, phi0dot, infmodel)
    # Estimate Hubble
    H0 = compute_hubble(rho, 0)

    # Construct Rmax and kappa
    parameters['Rmax'] = Rmax = parameters['Rmaxfactor'] / H0
    parameters['kappa'] = kappa = parameters['kappafactor'] * H0

    ####################
    # Wavenumber grids #
    ####################
    num_k_modes = parameters['num_k_modes']
    k_grids = get_jn_roots(1, num_k_modes)

    # Iterate through all wavenumbers, dividing by Rmax
    # to turn the roots into wavenumbers
    for ell in range(2):
        k_grids[ell] /= Rmax
    parameters['k_grids'] = k_grids

    # Make a tuple storing the number of wavenumbers for each ell
    parameters['total_wavenumbers'] = total_wavenumbers = 2 * num_k_modes - 1

    # Construct wavenumbers squared
    k2_grids = [grid*grid for grid in k_grids]

    # Construct a list of all wavenumbers in order, and their squares
    parameters['all_wavenumbers'] = np.concatenate(k_grids)
    all_wavenumbers2 = parameters['all_wavenumbers']**2

    # Compute Gaussian suppression for wavenumbers
    factor = 2*kappa**2
    gaussian_profile = [
        np.exp(-k_grids[0]**2 / factor),
        np.exp(-k_grids[1]**2 / factor)
    ]

    # Compute the normalizations associated with each wavenumber
    normalizations = [None for ell in range(2)]
    factor = np.sqrt(2) / Rmax**1.5
    for ell in range(2):
        normalizations[ell] = factor/np.abs(spherical_jn(ell+1, k_grids[ell] * Rmax))
    parameters['normalizations'] = normalizations

    # Compute 1 / |j_{ell + 1}(k R)|^2, which we'll need for gradient Hartree corrections
    denom_fac = [None for ell in range(2)]
    for ell in range(2):
        denom_fac[ell] = 1 / np.abs(spherical_jn(ell+1, k_grids[ell] * Rmax))**2

    ###############################
    # Construct mode coefficients #
    ###############################
    poscoeffs = [None, [None]*3]
    velcoeffs = [None, [None]*3]

    # Bunch-Davies initial conditions
    poscoeffs[0] = 1 / np.sqrt(2*k_grids[0])
    velcoeffs[0] = np.sqrt(k_grids[0] / 2) * (-1j - H0 / k_grids[0])
    if parameters['l1modeson']:
        for i in range(3):
            poscoeffs[1][i] = 1 / np.sqrt(2*k_grids[1])
            velcoeffs[1][i] = np.sqrt(k_grids[1] / 2) * (-1j - H0 / k_grids[1])
    else:
        for i in range(3):
            poscoeffs[1][i] = np.zeros_like(k_grids[1])
            velcoeffs[1][i] = np.zeros_like(k_grids[1])

    ###########################
    # Construct EOMParameters #
    ###########################
    parameters['eomparams'] = EOMParameters(Rmax,
                                            num_k_modes,
                                            parameters['hartree'],
                                            infmodel,
                                            k2_grids,
                                            all_wavenumbers2,
                                            denom_fac,
                                            gaussian_profile,
                                            poscoeffs,
                                            velcoeffs)

    ################################
    # Construct Initial Conditions #
    ################################
    # The starting value of the scale factor
    a = 1
    # phi0 and phi0dot have been obtained already

    # Initialize the fields phi_{nlm}
    phiA = np.ones(total_wavenumbers)
    phidotA = np.zeros(total_wavenumbers)
    phiB = np.zeros(total_wavenumbers)
    phidotB = np.ones(total_wavenumbers)

    # Compute the Hartree corrections
    if parameters['hartree']:
        phi2pt, phi2ptdt, phi2ptgrad = compute_hartree(phiA, phidotA,
                                                       phiB, phidotB,
                                                       parameters['eomparams'])
        deltarho2 = compute_deltarho2(a, phi0, phi2pt, phi2ptdt, phi2ptgrad, infmodel)
    else:
        phi2pt, phi2ptdt, phi2ptgrad = (0, 0, 0)
        deltarho2 = 0

    # Compute adot from Hubble
    adot = compute_hubble(rho, deltarho2) * a

    # Now compute the initial values for the psi fields
    psiA, psiB = compute_initial_psi(a, adot, phi0, phi0dot,
                                     phiA, phidotA, phiB, phidotB,
                                     phi2pt, phi2ptdt, phi2ptgrad,
                                     parameters['eomparams'])

    # Pack all the initial data together
    parameters['initial_data'] = pack(a, phi0, phi0dot,
                                      phiA, phidotA, psiA,
                                      phiB, phidotB, psiB)

    # Return everything
    return parameters
