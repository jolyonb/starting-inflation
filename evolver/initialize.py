#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initialize

Initializes parameters for a run
"""
import numpy as np
from scipy.special import spherical_jn
from evolver.besselroots import get_jn_roots
from evolver.integrator import AbstractModel, AbstractParameters
from evolver.eoms import (eoms, compute_hubble, compute_initial_psi, compute_hartree,
                          compute_2ptpsi,
                          compute_rho, compute_deltarho2, compute_hubble_constraint_viol,
                          compute_all)

class Parameters(AbstractParameters):
    """
    Stores all settings for the evolution, along with the quantities computed from them.
    """
    def __init__(self, Rmax, k_modes, hartree, model, kappa, filename):
        """
        Construct grids based on the settings

        Arguments:
            * Rmax: The domain boundary (called R in the notes)
            * k_modes: This is the number of k modes we will use for ell = 0
            * hartree: Whether or not to compute Hartree corrections
            * model: InflationModel class
            * kappa: Regularization wavenumber
            * filename: The output file to write to
        """
        # Store the basic values
        self.Rmax = Rmax
        self.k_modes = k_modes
        self.hartree = hartree
        self.model = model
        self.kappa = kappa
        self.filename = filename
        self.halt = False
        self.haltmsg = ""

        # Initialize evolution
        self.slowroll = False

        # Set up the wavenumber grids
        self.k_grids = get_jn_roots(1, self.k_modes)
        assert len(self.k_grids[0]) == k_modes
        assert len(self.k_grids[1]) == k_modes - 1

        # Iterate through all wavenumbers, dividing by Rmax to turn the roots
        # into wavenumbers
        for ell in range(2):
            self.k_grids[ell] /= Rmax

        # Construct wavenumbers squared
        self.k2_grids = [grid*grid for grid in self.k_grids]

        # Compute the normalizations associated with each wavenumber
        self.normalizations = [None for ell in range(2)]
        factor = np.sqrt(2) / Rmax**1.5
        for ell in range(2):
            self.normalizations[ell] = factor/np.abs(spherical_jn(ell+1, self.k_grids[ell] * Rmax))

        # Compute 1 / |j_{ell + 1}(k R)|^2, which we'll need for gradient Hartree corrections
        self.denom_fac = [None for ell in range(2)]
        for ell in range(2):
            self.denom_fac[ell] = 1 / np.abs(spherical_jn(ell+1, self.k_grids[ell] * Rmax))**2

        # Make a tuple storing the number of wavenumbers for each ell
        self.num_wavenumbers = tuple(len(self.k_grids[i]) for i in range(2))

        # Construct the total number of wavenumbers
        self.total_wavenumbers = sum(self.num_wavenumbers)

        # Construct a list of all wavenumbers in order
        self.all_wavenumbers = np.concatenate(self.k_grids)
        self.all_wavenumbers2 = self.all_wavenumbers**2

        # Compute Gaussian suppression for wavenumbers
        factor = kappa*kappa*2
        self.gaussian_profile = [
            np.exp(-self.k_grids[0]*self.k_grids[0]/factor),
            np.exp(-self.k_grids[1]*self.k_grids[1]/factor)
        ]

    def write_info(self, data):
        """
        Writes initialization info to file.
        """
        unpacked_data = unpack(data, self.total_wavenumbers)
        a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data

        (rho, deltarho2, H, adot, Hdot, addot, epsilon,
         phi0ddot, phi2pt, phi2ptdt, phi2ptgrad) = compute_all(unpacked_data, self)

        ratio = phi2pt/(self.kappa**2/4/np.pi**2)

        self.write_info_line("Evolution Information")
        self.write_info_line("Number of l=0 modes: {}".format(self.k_modes))
        self.write_info_line("Number of l=1 modes: {}".format(self.k_modes - 1))
        self.write_info_line("Hartree corrections on: {}".format(self.hartree))
        self.write_info_line("R_max: {}".format(self.Rmax))
        self.write_info_line("kappa: {}".format(self.kappa))
        self.write_info_line("Model: {}".format(type(self.model).__name__))
        self.write_info_line(self.model.info())

        self.write_info_line("Initial phi0: {}".format(phi0))
        self.write_info_line("Initial phi0dot: {}".format(phi0dot))
        self.write_info_line("Initial H: {}".format(H))

        self.write_info_line("Initial rho: {}".format(rho))
        self.write_info_line("Initial deltarho2: {}".format(deltarho2))
        self.write_info_line("deltarho2/rho: {}".format(deltarho2/rho))

        self.write_info_line("Initial <deltaphi^2>: {}".format(phi2pt))
        self.write_info_line(r"<deltaphi^2> / (H^2 \bar\kappa^2 / (4 pi^2)): {}".format(ratio))

class Model(AbstractModel):
    """The model to be integrated"""

    def derivatives(self, time, data):
        """
        Computes derivatives for evolution

        Arguments:
            * time: The current time
            * data: The current data as a numpy array

        Returns:
            * derivatives: The derivatives given the current data and time. Must be the
                           same size numpy array as data.
        """
        # Unpack the data
        unpacked_data = unpack(data, self.parameters.total_wavenumbers)
        _, _, _, phi0dot, _, phidotA, _, _, phidotB, _ = unpacked_data

        # Use the equations of motion
        (adot, addot, epsilon, phi0ddot,
         phiddotA, psidotA, phiddotB, psidotB) = eoms(unpacked_data,
                                                      self.parameters,
                                                      time)

        # Check for slowroll
        if epsilon < 0.1:
            self.parameters.slowroll = True
        elif self.parameters.slowroll and epsilon >= 1:
            self.parameters.halt = True
            self.parameters.haltmsg = "Inflation has ended"

        # Combine everything into a single array
        return eqpack(adot, addot, phi0dot, phi0ddot,
                      phidotA, phiddotA, psidotA,
                      phidotB, phiddotB, psidotB)

    def solout(self, t, data):
        if self.parameters.halt:
            return -1
        return 0

    def write_extra_data(self):
        """
        Writes auxiliary data to the second output file

        Returns: None
        """
        unpacked_data = unpack(self.data, self.parameters.total_wavenumbers)
        a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data

        (rho, deltarho2, H, adot, Hdot, addot, epsilon, phi0ddot,
         phi2pt, phi2ptdt, phi2ptgrad) = compute_all(unpacked_data, self.parameters)

        # compute 2pt function of Psi
        psi2pt = compute_2ptpsi(psiA, psiB, self.parameters)

        hubble_violation = compute_hubble_constraint_viol(a, adot, rho, deltarho2)

        model = self.parameters.model
        V = model.potential(phi0)
        Vd = model.dpotential(phi0)
        Vdd = model.ddpotential(phi0)
        Vddd = model.dddpotential(phi0)
        Vdddd = model.ddddpotential(phi0)

        extradata = [H, Hdot, addot, phi0ddot, phi2pt, phi2ptdt, phi2ptgrad, psi2pt,
                     rho, deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd]

        sep = self.separator
        self.parameters.f2.write(str(self.time) + sep + sep.join(map(str, extradata)) + "\n")

    def compute_timestep(self):
        """Computes the time step at this point in the evolution"""
        factor1, factor2 = self.timestepinfo
        # We want to take factor timesteps in each e-fold, roughly
        # Delta t = Delta a / adot
        # Change in a we want to see is 1 efold / factor
        # 1 efold = e * a
        # Delta a = (e-1) * a / factor
        a = self.data[0]
        adot = np.exp(self.data[1])
        H = adot / a

        # Get the shortest wavelength
        lamda = 2*np.pi/self.parameters.k_grids[0][-1]
        # Inflate it
        lamda *= a
        # Get the horizon scale
        horizon = 1/H
        # Are we inside or outside the horizon?
        inside = False if lamda > 10 * horizon else True
        if inside:
            factor = factor1
        else:
            factor = factor2

        # Compute the timestep
        timestep = 1.71828 * a / factor / adot
        # print(timestep, 1.0 * np.sqrt(1e-6/self.parameters.model.lamda))
        return timestep
        # Old code:
        # return 1.0 * np.sqrt(1e-6/self.parameters.model.lamda)

def make_initial_data(phi0, phi0dot, k_modes, hartree, model,
                      filename, Rmaxfactor=2, kappafactor=20, l1modeson=True):
    """
    Constructs parameters and initial data for the evolution

    Arguments:
        * phi0: Starting value for the background field
        * phi0dot: Starting value for the time derivative of the background field
        * k_modes: This is the number of k modes we will use for ell = 0
        * hartree: Whether or not to compute Hartree corrections
        * model: An initialized InflationModel class
        * filename: The output file to write to
        * filename2: The output file for auxiliary variables
        * Rmaxfactor: The factor by which to increase Rmax from the initial Hubble radius
        * kappafactor: Sets the scale of the regulator in terms of Hubble
        * l1modeson: If set to False, initializes all l=1 modes with zero coefficients

    Returns: (initial_data, params)
        * initial_data: An numpy array containing all of the initial data for the simulation
        * params: Parameters class containing all of the parameters for the simulation

    The data in initial_data is stored as:
    [a, adot, phi0, phi0dot, phi^A, phidot^A, psi^A, phi^B, phidot^B, psi^B]
    where the last six entries are vectors with 2n-1 entries, where n is the number of
    ell = 0 entries. Note that A modes begin with unit position, and B modes begin with
    unit velocity.
    """
    # # Seed the random number generator if needed
    # if seed is None and randomize:
    #     seed = random.randrange(sys.maxsize)
    # elif not randomize:
    #     seed = None

    # Estimate Hubble radius from background quantities
    rho = compute_rho(phi0, phi0dot, model)
    H0 = compute_hubble(rho, 0)

    # Construct Rmax and kappa
    Rmax = Rmaxfactor / H0
    kappa = kappafactor * H0

    # Construct the parameters
    params = Parameters(Rmax, k_modes, hartree, model, kappa, filename)

    # How many fields do we have?
    numfields = params.total_wavenumbers

    # Construct the initial conditions

    # We just take the initial data for the background field values from the arguments
    # phi0 = phi0
    # phi0dot = phi0dot

    # The starting value of the scale factor
    a = 1

    # Generate coefficients for all of the modes
    poscoeffs = [None, [None]*3]
    velcoeffs = [None, [None]*3]

    # Bunch-Davies initial conditions
    poscoeffs[0] = 1 / np.sqrt(2*params.k_grids[0])
    velcoeffs[0] = np.sqrt(params.k_grids[0] / 2) * (-1j - H0 / params.k_grids[0])
    if l1modeson:
        for i in range(3):
            poscoeffs[1][i] = 1 / np.sqrt(2*params.k_grids[1])
            velcoeffs[1][i] = np.sqrt(params.k_grids[1] / 2) * (-1j - H0 / params.k_grids[1])
    else:
        for i in range(3):
            poscoeffs[1][i] = np.zeros_like(params.k_grids[1])
            velcoeffs[1][i] = np.zeros_like(params.k_grids[1])

    # Attach these to params
    params.poscoeffs = poscoeffs
    params.velcoeffs = velcoeffs

    # Everything else to initialize is for the fields phi_{nlm}
    # Go and initialize the fields
    phiA = np.ones(numfields)
    phidotA = np.zeros(numfields)
    phiB = np.zeros(numfields)
    phidotB = np.ones(numfields)

    # Compute the Hartree corrections
    if hartree:
        phi2pt, phi2ptdt, phi2ptgrad = compute_hartree(phiA, phidotA, phiB, phidotB, params)
        deltarho2 = compute_deltarho2(a, phi0, phi2pt, phi2ptdt, phi2ptgrad, model)
    else:
        phi2pt, phi2ptdt, phi2ptgrad = (0, 0, 0)
        deltarho2 = 0

    # Compute adot from Hubble
    adot = compute_hubble(rho, deltarho2) * a

    # Now compute the initial values for the psi fields
    psiA, psiB = compute_initial_psi(a, adot, phi0, phi0dot,
                                     phiA, phidotA, phiB, phidotB,
                                     phi2pt, phi2ptdt, phi2ptgrad, params)

    # Pack all the initial data together
    data = pack(a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB)

    # Return the data
    return params, data

def pack(a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB):
    """
    Pack all field values into a data array for integration.

    Arguments:
        * a, adot, phi0, phi0dot: Respective values to pack
        * phiA, phidotA, psiA, phiB, phidotB, psiB: Arrays of values for each wavenumber

    Returns:
        * data: A numpy array containing all data
    Note that adot is stored as a logarithm
    """
    background = np.array([a, np.log(adot), phi0, phi0dot])
    return np.concatenate((background, phiA, phidotA, psiA, phiB, phidotB, psiB))

def eqpack(adot, addot, phi0dot, phi0ddot,
           phidotA, phiddotA, psidotA,
           phidotB, phiddotB, psidotB):
    """
    Pack all field values into a data array for integration.

    Arguments:
        * adot, addot, phi0dot, phi0ddot: Respective values to pack
        * phidotA, phiddotA, psidotA,
          phidotB, phiddotB, psidotB: Arrays of values for each wavenumber

    Returns:
        * data: A numpy array containing all data
    Note that adot is stored as d(ln(adot))dt = addot/adot
    """
    background = np.array([adot, addot/adot, phi0dot, phi0ddot])
    return np.concatenate((background, phidotA, phiddotA, psidotA,
                           phidotB, phiddotB, psidotB))

def unpack(data, total_wavenumbers):
    """
    Unpack field values from a data array into a meaningful data structure.
    This reverses the operations performed in pack.

    Arguments:
        * data: The full array of all fields, their derivatives, and auxiliary values
        * total_wavenumbers: The total number of modes being packed/unpacked

    Returns:
        * (a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB)
          where these quantities are as initialized in make_initial_data
    """
    # Grab a, adot, phi0 and phi0dot
    a = data[0]
    adot = np.exp(data[1])  # Note this needs to be exponentiated
    phi0 = data[2]
    phi0dot = data[3]

    # How many fields do we have here?
    numfields = total_wavenumbers

    # Unpack all the data
    fields = data[4:]
    phiA = fields[0:numfields]
    phidotA = fields[numfields:2*numfields]
    psiA = fields[2*numfields:3*numfields]
    phiB = fields[3*numfields:4*numfields]
    phidotB = fields[4*numfields:5*numfields]
    psiB = fields[5*numfields:6*numfields]

    # Return the results
    return a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB
