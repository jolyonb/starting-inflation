#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the evolution of background fields + perturbative modes in an inflationary picture
"""
import sys
import numpy as np
import random
from math import sqrt
from scipy.special import spherical_jn
from besselroots import get_jn_roots
from integrator import AbstractModel, AbstractParameters
from eoms import (eoms, compute_hubble, compute_initial_psi, compute_hartree,
                  compute_hubbledot, compute_phi0ddot, compute_rho, compute_deltarho2, hartree_kinetic, hartree_gradient, hartree_potential,
                  compute_hubble_constraint_viol, potential, dpotential, ddpotential,
                  dddpotential, ddddpotential, compute_phi0ddot)

class Parameters(AbstractParameters):
    """
    Stores all settings for the evolution, along with the quantities computed from them.
    """
    def __init__(self, Rmax, k_modes, hartree, lamda, kappa, filename, filename2, seed, solution):
        """
        Construct grids based on the settings

        Arguments:
            * Rmax: The domain boundary (called R in the notes)
            * k_modes: This is the number of k modes we will use for ell = 0
            * hartree: Whether or not to compute Hartree corrections
            * lamda: The parameter associated with the potential
            * kappa: Regularization wavenumber
            * filename: The output file to write to
            * filename2: The output file for auxiliary variables
            * seed: Seed used for random number generation
            * solution: Which solution to use for the scaling (1 = kinetic, 2 = potential)

        Returns: None
        """
        # Store the basic values
        self.Rmax = Rmax
        self.k_modes = k_modes
        self.hartree = hartree
        self.lamda = lamda
        self.kappa = kappa
        self.filename = filename
        self.filename2 = filename2
        self.seed = seed

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
            self.normalizations[ell] = factor / np.abs(spherical_jn(ell+1, self.k_grids[ell] * Rmax))

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
        factor = kappa*kappa*4
        self.gaussian_profile = [
            np.exp(-self.k_grids[0]*self.k_grids[0]/factor),
            np.exp(-self.k_grids[1]*self.k_grids[1]/factor)
        ]

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
        _, adot, _, phi0dot, _, phidotA, _, _, phidotB, _ = unpacked_data

        # Use the equations of motion
        addot, phi0ddot, phiddotA, psidotA, phiddotB, psidotB = eoms(unpacked_data, self.parameters, time)

        # Combine everything into a single array
        derivs = pack(adot, addot, phi0dot, phi0ddot,
                      phidotA, phiddotA, psidotA,
                      phidotB, phiddotB, psidotB)

        assert len(derivs) == len(data)

        return derivs

    def write_extra_data(self):
        """
        Writes auxiliary data to the second output file

        Returns: None
        """
        unpacked_data = unpack(self.data, self.parameters.total_wavenumbers)
        a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data
        H = adot/a

        hpotential0, hgradient0, hkinetic0 = compute_hartree(phiA, phidotA, phiB, phidotB, self.parameters)
        if self.parameters.hartree:
            hpotential, hgradient, hkinetic = (hpotential0, hgradient0, hkinetic0)
        else:
            hpotential, hgradient, hkinetic = (0, 0, 0)

        Hdot = compute_hubbledot(a, phi0dot, hkinetic, hgradient, self.parameters)
        addot = a*(Hdot + H*H)
        phi0ddot = compute_phi0ddot(phi0, phi0dot, H, hpotential, self.parameters)

        rho = compute_rho(phi0, phi0dot, self.parameters)
        deltarho2 = compute_deltarho2(a, phi0, hkinetic0, hgradient0, hpotential0, self.parameters)


        hubble_violation = compute_hubble_constraint_viol(a, adot, phi0, phi0dot,
                                                          hpotential, hkinetic, hgradient,
                                                          self.parameters)

        # psi_violation = compute_psi_constraint_viol(a, adot, phi0, phi0dot, phiA,phidotA, phiB, phidotB, psiA, psiB, hkinetic, hgradient, hpotential, self.parameters)

        V = potential(phi0, self.parameters)
        Vd = dpotential(phi0, self.parameters)
        Vdd = ddpotential(phi0, self.parameters)
        Vddd = dddpotential(phi0, self.parameters)
        Vdddd = ddddpotential(phi0, self.parameters)

        extradata = [H, Hdot, addot, phi0ddot, hpotential0, hgradient0, hkinetic0,
                     rho, deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd]

        sep = self.separator
        self.parameters.f2.write(str(self.time) + sep + sep.join(map(str, extradata)) + "\n")


def make_initial_data(phi0, phi0dot, Rmax, k_modes, hartree, lamda, kappa,
                      perturbed_ratio, randomize, seed, solution,
                      filename, filename2):
    """
    Constructs parameters and initial data for the evolution

    Arguments:
        * phi0: Starting value for the background field
        * phi0dot: Starting value for the time derivative of the background field
        * Rmax: The domain boundary (called R in the notes)
        * k_modes: This is the number of k modes we will use for ell = 0
        * hartree: Whether or not to compute Hartree corrections
        * lamda: The parameter associated with the potential
        * kappa: Regularization wavenumber
        * perturbed_ratio: Desired ratio of delta rho_2/rho_0
        * randomize: Use random initialized data
        * seed: Seed for random initial data (use None for random)
        * solution: Which solution to use for the scaling (0 = random, 1 = kinetic, 2 = potential)
        * filename: The output file to write to
        * filename2: The output file for auxiliary variables

    Returns:
        * numpy array: An array containing all of the initial data for the simulation

    The data is stored as:
    [a, adot, phi0, phi0dot, phi^A, phidot^A, psi^A, phi^B, phidot^B, psi^B]
    where the last six  entries are vectors with 2n-1 entries, where n is the number of
    ell = 0 entries. Note that A modes begin with unit position, and B modes begin with
    unit velocity.
    """
    # Seed the random number generator if needed
    if seed is None and randomize:
        seed = random.randrange(sys.maxsize)
    elif not randomize:
        seed = None
    if solution not in [0, 1, 2]:
        solution = 0
    # Construct the parameters
    params = Parameters(Rmax, k_modes, hartree, lamda, kappa, filename, filename2, seed, solution)

    # We just take the initial data for the background field values from the arguments
    # The starting value of the scale factor
    a = 1

    # Go and generate random coefficients for all of the modes
    # def make_random_complex(num):
    #     """Generate a list of num random complex fields with components between -1 and 1"""
    #     return np.random.rand(num) + 1j*np.random.rand(num)
    poscoeffs = [None, None]
    velcoeffs = [None, None]

    if randomize:
        random.seed(seed)
        # Initial data looks like
        # delta_k = A + i B
        # delta_kdot = C + I D
        # B = 0 by gauge choice
        # A = - 1 / D by quantization condition
        # C and D are chosen randomly
        # C is chosen between -1 and 1
        # D is chosen from [-1, -0.5] and [0.5, 1]

        def random_arrays(modes):
            """Helper function to generate random initial data"""
            Darray = np.array([random.uniform(0.5, 1) * random.randrange(-1, 2, 2)
                               for i in range(modes)])
            Carray = np.array([random.uniform(-1, 1)
                               for i in range(modes)])
            Barray = np.array([0 for i in range(modes)])
            Aarray = -1 / Darray
            return Aarray + 1j*Barray, Carray + 1j*Darray

        poscoeffs[0], velcoeffs[0] = random_arrays(params.k_modes)
        poscoeff1s[1] = [None] * 3
        velcoeffs[1] = [None] * 3
        for i in range(3):
            P, V = random_arrays(params.k_modes - 1)
            poscoeffs[1][i] = P
            velcoeffs[1][i] = V
    else:
        # Bunch-Davies initial conditions
        H0back = np.sqrt((1.0/3.0)*compute_rho(phi0, phi0dot, params))
        poscoeffs[0] = 1/np.sqrt(2*params.k_grids[0])
        velcoeffs[0] = (np.sqrt(params.k_grids[0])/2)*(-1j-(H0back/params.k_grids[0]))
        # poscoeffs[1] = [np.ones_like(params.k_grids[1])] * 3
        # velcoeffs[1] = [1j*np.ones_like(params.k_grids[1])] * 3
        poscoeffs[1] = [0*np.ones_like(params.k_grids[1])] * 3
        velcoeffs[1] = [0*1j*np.ones_like(params.k_grids[1])] * 3
       

        # debug 20. Nov 2018 to check why Bunch-Davies is returning nonsense
        print ("k_grids[0]:",params.k_grids[0])
        print ("k_grids[1]:",params.k_grids[1])

    # Attach these to params
    params.poscoeffs = poscoeffs
    params.velcoeffs = velcoeffs

    # Everything else to initialize is for the fields phi_{nlm}
    # How many fields do we have here?
    numfields = params.total_wavenumbers

    # Go and initialize the fields
    phiA = np.ones(numfields)
    phidotA = np.zeros(numfields)
    phiB = np.zeros(numfields)
    phidotB = np.ones(numfields)

    # Compute the Hartree corrections
    if params.hartree:
        hpotential, hgradient, hkinetic = compute_hartree(phiA, phidotA, phiB, phidotB, params)
    else:
        hpotential, hgradient, hkinetic = (0, 0, 0)

    # We compute its time derivative by using the Friedmann equation + corrections
    adot = compute_hubble(a, phi0, phi0dot, hpotential, hkinetic, hgradient, params) * a

    # Now compute the initial values for the psi fields
    psiA, psiB = compute_initial_psi(a, adot, phi0, phi0dot,
                                     phiA, phidotA,
                                     phiB, phidotB,
                                     hkinetic, hgradient, hpotential,
                                     params)

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
    """
    background = np.array([a, adot, phi0, phi0dot])
    return np.concatenate((background, phiA, phidotA, psiA, phiB, phidotB, psiB))

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
    adot = data[1]
    phi0 = data[2]
    phi0dot = data[3]

    # print (adot/a)

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

