# -*- coding: utf-8 -*-
"""
model.py

Defines the model for evolution, built on top of AbstractModel
"""
from math import pi
from evolver.eoms import eoms, compute_all, compute_2ptpsi
from evolver.integrator import AbstractModel
from evolver.utilities import pack, unpack

class Model(AbstractModel):
    """Set up the model for evolution"""

    def __init__(self, run_params):
        """
        Set defaults and saves parameters for rapid access
        """
        # Defaults
        defaults = {
            'timestepinfo': [200, 10]
        }
        defaults.update(run_params)
        if not run_params['perform_run']:
            defaults['end_time'] = defaults['start_time']

        # Call the AbstractModel initializer
        super().__init__(defaults)

        # Save parameters for future use (faster than dictionary access)
        self.basefilename = self.parameters['basefilename']
        self.total_wavenumbers = self.parameters['total_wavenumbers']
        self.timestepinfo = self.parameters['timestepinfo']
        self.k_grids = self.parameters['k_grids']
        self.infmodel = self.parameters['infmodel']
        self.eomparams = self.parameters['eomparams']

        # Set internal flags
        self.slowroll = False

    def begin(self):
        """Open file handles for evolution and write initial description"""
        # Set up the file handles for writing data
        self.datafile = open(self.basefilename + ".dat", "w")
        self.extrafile = open(self.basefilename + ".dat2", "w")

        # Write the initial description to file
        params = self.eomparams
        unpacked_data = unpack(self.initial_data, self.total_wavenumbers)
        a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data

        (rho, deltarho2, H, adot, Hdot, addot, epsilon,
         phi0ddot, phi2pt, phi2ptdt, phi2ptgrad) = compute_all(unpacked_data, params)

        ratio = phi2pt/(params.kappa**2/4/pi**2)

        with open(self.basefilename + ".info", "w") as f:
            f.write(f"""Evolution Parameters and Initial Conditions
Number of l=0 modes: {params.k_modes}
Number of l=1 modes: {params.k_modes - 1}
Hartree corrections on: {params.hartree}
R_max: {params.Rmax}
kappa: {params.kappa}
Model: {type(params.model).__name__}
{params.model.info()}
Initial phi0: {phi0}
Initial phi0dot: {phi0dot}
Initial H: {H}
Initial rho: {rho}
Initial deltarho2: {deltarho2}
deltarho2/rho: {deltarho2/rho}
Initial <deltaphi^2>: {phi2pt}
""")
            f.write(r"<deltaphi^2> / (H^2 \bar\kappa^2 / (4 pi^2)): {}\n".format(ratio))

    def cleanup(self):
        """A function that is called after evolution finishes"""
        self.datafile.close()
        self.extrafile.close()

    def derivatives(self, time, data):
        """Computes derivatives for evolution"""
        # Unpack the data
        unpacked_data = unpack(data, self.total_wavenumbers)

        # Use the equations of motion
        (adot, epsilon, phi0dot, phi0ddot, phidotA, phiddotA,
         psidotA, phidotB, phiddotB, psidotB) = eoms(unpacked_data,
                                                     self.eomparams,
                                                     time)

        # Check for slowroll
        if epsilon < 0.1:
            self.slowroll = True
        elif self.slowroll and epsilon >= 1:
            self.halt = True
            self.haltmsg = "Inflation has ended"

        # Recombine the derivatives
        return pack(adot, phi0dot, phi0ddot,
                    phidotA, phiddotA, psidotA,
                    phidotB, phiddotB, psidotB)

    def compute_timestep(self, time, data):
        """Compute the desired timestep at this point in the evolution"""
        factor1, factor2 = self.timestepinfo
        # We want to take factor timesteps in each e-fold, roughly
        # Delta t = Delta a / adot
        # Change in a we want to see is 1 efold / factor
        # 1 efold = e * a
        # Delta a = (e-1) * a / factor
        a = data[0]
        unpacked_data = unpack(data, self.total_wavenumbers)
        _, _, H, adot, _, _, _, _, _, _, _ = compute_all(unpacked_data, self.eomparams)

        # Get the shortest wavelength
        lamda = 2 * pi / self.k_grids[0][-1]
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
        return timestep

    def write_data(self, time, data):
        """
        A function that is called between timesteps (and also at the
        start/end of integration), when data is ready for writing.
        """
        # Write the raw data
        text = self.format_data(time, data)
        self.datafile.write(text)

        # Compute derived data
        unpacked_data = unpack(data, self.total_wavenumbers)
        a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpacked_data

        (rho, deltarho2, H, adot, Hdot, addot, epsilon, phi0ddot,
         phi2pt, phi2ptdt, phi2ptgrad) = compute_all(unpacked_data, self.eomparams)

        psi2pt = compute_2ptpsi(psiA, psiB, self.eomparams)
        V = self.infmodel.potential(phi0)
        Vd = self.infmodel.dpotential(phi0)
        Vdd = self.infmodel.ddpotential(phi0)
        Vddd = self.infmodel.dddpotential(phi0)
        Vdddd = self.infmodel.ddddpotential(phi0)

        extradata = [H, Hdot, addot, phi0ddot, phi2pt, phi2ptdt, phi2ptgrad, psi2pt,
                     rho, deltarho2, epsilon, V, Vd, Vdd, Vddd, Vdddd]

        # Write the derived data
        text = self.format_data(time, extradata)
        self.extrafile.write(text)
