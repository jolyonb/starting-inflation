#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform a multi-process sweep based on an initialization file.
Each process runs the same sweep.

The initialization file should be a json file with the following structure:
{
    "num_threads": [num],   (only used for hartree runs)
    "lambda": [lambda],
    "phi0": {
        "min": [minvalue],
        "max": [maxvalue],
        "steps": [stepvalue]
    },
    "phi0dot": {
        "min": [minvalue],
        "max": [maxvalue],
        "steps": [stepvalue]
    },
    "outputdir": [outputdirectory],
    "type": ["off"/"bd"/"hartree"],
    "hartree_runs": [num],
    "num_modes": [num]
}

Note that data is written to [outputdir]/[inifilename]/[threadnum]/

Run this file as
python run_aws.py inifiles/[init.json]
"""
# Used for setting up the subprocesses
import os
import time
from shutil import copyfile
import argparse
import json
import multiprocessing
# Used for setting up the runs
import pickle
from math import sqrt
import numpy as np
from evolver.integrator import Driver, Status
from evolver.initialize import create_package, create_parameters
from evolver.inflation import LambdaPhi4
from evolver.model import Model

def worker(directory, inifile):
    """Worker thread function"""
    os.chdir(directory)
    filename = "output"

    # Read quantities from the inifile json
    setting = inifile["type"]
    num_modes = inifile["num_modes"]
    hartree_runs = inifile.get("hartree_runs", 1)
    if setting == "off":
        num_modes = 2
        hartree_runs = 1
    elif setting == "bd":
        hartree_runs = 1

    # Background fields
    # Note that a step of 1 only does the start value
    lamda = inifile["lambda"]
    phi0start = inifile["phi0"]["min"]
    phi0stop = inifile["phi0"]["max"]
    phi0steps = inifile["phi0"]["steps"]
    phi0dotstart = inifile["phi0dot"]["min"]
    phi0dotstop = inifile["phi0dot"]["max"]
    phi0dotsteps = inifile["phi0dot"]["steps"]

    # Construct our steps
    phi0s = np.linspace(phi0start, phi0stop, phi0steps)
    phi0dots = np.linspace(phi0dotstart, phi0dotstop, phi0dotsteps)

    # Construct the package
    package = create_package(phi0=None,
                             phi0dot=None,
                             infmodel=LambdaPhi4(lamda=lamda),
                             end_time=5000*sqrt(1e-6/lamda),
                             basefilename=None,
                             perturbBD=False,  #
                             hartree=False,    # Hartree off settings by default
                             timestepinfo=[200, 10],
                             num_k_modes=num_modes,
                             fulloutput=False)
    # Apply settings
    if setting == "bd":
        package["hartree"] = True
    elif setting == "hartree":
        package["hartree"] = True
        package["perturbBD"] = True

    # Set up the infofile
    infofile = open(filename + "-info.txt", "w")
    infofile.write("filename\tphi0\tphi0dot\n")

    # Sweep through all the runs
    run = 0
    for _ in range(hartree_runs):
        for phi0 in phi0s:
            for phi0dot in phi0dots:
                run += 1
                fn = filename + "-{}".format(run)
                if perform_run(phi0, phi0dot, fn, package):
                    infofile.write("{}\t{}\t{}\n".format(fn, phi0, phi0dot))

    infofile.close()

def perform_run(phi0, phi0dot, filename, package):
    # Update package
    package['phi0'] = phi0
    package['phi0dot'] = phi0dot
    package['basefilename'] = filename

    # Create the model
    parameters = create_parameters(package)
    if parameters is None:
        return False
    model = Model(parameters)
    model.save(filename + ".params")

    # Construct the driver
    driver = Driver(model)

    # Perform the evolution
    driver.run()

    # Check to see what our status is
    if driver.status == Status.IntegrationError:
        with open(filename + ".info", "a") as f:
            f.write("Unable to integrate further: {}\n".format(driver.error_msg))
    elif driver.status == Status.Terminated:
        with open(filename + ".info", "a") as f:
            f.write("Evolution completed with message: {}\n".format(driver.error_msg))

    return True

def createcsv(inifile, num_threads, outputdir, csvname):
    fulldata = []
    for i in range(1, num_threads + 1):
        thisdir = os.path.join(outputdir, str(i))

        # Find all runs for this thread
        with open(os.path.join(thisdir, "output-info.txt")) as f:
            lines = f.readlines()

        # Iterate through them all
        for line in lines[1:]:
            if not line:
                continue
            fn = line.strip().split("\t")[0]
            fn = os.path.join(thisdir, fn)

            with open(fn + ".quick", "rb") as f:
                quickdata = pickle.load(f)

            # The data fields we'll pick out from quickdata
            plot_data = {
              "phi0": 0.0,
              "phi0dot": 0.0,
              "H": 0.0,
              "rho": 0.0,
              "deltarho2": 0.0,
              "phi2pt": 0.0,
              "psirms": 0.0,
              "efolds": 0.0,
              "kappa": 0.0,
              "infl": 0
            }

            for key in quickdata:
                if key in plot_data:
                    plot_data[key] = quickdata[key]

            # Add in any extra details about this run
            plot_data["filename"] = fn
            plot_data['infl'] = 1 if quickdata["inflationended"] else 0
            # Type:
            # 0 = Hartree Off
            # 1 = Bunch-Davies
            # 2 = Perturbed
            if inifile["type"] == "off":
                plot_data["type"] = 0
            elif inifile["type"] == "bd":
                plot_data["type"] = 1
            else:
                plot_data["type"] = 2

            fulldata.append(plot_data)

    # All data from the sweep is now stored in fulldata
    # Output it to a file!
    template = "{phi0},{phi0dot},{H},{rho},{deltarho2},{phi2pt},{psirms},{efolds},{kappa},{infl},{type},{filename}\n"
    with open(os.path.join(outputdir, csvname), "w") as f:
        for entry in fulldata:
            f.write(template.format(**entry))

if __name__ == "__main__":
    start = time.time()

    # Deal with command line arguments
    parser = argparse.ArgumentParser(description="Set up a multi-process sweep")
    parser.add_argument("inifile", help=".json file to read initialization from")
    parser.add_argument("-csv", help="Only construct CSV files (do not do full runs)", action="store_true", dest="csv", default=False)
    args = parser.parse_args()

    # Read the ini file
    with open(args.inifile) as f:
        inifile = json.load(f)
    _, inifilename = os.path.split(args.inifile)

    # Create the directory for outputting data
    inifiledir = inifilename
    if inifiledir.endswith(".json"):
        inifiledir = inifiledir[:-5]
    outputdir = os.path.join(inifile["outputdir"], inifiledir)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    # Figure out threading
    num_threads = inifile["num_threads"]
    if inifile["type"] != "hartree":
        num_threads = 1

    if not args.csv:
        # Perform the whole riun

        # Copy the inifile to the output directory
        copyfile(args.inifile, os.path.join(outputdir, inifilename))

        # Run all the processes
        jobs = []
        for i in range(1, num_threads + 1):
            # Make the output directory for this subprocess
            thisdir = os.path.join(outputdir, str(i))
            if not os.path.isdir(thisdir):
                os.mkdir(thisdir)

            # Create the subprocess
            subprocess = multiprocessing.Process(target=worker, args=(thisdir, inifile))
            jobs.append(subprocess)
            subprocess.start()

        # Wait for all processes to complete
        for subprocess in jobs:
            subprocess.join()

    # Create the CSV file for this run
    createcsv(inifile, num_threads, outputdir, inifiledir + ".csv")

    # Report on time taken
    end = time.time()
    with open(os.path.join(outputdir, "time.txt"), "w") as f:
        f.write("Finished in {} s".format(round(end - start, 4)))
