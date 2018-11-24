#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
from evolver.analysis import load_data, analyze

# Give it the info file to read from
filename = "data/output-info.txt"

# Suck up the data
with open(filename) as f:
    lines = f.readlines()

# Process it into nice lines
data = []
for line in lines[1:]:
    if line:
        fn, phi0, phi0dot = line.strip().split("\t")
        phi0 = float(phi0)
        phi0dot = float(phi0dot)
        data.append((fn, phi0, phi0dot))

# Read the data files for each run in the sweep
for file, phi0, phi0dot in data:
    results = load_data(file)
    details = analyze(results["a"], results["adot"], results["addot"])
    print(details["efolds"])
