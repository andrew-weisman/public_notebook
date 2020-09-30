#!/bin/bash

# TEST INTERACTIVE 2

ROOT=/gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0
SWIFT=$ROOT/swift-t/2020-09-02

PATH=$SWIFT/stc/bin:$PATH
PATH=$SWIFT/turbine/bin:$PATH

export LD_LIBRARY_PATH=/sw/summit/gcc/6.4.0/lib64:$LD_LIBRARY_PATH

# Suppress warning:
OMP_NUM_THREADS=1

# Works for Andrew
# stc hello.swift
# jsrun turbine-pilot hello.tic

# Works for Andrew
# stc b.swift
# jsrun turbine-pilot b.tic

# Does not work for Andrew
swift-t hello.swift
