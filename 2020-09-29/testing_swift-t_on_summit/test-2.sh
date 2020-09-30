#!/bin/bash

WORKFLOWS_ROOT=
PROCS=
source "/gpfs/alpine/world-shared/med106/weismana/sw/candle/initial/Supervisor/workflows/common/sh/env-summit.sh"

which swift-t
# which turbine
# which stc

# echo "------------------------"
# cat a.tic
# echo "------------------------"
# turbine -n 5 a.tic

# echo "------------------------"
# cat b.swift
# echo "------------------------"
# stc b.swift

# echo "------------------------"
# cat b.tic
# echo "------------------------"
# turbine -n 5 b.tic

ROOT=/gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0
SWIFT=$ROOT/swift-t/2020-09-02

PATH=$SWIFT/stc/bin:$PATH
PATH=$SWIFT/turbine/bin:$PATH

export PROJECT=MED106 # CSC249ADCD01
# export QUEUE=debug
export WALLTIME=00:05:00
export PROCS=4
export PPN=4

swift-t -m lsf hello.swift
