#!/bin/bash

 #BSUB -W 00:05
 #BSUB -nnodes 2
 #BSUB -P med106
 #BSUB -q debug

WORKFLOWS_ROOT=
PROCS=
source "/gpfs/alpine/world-shared/med106/weismana/sw/candle/initial/Supervisor/workflows/common/sh/env-summit.sh"
#source "/ccs/home/weismana/checkouts/ECP-CANDLE/Supervisor/workflows/common/sh/env-summit.sh"

which swift-t
which turbine
which stc

echo "------------------------"
cat a.tic
echo "------------------------"
turbine -n 5 a.tic

echo "------------------------"
cat b.swift
echo "------------------------"
stc b.swift

echo "------------------------"
cat b.tic
echo "------------------------"
turbine -n 5 b.tic
