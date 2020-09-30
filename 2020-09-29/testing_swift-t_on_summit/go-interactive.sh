#!/bin/bash

bsub -W 1:00 -nnodes 1 -P med106 -Is /bin/bash -l
