# Solution to my Swift/T on Summit problem

My issue was that running Swift/T on Summit the same way as I ran it on Biowulf did not work, dying with errors like:

```
can't find package turbine 1.2.3
    while executing
"package require turbine 1.2.3"
    (file "./swift-t-b.eZd.tic" line 70)
```

The kicker that Justin pointed out was that I was not actually running it the same way because whereas on Biowulf I had set $TURBINE_LAUNCHER (I believe during Swift/T compilation), I did not set $TURBINE_LAUNCHER on Summit, i.e.,

```bash
export TURBINE_LAUNCHER=jsrun
```

In addition, I needed to set

```bash
export LD_LIBRARY_PATH=/sw/summit/gcc/6.4.0/lib64:$LD_LIBRARY_PATH
```

Making those two changes, and calling Swift/T like `swift-t -n 5 b.swift`, allowed me to get Swift/T working in interactive mode on Summit. I.e., *actually* getting everything same on Summit as on Biowulf ($TURBINE_LAUNCHER, `swift-t -n ...`, interactive mode) gets Swift/T working.

## Working example on Summit

```
weismana@login1:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ bsub -W 01:00 -nnodes 2 -P med106 -q debug -Is /bin/bash
Job <379638> is submitted to queue <debug>.
<<Waiting for dispatch ...>>
<<Starting on batch2>>
(base) bash-4.2$ . ~/.bash_profile
weismana@batch2:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ export LD_LIBRARY_PATH=/sw/summit/gcc/6.4.0/lib64:$LD_LIBRARY_PATH
weismana@batch2:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ export TURBINE_LAUNCHER=jsrun
weismana@batch2:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ /gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/turbine/bin/turbine -n 5 b.tic
HELLO
weismana@batch2:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ /gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/stc/bin/swift-t -n 5 b.swift
HELLO
weismana@batch2:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ unset TURBINE_LAUNCHER
weismana@batch2:~/checkouts/andrew-weisman/public_notebook/2020-09-29/testing_swift-t_on_summit $ /gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/stc/bin/swift-t -n 5 b.swift
Warning: location of mpiexec differs from OPAL_PREFIX / MPI_ROOT.
mpiexec at /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.3.1.2-20200121-awz2q5brde7wgdqqw4ugalrkukeub4eb/bin
OPAL_PREFIX=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2
MPI_ROOT=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2
can't find package turbine 1.2.3
    while executing
"package require turbine 1.2.3"
    (file "./swift-t-b.eZd.tic" line 70)
can't find package turbine 1.2.3
    while executing
"package require turbine 1.2.3"
    (file "./swift-t-b.eZd.tic" line 70)
can't find package turbine 1.2.3
    while executing
"package require turbine 1.2.3"
    (file "./swift-t-b.eZd.tic" line 70)
can't find package turbine 1.2.3
    while executing
"package require turbine 1.2.3"
    (file "./swift-t-b.eZd.tic" line 70)
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[41364,1],4]
  Exit code:    1
--------------------------------------------------------------------------
```
