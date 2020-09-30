# Problem

Swift/T does not seem to work on Summit.

Can show this by running `bsub test.sh`.

Sample output is at Not_Specified.375996.

## Scheduled

From login node:
```
$ ./test-2.sh
```

### Andrew's result

Job fails; in `turbine-output/output.txt`:

```
+ jsrun -n 4 -r 4 -E TCLLIBPATH -E ADLB_PRINT_TIME=1 -E PYTHONPATH -E TURBINE_OUTPUT=/ccs/home/weismana/turbine-output/2020/09/30/00/39/31 -E TURBINE_JOBNAME=SWIFT -E TCLLIBPATH=/gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/turbine/lib -E ADLB_SERVERS=1 -E TURBINE_WORKERS=3 -E TURBINE_STDOUT= -E TURBINE_LOG=0 -E TURBINE_DEBUG=0 -E ADLB_DEBUG=0 -E ADLB_TRACE=0 /gpfs/alpine/world-shared/med106/gounley1/sw/tcl-200327/bin/tclsh8.6 /ccs/home/weismana/turbine-output/2020/09/30/00/39/31/swift-t-hello.7XX.tic
couldn't load file "/gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/turbine/lib/libtclturbine.so": /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/turbine/lib/libtclturbine.so)
    while executing
"load /gpfs/alpine/world-shared/med106/wozniak/sw/gcc-6.4.0/swift-t/2020-09-02/turbine/lib/libtclturbine.so"
    ("package ifneeded turbine 1.2.3" script)
    invoked from within
"package require turbine 1.2.3"
    (file "/ccs/home/weismana/turbine-output/2020/09/30/00/39/31/swift-t-hello.7XX.tic" line 70)

TURBINE-LSF: jsrun returned an error code!


TURBINE: EXIT CODE: 1

TURBINE: MPIEXEC TIME: 1.168
TURBINE: DATE STOP:  2020-09-30 00:40:52
```

## Interactive

From login node:
```
$ ./go-interactive.sh
```

Once compute node session starts:
```
$ ./test-interactive-2.sh
```

### Andrew's result

This works, but it doesn't test the type of call to Swift/T that CANDLE uses, and that is what's not working for me.

I.e., while it tests `stc hello.swift` and `jsrun turbine-pilot hello.tic`, it does not test e.g. `swift-t hello.swift`, which fails with my usual error:

```
Warning: location of mpiexec differs from OPAL_PREFIX / MPI_ROOT.
mpiexec at /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.3.1.2-20200121-awz2q5brde7wgdqqw4ugalrkukeub4eb/bin
OPAL_PREFIX=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2
MPI_ROOT=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2
can't find package turbine 1.2.3
    while executing
"package require turbine 1.2.3"
    (file "./swift-t-hello.wQl.tic" line 70)
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[2782,1],1]
  Exit code:    1
--------------------------------------------------------------------------
```

This is the form of Swift/T (`swift-t -n NTASKS PROG.swift`) that CANDLE uses and is what I need to get working in order to get CANDLE working on Summit.
