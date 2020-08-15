# Just temporary and for reference to help debugging: Selected notes on attempts to get CANDLE working on Biowulf

## Attempts on 8/15/20

Adding `unset SLURM_MEM_PER_NODE` prior to mpirun/srun in interactive mode doesn't seem to help:

```
weismanal@biowulf:~ $ sinteractive -n 2 -N 2 --gres=gpu:p100:1
salloc.exe: Pending job allocation 63094233
salloc.exe: job 63094233 queued and waiting for resources
salloc.exe: job 63094233 has been allocated resources
salloc.exe: Granted job allocation 63094233
salloc.exe: Waiting for resource configuration
salloc.exe: Nodes cn[2352-2353] are ready for job
```

```
[weismanal@cn2352 ~]$ . ~/.bash_profile
weismanal@cn2352:~ $ module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
weismanal@cn2352:~ $ unset SLURM_MEM_PER_NODE
weismanal@cn2352:~ $ mpicc -o a-on_p100_nodes.out /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn2352:~ $ mpirun a-on_p100_nodes.out
^Cweismanal@cn2352:~ $ mpiexec a-on_p100_nodes.out
^Cweismanal@cn2352:~ $ srun a-on_p100_nodes.out
srun: Job 63094233 step creation temporarily disabled, retrying


^Csrun: Cancelled pending job step with signal 2
srun: error: Unable to create step for job 63094233: Job/step already completing or completed
```

Running `unset SLURM_MEM_PER_NODE` on command line prior to submitting script that failed yesterday:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

Still results in `sbatch.exe: error: Batch job submission failed: Requested node configuration is not available`.

Adding `unset SLURM_MEM_PER_NODE` in script itself:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

Results in same error message `sbatch.exe: error: Batch job submission failed: Requested node configuration is not available`.

Now trying non-multinode partition using `unset SLURM_MEM_PER_NODE` on command line prior to submission:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
 #SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
 #SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

#unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

Get similar error as yesterday I believe (didn't record this yesterday):

```
sbatch.exe: error: QOSMaxNodePerJobLimit
sbatch.exe: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
```

I assume this is because multiple nodes are requested and the `multinode` partition is not selected.

I thought this is what used to work, so I am switching for now to investigate what we used to run. First of all, I see that what used to work now dies with:

```
which: no mpicc in (/usr/local/Java/jdk1.8.0_181/bin:/usr/local/apps/ant/1.10.3/bin:/usr/local/Tcl_Tk/8.6.8/gcc_7.2.0/bin:/usr/local/OpenMPI/3.1.3/CUDA-9.2/gcc-7.3.0-pmi2/bin:/usr/local/CUDA/9.2.148/bin:/usr/local/GCC/7.3.0/bin:/usr/local/Anaconda/envs/py3.6/bin:/data/weismanal/miniconda3/condabin:/usr/local/slurm/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/X11R6/bin:/usr/local/jdk/bin:/usr/bin:/usr/local/mysql/bin:/home/weismanal/bin:/data/BIDS-HPC/public/software/distributions/candle/dev/wrappers/templates/scripts:/usr/local/GSL/gcc-7.2.0/2.4/bin:/usr/local/apps/R/3.5/3.5.0_build2/bin:/data/BIDS-HPC/public/software/distributions/candle/dev/swift-t-install/stc/bin:/data/BIDS-HPC/public/software/distributions/candle/dev/swift-t-install/turbine/bin)
```

This is at least consistent with my current findings below that `mpicc` is no longer found in the desired `OpenMPI` distribution, as Jean suggested should no longer work. (I doubt `mpicc` is actually needed when *running* CANDLE, so I can probably make CANDLE work again by removing this `which mpicc` command wherever it is, but for now that's besides the point.)

But at least this spits out the `sbatch` line that CANDLE used to use that worked,

```bash
 --ntasks=3 --gres=gpu:k80:1 --mem-per-cpu=7G --cpus-per-task=1 --ntasks-per-core=1 --partition=gpu --time=00:20:00 --ntasks-per-node=2 --nodes=2
```

so now I can try this pretty much verbatim on my hello world program; I am running the script:

```bash
#!/bin/bash

# This is an example sbatch command line that used to work:
# --ntasks=3 --gres=gpu:k80:1 --mem-per-cpu=7G --cpus-per-task=1 --ntasks-per-core=1 --partition=gpu --time=00:20:00 --ntasks-per-node=2 --nodes=2

 #SBATCH -n 2
 #SBATCH -N 2
 #SBATCH --cpus-per-task=8
 #SBATCH --gres=gpu:p100:1
 #SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

#unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

using the command line:

```bash
sbatch --ntasks=3 --gres=gpu:k80:1 --mem-per-cpu=7G --cpus-per-task=1 --ntasks-per-core=1 --partition=gpu --time=00:20:00 --ntasks-per-node=2 --nodes=2 /home/weismanal/notebook/2020-08-15/testing_mpicc/testing_mpi.sh
```

For some reason this seems to work (with questionable settings of `CUDA_VISIBLE_DEVICES` however and `srun` does not work); now trying putting those settings into the actual `sbatch` script:

```bash
#!/bin/bash

# This is an example sbatch command line that used to work:
# --ntasks=3 --gres=gpu:k80:1 --mem-per-cpu=7G --cpus-per-task=1 --ntasks-per-core=1 --partition=gpu --time=00:20:00 --ntasks-per-node=2 --nodes=2

#SBATCH --ntasks=3
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2

 #SBATCH -n 2
 #SBATCH -N 2
 #SBATCH --cpus-per-task=8
 #SBATCH --gres=gpu:p100:1
 #SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

#unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

and running using `sbatch testing_mpi.sh`.

This seems to work as well (again with questionable settings of `CUDA_VISIBLE_DEVICES` and `srun` does not work) so this shows I can rely on putting the `sbatch` settings into the submission script directly.

Note that now that I think of how we run CANDLE anyway, attempting to allocate 1 GPU per node, the questionable `CUDA_VISIBLE_DEVICES` settings (multiple ranks on the same node return the same value for that variable, e.g., `0`) shoudln't matter anyway. Note those settings possibly never would have been a problem anyway... the correct, separate GPUs may very well have been used, but I think it doesn't matter now anyway.

Now trying to figure out why these settings seem to work but the others didn't. First trying to add `--partition=gpu` to what didn't work last time, i.e.:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

#unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

This seems to work! Removing the `#SBATCH --partition=gpu` line as a final check of that being the difference-maker:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --gres=gpu:p100:1
 #SBATCH --partition=gpu

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

#unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

Same as before: this dies with

```
sbatch.exe: error: QOSMaxNodePerJobLimit
sbatch.exe: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
```

Now trying Tim's `unset` fix, i.e.:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

unset SLURM_MEM_PER_NODE

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

The `srun` part still doesn't work (everything else does), but that's okay for now, as I can try getting `CANDLE` to run using `mpirun` or `mpiexec` instead of `srun`.

Now trying to do run the above in interactive mode, which was one of my original problems, i.e., testing the `CANDLE` setup in interactive mode by running simple hello world programs in interactive mode, which I believe used to work but no longer seems to work.

Unfortunately, the jobs still hang in interactive mode:

```
weismanal@biowulf:~ $ sinteractive -n 2 -N 2 --gres=gpu:p100:1 --partition=gpu
salloc.exe: Pending job allocation 63099565
salloc.exe: job 63099565 queued and waiting for resources
salloc.exe: job 63099565 has been allocated resources
salloc.exe: Granted job allocation 63099565
salloc.exe: Waiting for resource configuration
salloc.exe: Nodes cn[2352-2353] are ready for job
[weismanal@cn2352 ~]$ . ~/.bash_profile
weismanal@cn2352:~ $ module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
weismanal@cn2352:~ $ mpicc -o a-on_p100_nodes.out /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn2352:~ $ mpirun a-on_p100_nodes.out
^Cweismanal@cn2352:~ $ unset SLURM_MEM_PER_NODE
weismanal@cn2352:~ $ mpirun a-on_p100_nodes.out
^Cweismanal@cn2352:~ $ 
weismanal@cn2352:~ $ mpiexec a-on_p100_nodes.out
^Cweismanal@cn2352:~ ^Cpiexec a-on_p100_nodes.out
weismanal@cn2352:~ $ srun a-on_p100_nodes.out
^Csrun: Cancelled pending job step with signal 2
srun: error: Unable to create step for job 63099565: Job/step already completing or completed
```

Confirming that these exact commands work in batch mode:

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu

. ~/.bash_profile
module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc -o a-on_p100_nodes.out /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
mpirun a-on_p100_nodes.out
# unset SLURM_MEM_PER_NODE
mpiexec a-on_p100_nodes.out
srun a-on_p100_nodes.out
```

Yes they do. So to summarize: I can no longer seem to get the above sort of commands working in interactive mode, where I'm almost positive that they used to (now, they just hang). They work however in batch mode. I got confused yesterday because I couldn't even get batch mode to work, however, I've (re-)discovered that in order to get it to work in batch mode, I need the `#SBATCH --partition=gpu` directive.

So, to proceed I can run my testing in batch mode, though this is much less efficient and will require more waiting, though I may be able to deal with this for the time being. The bottom-line question for Biowulf staff is why I can no longer perform multi-node testing using GPUs in interactive mode. Them fixing this would help but again for the time being I can probably work through this by performing my testing in batch mode.

Note that I ran one last test in batch mode, using the same script as above except moving the `unset` statement to above the `mpicc` line, with no luck; `srun` still doesn't work. Again, that may be okay for now.

## Email from staff on 8/14/20

This is something that changed with a fairly recent slurm upgrade. One thing we've found to get srun working again is to run:

```
unset SLURM_MEM_PER_NODE
```

*before* your mpirun/srun command. I'm not sure if it always works, but it's helped similar situations in the past. Please let me know if this works. If not, I'll have to dig in a bit deeper.

## Email to staff on 8/14/20

Sorry, one more datapoint: I am unable to do this testing in batch mode because the requested configuration is not available, using both GPUs (at least P100s) in the multinode partition.

Again, I’m pretty sure this used to work. CANDLE itself is typically run using many nodes with at least 1 GPU on each node (it typically runs deep learning methods and so needs GPUs). Is this still possible on Biowulf?

## Attempts on 8/14/20

Died with `sbatch.exe: error: Batch job submission failed: Requested node configuration is not available` (GPUs requested):

```bash
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

Succeeded (GPUs NOT requested):

```
#!/bin/bash

#SBATCH -n 2
#SBATCH -N 2
#SBATCH --cpus-per-task=8
 #SBATCH --gres=gpu:p100:1
#SBATCH --partition=multinode

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0

mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c

mpirun a.out
mpiexec a.out
srun a.out
```

Output of the above command is in `/home/weismanal/notebook/2020-08-14/testing_mpicc/slurm-63047052.out`.

## Email to staff on 8/14/20

P.S. I am nearly certain this used to work, at least when using a PMI2 build of OpenMPI.

## Email to staff on 8/14/20

At least when using interactive sessions, I cannot get an MPI program to work on GPU nodes, though it works on CPU nodes. The jobs just hang when I’m on a GPU node. For example:

```
weismanal@biowulf:~ $ sinteractive -n 2 -N 2 --gres=gpu:p100:1
salloc.exe: Pending job allocation 63042074
salloc.exe: job 63042074 queued and waiting for resources
salloc.exe: job 63042074 has been allocated resources
salloc.exe: Granted job allocation 63042074
salloc.exe: Waiting for resource configuration
salloc.exe: Nodes cn[2350-2351] are ready for job
```

```
[weismanal@cn2350 ~]$ . ~/.bash_profile
weismanal@cn2350:~ $ module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
weismanal@cn2350:~ $ mpicc -o a-on_p100_nodes.out /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn2350:~ $ mpirun a-on_p100_nodes.out 
^Cweismanal@cn2350:~ $ mpiexec a-on_p100_nodes.out 
^Cweismanal@cn2350:~ $ srun a-o^C
weismanal@cn2350:~ $ srun a-on_p100_nodes.out 
^Csrun: Cancelled pending job step with signal 2
srun: error: Unable to create step for job 63042074: Job/step already completing or completed
weismanal@cn2350:~ $ exit
exit
srun: error: cn2350: task 0: Exited with exit code 1
salloc.exe: Relinquishing job allocation 63042074
```

```
weismanal@biowulf:~ $ sinteractive -n 2 -N 2
salloc.exe: Pending job allocation 63043504
salloc.exe: job 63043504 queued and waiting for resources
salloc.exe: job 63043504 has been allocated resources
salloc.exe: Granted job allocation 63043504
salloc.exe: Waiting for resource configuration
salloc.exe: Nodes cn[1074-1075] are ready for job
```

```
[weismanal@cn1074 ~]$ . ~/.bash_profile
weismanal@cn1074:~ $ module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
weismanal@cn1074:~ $ mpicc -o a-on_cpu_nodes.out /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn1074:~ $ mpirun a-on_cpu_nodes.out 
--------------------------------------------------------------------------
The library attempted to open the following supporting CUDA libraries,
but each of them failed.  CUDA-aware support is disabled.
libcuda.so.1: cannot open shared object file: No such file or directory
libcuda.dylib: cannot open shared object file: No such file or directory
/usr/lib64/libcuda.so.1: cannot open shared object file: No such file or directory
/usr/lib64/libcuda.dylib: cannot open shared object file: No such file or directory
If you are not interested in CUDA-aware support, then run with
--mca opal_warn_on_missing_libcuda 0 to suppress this message.  If you are interested
in CUDA-aware support, then try setting LD_LIBRARY_PATH to the location
of libcuda.so.1 to get passed this issue.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.
 
  Local host:   cn1074
  Local device: mlx5_0
--------------------------------------------------------------------------
[1597441641.303711] [cn1074:72336:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
[1597441641.339510] [cn1075:58448:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn1074 (hostname cn1074), processor cn1074, rank 0 / 2 (CUDA_VISIBLE_DEVICES=(null))
Hello from slurm topology address cn1075 (hostname cn1075), processor cn1075, rank 1 / 2 (CUDA_VISIBLE_DEVICES=(null))
[cn1074:72325] 1 more process has sent help message help-mpi-common-cuda.txt / dlopen failed
[cn1074:72325] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[cn1074:72325] 1 more process has sent help message help-mpi-btl-openib.txt / error in device init
weismanal@cn1074:~ $ mpiexec a-on_cpu_nodes.out 
--------------------------------------------------------------------------
The library attempted to open the following supporting CUDA libraries,
but each of them failed.  CUDA-aware support is disabled.
libcuda.so.1: cannot open shared object file: No such file or directory
libcuda.dylib: cannot open shared object file: No such file or directory
/usr/lib64/libcuda.so.1: cannot open shared object file: No such file or directory
/usr/lib64/libcuda.dylib: cannot open shared object file: No such file or directory
If you are not interested in CUDA-aware support, then run with
--mca opal_warn_on_missing_libcuda 0 to suppress this message.  If you are interested
in CUDA-aware support, then try setting LD_LIBRARY_PATH to the location
of libcuda.so.1 to get passed this issue.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.
 
  Local host:   cn1075
  Local device: mlx5_0
--------------------------------------------------------------------------
[1597441676.169011] [cn1075:58517:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
[1597441676.170199] [cn1074:72400:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn1074 (hostname cn1074), processor cn1074, rank 0 / 2 (CUDA_VISIBLE_DEVICES=(null))
Hello from slurm topology address cn1075 (hostname cn1075), processor cn1075, rank 1 / 2 (CUDA_VISIBLE_DEVICES=(null))
[cn1074:72389] 1 more process has sent help message help-mpi-common-cuda.txt / dlopen failed
[cn1074:72389] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[cn1074:72389] 1 more process has sent help message help-mpi-btl-openib.txt / error in device init
weismanal@cn1074:~ $ srun a-on_cpu_nodes.out 
[cn1075:58548] OPAL ERROR: Not initialized in file /usr/local/src/openmpi/openmpi-4.0.4/opal/mca/pmix/ext2x/ext2x_client.c at line 112
--------------------------------------------------------------------------
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support and therefore cannot
execute. There are several options for building PMI support under
SLURM, depending upon the SLURM version you are using:
 
  version 16.05 or later: you can use SLURM's PMIx support. This
  requires that you configure and build SLURM --with-pmix.
 
  Versions earlier than 16.05: you must use either SLURM's PMI-1 or
  PMI-2 support. SLURM builds PMI-1 by default, or you can manually
  install PMI-2. You must then build Open MPI using --with-pmi pointing
  to the SLURM PMI library location.
 
Please configure as appropriate and try again.
--------------------------------------------------------------------------
*** An error occurred in MPI_Init
*** on a NULL communicator
*** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
***    and potentially your MPI job)
[cn1075:58548] Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
srun: error: cn1075: task 1: Exited with exit code 1
[cn1074:72433] OPAL ERROR: Not initialized in file /usr/local/src/openmpi/openmpi-4.0.4/opal/mca/pmix/ext2x/ext2x_client.c at line 112
--------------------------------------------------------------------------
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support and therefore cannot
execute. There are several options for building PMI support under
SLURM, depending upon the SLURM version you are using:
 
  version 16.05 or later: you can use SLURM's PMIx support. This
  requires that you configure and build SLURM --with-pmix.
 
  Versions earlier than 16.05: you must use either SLURM's PMI-1 or
  PMI-2 support. SLURM builds PMI-1 by default, or you can manually
  install PMI-2. You must then build Open MPI using --with-pmi pointing
  to the SLURM PMI library location.
 
Please configure as appropriate and try again.
--------------------------------------------------------------------------
*** An error occurred in MPI_Init
*** on a NULL communicator
*** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
***    and potentially your MPI job)
[cn1074:72433] Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
srun: error: cn1074: task 0: Exited with exit code 1
```

I’d really like to be able to test MPI code that needs to use the GPUs in interactive mode. Perhaps they’d work in batch mode? I’m not sure about that but even if that’s the fix, that would make it harder to debug, waiting in the queue each time.

## Email to staff on 8/14/20

Interesting.  Okay thanks Jean, I’ll give it a try.  I’m worried because at least in the past I needed to use srun to use CANDLE, but I can give this a fresh try now and maybe be okay with mpirun or mpiexec.

## Email from staff on 8/14/20

Those modules shouldn’t be there. Users should *not* use pmi2 versions anymore.  
Please run ‘module avail openmpi’ and pick non-pmi2 ones.

## Email to staff on 8/14/20

I see mpicc in that module… the problem is that at least in the past I had needed to use modules like these:

```
openmpi/4.0.1/cuda-10.1/gcc-9.2.0-pmi2
openmpi/3.1.3/cuda-9.2/gcc-7.3.0-pmi2
openmpi/3.1.2/cuda-9.0/gcc-7.3.0-pmi2,
```

none of which seem to have mpicc, though they definitely used to, at least the latter two.  In fact, for the first one (4.0.1), I was in the process of using it, then my allocation froze, and then upon requesting a new allocation I found that mpicc in that module had disappeared!  This led me to think that Biowulf staff may have somehow set up a script that yanked mpicc whenever it found it was being used? It was weird! But then I noticed at [https://hpc.nih.gov/development/MPI.html](https://hpc.nih.gov/development/MPI.html) that perhaps this was being done on purpose? It’s possible I’m misinterpreting all this.

Is there a particular reason mpicc is unavailable in openmpi/4.0.1/cuda-10.1/gcc-9.2.0-pmi2?

## Email from staff on 8/14/20

Loading openmpi module will add the correct paths and variables for you.  
Then you can use mpicc:

```
$ module load openmpi/3.0.4/gcc-7.4.0
[+] Loading gcc  7.4.0  ...
[+] Loading openmpi 3.0.4  for GCC 7.4.0
[maoj@biowulf /usr/local/lmod/modulefiles/openmpi]$ which mpicc
/usr/local/OpenMPI/3.0.4/gcc-7.4.0/bin/mpicc
```

## Email from staff on 8/13/20

We have many versions of openmpi.

```
$ module avail openmpi
```

Pick the one you want and do:

```
$ module load openmpi/3.0.4/gcc-7.4.0
```

## Email to staff on 8/13/20

I am not finding mpicc available on Biowulf… am I missing something? I am nervous because per [https://hpc.nih.gov/development/MPI.html](https://hpc.nih.gov/development/MPI.html) it appears that mpicc has been removed on purpose for OpenMPI. I am pretty confused as that precludes development using OpenMPI, right? Am I misunderstanding something?

Thanks very much in advance! (I am trying to re-build and test CANDLE as it may have stopped working on Biowulf and part of my setup workflow is building and testing a simple MPI hello world program.)
