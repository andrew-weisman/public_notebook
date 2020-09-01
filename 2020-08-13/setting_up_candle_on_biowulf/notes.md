# Just temporary and for reference to help debugging: Selected notes on attempts to get CANDLE working on Biowulf

## Summary of recommended ways to go for interactive and batch jobs - 9/1/20

### (1) Interactive building and testing of an MPI program on Biowulf

```
sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k80:1,lscratch:400 --mem=20G --no-gres-shell
. ~/.bash_profile
module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
srun --mpi=pmix --ntasks=3 --cpus-per-task=16 --mem=0 ./a.out # both "--mpi=pmix" and "--mem=0" are key here
```

### (2) Batch building and testing of an MPI program on Biowulf (by running e.g. `sbatch hello_world.sh`)

```bash
#!/bin/bash
#SBATCH -n 3
#SBATCH -N 3
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:k80:1,lscratch:400
#SBATCH --mem=20G
#SBATCH --partition=gpu
module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
srun --mpi=pmix --ntasks=3 --cpus-per-task=16 ./a.out
```

## Summary of following emails from staff on 8/28/20

The K20x GPUs are our oldest ones and a bit odd from a configuration perspective. I'd recommend compiling/testing on a K80 or P100 as well, just to make sure that things work right. The K80 would be best for portability across biowulf (executablers will generally run on a system newer than the one they were compiled on but not necessarily on an older system).

Remember, you can use your user dashboard (https://hpc.nih.gov/dashboard) to monitor your jobs and make sure that they are using resources as you expect.

In terms of your question - the --mpi=pmix flag does a bit of magic to talk to the resource manager (slurm) to launch the MPI runtime with the requested resources, so your synopsis is pretty much correct, at least as far as I understand the process.

Otherwise, what you have looks like the way I would do it.

## Email to staff on 8/27/20

I’m wondering if you wouldn’t mind taking a quick look at the following two blocks of code (sample output is at the top section of this webpage titled “Summary on 8/27/20 of what we found works, particularly for testing MPI on GPU nodes interactively”) and confirming that the way I’m both compiling and running an MPI hello world program in both interactive and batch modes is 100% the way you would recommend building and running MPI programs (using one GPU per task) on Biowulf?

\# (1) Interactive building and testing of an MPI program on Biowulf

```
sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
. ~/.bash_profile
module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
srun --mpi=pmix --ntasks=3 --cpus-per-task=16 --mem=0 ./a.out # both "--mpi=pmix" and "--mem=0" are key here
```

\# (2) Batch building and testing of an MPI program on Biowulf (by running e.g. sbatch hello_world.sh):

```bash
#!/bin/bash
#SBATCH -n 3
#SBATCH -N 3
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:k20x:1,lscratch:400
#SBATCH --mem=60G
#SBATCH --partition=gpu
module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
srun --mpi=pmix --ntasks=3 --cpus-per-task=16 ./a.out
```

Please don’t leave out feelings like “Why is Andrew doing that?” or “I would probably do it differently.” I’m trying to learn here and to test and compile programs required for CANDLE as consistently as possible between interactive and batch mode usage of SLURM.

Perhaps you could also please confirm this: Either in interactive or batch SLURM modes, setting --mpi=pmix in the call to srun somehow automatically invokes the currently loaded MPI installation loaded with e.g. “module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0”. Meaning, srun uses the MPI already loaded in the background, which brings meaning to first compiling a program with mpicc and successively running it using srun instead of mpirun. Please let me know if you don’t know what I mean.

Just to be clear, the code blocks I sent you work...I was just making sure that this was the right way to go, particularly the batch part of it, as we didn't previously talk about that part. (I had done some reading on srun and OpenMPI and it made my head spin a bit...) No rush, and thanks very much for your help!

## Summary on 8/27/20 of what we found works, particularly for testing MPI on GPU nodes interactively

```bash
sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
. ~/.bash_profile
module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
srun --mpi=pmix --ntasks=3 --cpus-per-task=16 --mem=0 ./a.out # both "--mpi=pmix" and "--mem=0" are key here
```

Sample output on 8/27/20:

```
weismanal@cn0603:~/notebook/2020-08-27/refactoring_candle/take2 $ srun --mpi=pmix --ntasks=3 --cpus-per-task=16 --mem=0 ./a.out
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           cn0603
  Local device:         mlx4_0
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           cn0604
  Local device:         mlx4_0
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           cn0605
  Local device:         mlx4_0
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
[1598574126.620213] [cn0603:23343:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn0603 (hostname cn0603), processor cn0603, rank 0 / 3 (CUDA_VISIBLE_DEVICES=0)
[1598574126.642159] [cn0604:17521:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn0604 (hostname cn0604), processor cn0604, rank 1 / 3 (CUDA_VISIBLE_DEVICES=0)
[1598574126.642568] [cn0605:36670:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn0605 (hostname cn0605), processor cn0605, rank 2 / 3 (CUDA_VISIBLE_DEVICES=0)
```

Here is the equivalent call in batch mode (e.g., `sbatch hello_world.sh`):

```bash
#!/bin/bash

#SBATCH -n 3
#SBATCH -N 3
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:k20x:1,lscratch:400
#SBATCH --mem=60G
#SBATCH --partition=gpu

module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
srun --mpi=pmix --ntasks=3 --cpus-per-task=16 ./a.out
```

Sample output on 8/27/20:

```
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           cn0603
  Local device:         mlx4_0
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           cn0605
  Local device:         mlx4_0
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           cn0604
  Local device:         mlx4_0
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
[1598574622.207036] [cn0603:24091:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn0603 (hostname cn0603), processor cn0603, rank 0 / 3 (CUDA_VISIBLE_DEVICES=0)
[1598574622.227353] [cn0604:18224:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn0604 (hostname cn0604), processor cn0604, rank 1 / 3 (CUDA_VISIBLE_DEVICES=0)
[1598574622.225891] [cn0605:37375:0]    ucp_context.c:1437 UCX  WARN  UCP version is incompatible, required: 1.8, actual: 1.7 (release 0 /lib64/libucp.so.0)
Hello from slurm topology address cn0605 (hostname cn0605), processor cn0605, rank 2 / 3 (CUDA_VISIBLE_DEVICES=0)
```

## Email from staff on 8/17/20

That's great news - and yes, the flags I gave are my recommended way of running (with the obvious adjustment for changing numbers of cores, threads, etc.).

Slurm does not make this as easy as it ought to be, that's for sure...

## Email to staff on 8/17/20

Success!!  And, it certainly needs the “--mem=0” option; otherwise, it hangs with:

```
weismanal@biowulf:~ $ squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63186810
JOB_ID              STEPID              PARTITION           NAME                STATE               
63186810            63186810.0          interactive         bash                RUNNING             
63186810            63186810.Extern     interactive         extern              RUNNING
```

This could be great Tim, thanks so much. At the moment I don’t remember what launcher we’re using for CANDLE but I’ll try to use srun with these options, which I assume is what you suggest.

## Email from staff on 8/17/20

Hmmm ... I'm not sure why srun is complaining. That version was compiled against PMIx (not PMI2), which is compatible with SLURM. Can you try adding the following flag to the srun command: "--mpi=pmix".

In my experience, we need to be using srun to get this to work...

## Email to staff on 8/17/20

Thanks Tim:

* sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
* . ~/.bash_profile
* module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
* mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
* srun --ntasks=3 --cpus-per-task=16 --mem=0 ~/a.out --> dies (doesn't hang) because "OMPI was not built with SLURM's PMI support"
* mpirun ~/a.out --> hangs
* On Biowulf: squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63186810
* Output:

```
JOB_ID              STEPID              PARTITION           NAME                STATE              
63186810            63186810.0          interactive         bash                RUNNING            
63186810            63186810.Extern     interactive         extern              RUNNING
```

Thanks!

## Following instructions from previous email

```
weismanal@biowulf:~ $ sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
salloc.exe: Pending job allocation 63186810
salloc.exe: job 63186810 queued and waiting for resources
salloc.exe: job 63186810 has been allocated resources
salloc.exe: Granted job allocation 63186810
salloc.exe: Waiting for resource configuration
salloc.exe: Nodes cn[0610,0623-0624] are ready for job
[weismanal@cn0610 ~]$ . ~/.bash_profile
weismanal@cn0610:~ $ mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
bash: mpicc: command not found
weismanal@cn0610:~ $ module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
weismanal@cn0610:~ $ mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn0610:~ $ srun --ntasks=3 --cpus-per-task=16 --mem=0 ~/a.out 
[cn0610:26671] OPAL ERROR: Not initialized in file /usr/local/src/openmpi/openmpi-4.0.4/opal/mca/pmix/ext2x/ext2x_client.c at line 112
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
[cn0610:26671] Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
srun: error: cn0610: task 0: Exited with exit code 1
[cn0624:24721] OPAL ERROR: Not initialized in file /usr/local/src/openmpi/openmpi-4.0.4/opal/mca/pmix/ext2x/ext2x_client.c at line 112
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
[cn0624:24721] Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
srun: error: cn0624: task 2: Exited with exit code 1
[cn0623:16087] OPAL ERROR: Not initialized in file /usr/local/src/openmpi/openmpi-4.0.4/opal/mca/pmix/ext2x/ext2x_client.c at line 112
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
[cn0623:16087] Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
srun: error: cn0623: task 1: Exited with exit code 1
weismanal@cn0610:~ $ mpirun ~/a.out 
^Cweismanal@cn0610:~ ^Crun --ntasks=3 --cpus-per-task=16 --mem=0 ~/a.out 
weismanal@cn0610:~ $ 
weismanal@cn0610:~ $ 
weismanal@cn0610:~ $ mpirun --ntasks=3 --cpus-per-task=16 --mem=0 ~/a.out 
mpirun: Error: unknown option "--ntasks=3"
Type 'mpirun --help' for usage.
weismanal@cn0610:~ $ mpirun -n 3 --cpus-per-task=16 --mem=0 ~/a.out 
mpirun: Error: unknown option "--cpus-per-task=16"
Type 'mpirun --help' for usage.
weismanal@cn0610:~ $ mpirun -n 3 -c 16 --mem=0 ~/a.out 
mpirun: Error: unknown option "--mem=0"
Type 'mpirun --help' for usage.
weismanal@cn0610:~ $ mpirun -n 3 -c 16 -m 0 ~/a.out 
mpirun: Error: unknown option "-m"
Type 'mpirun --help' for usage.
weismanal@cn0610:~ $ mpirun -n 3 -c 16 ~/a.out 
^Cweismanal@cn0610:~ $ man mpirun
weismanal@cn0610:~ $ mpirun -n 3 ~/a.out 
^Cweismanal@cn0610:~ $ mpirun ~/a.out 
^Cweismanal@cn0610:~ $ 
```

On Biowulf:

```
weismanal@biowulf:~ $ squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63186810
JOB_ID              STEPID              PARTITION           NAME                STATE               
63186810            63186810.0          interactive         bash                RUNNING             
63186810            63186810.Extern     interactive         extern              RUNNING
```

Notes:

* `sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell`
* `. ~/.bash_profile`
* `module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0`
* `mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c`
* `srun --ntasks=3 --cpus-per-task=16 --mem=0 ~/a.out` --> dies (doesn't hang) because "OMPI was not built with SLURM's PMI support"
* `mpirun ~/a.out` --> hangs
* On Biowulf: `squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63186810`
* Output:

```
JOB_ID              STEPID              PARTITION           NAME                STATE               
63186810            63186810.0          interactive         bash                RUNNING             
63186810            63186810.Extern     interactive         extern              RUNNING
```

## Email from staff on 8/17/20

I think my instructions got a bit jumbled - let's try this.

First - allocate an interactive node:

```
biowulf$ sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
```

From within that session - try to launch an srun job as follows:

```
cnXXXX$ srun --ntasks=3 --cpus-per-task=16 --mem=0 /path/to/your_mpi_program
```

If the srun command hangs - back on biowulf, run:

```
biowulf$ squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j <your job ID>
```

and send me the output.

## Email to staff on 8/17/20

I believe I tried everything you suggested; here are my notes:

* Original command that used to work (hello world jobs would not hang): sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
* Swapping out sinteractive for salloc.exe throws salloc.exe: unrecognized option '--no-gres-shell'
* Removing --no-gres-shell throws salloc.exe: error: QOSMaxNodePerJobLimit
* Adding --partition=gpu seems to allocate the job (id 63184253) successfully
* Running srun --no-gres-shell --mem=0 --jobid=63184253 --pty /bin/bash throws srun: unrecognized option '--no-gres-shell'
* Removing --no-gres-shell seems to enter the job allocation successfully
* Running my MPI hello world program still hangs; here's exactly what I'm doing once I enter the allocation's shell using srun:
  * module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
  * mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
  * mpirun (or mpiexec or srun) a.out
  * All of these hang, even when adding unset SLURM_MEM_PER_NODE prior to the mpicc command
  * While one of the jobs is hanging (e.g., mpirun a.out), if I run on Biowulf squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63184253 I get:

```
weismanal@biowulf:~ $ squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63184253
JOB_ID              STEPID              PARTITION           NAME                STATE              
63184253            63184253.0          gpu                 bash                RUNNING            
63184253            63184253.Extern     gpu                 extern              RUNNING
```
 
I have left the job allocation (id 63184253) running.

## Attempts on 8/17/20

What I tried:

```
weismanal@biowulf:~ $ salloc.exe -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell
salloc.exe: unrecognized option '--no-gres-shell'
Try "salloc --help" for more information
weismanal@biowulf:~ $ salloc.exe -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G
salloc.exe: error: QOSMaxNodePerJobLimit
salloc.exe: error: Job submit/allocate failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
weismanal@biowulf:~ $ salloc.exe -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --partition=gpu
salloc.exe: Pending job allocation 63184253
salloc.exe: job 63184253 queued and waiting for resources
salloc.exe: job 63184253 has been allocated resources
salloc.exe: Granted job allocation 63184253
salloc.exe: Waiting for resource configuration
salloc.exe: Nodes cn[0610,0623-0624] are ready for job
weismanal@biowulf:~ $ srun --no-gres-shell --mem=0 --jobid=63184253 --pty /bin/bash
srun: unrecognized option '--no-gres-shell'
srun: unrecognized option '--no-gres-shell'
Try "srun --help" for more information
weismanal@biowulf:~ $ srun --mem=0 --jobid=63184253 --pty /bin/bash
weismanal@cn0610:~ $ module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0
[+] Loading gcc  9.2.0  ... 
[+] Loading openmpi 4.0.4/CUDA-10.2  for GCC 9.2.0 
weismanal@cn0610:~ $ mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn0610:~ $ mpirun a.out
^Cweismanal@cn0610:~ $ unset SLURM_MEM_PER_NODE
weismanal@cn0610:~ $ mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c
weismanal@cn0610:~ $ mpirun a.out
^Cweismanal@cn0610:~ $ mpiexec a.out
^Cweismanal@cn0610:~ $ srun a.out
^Csrun: Cancelled pending job step with signal 2
srun: error: Unable to create step for job 63184253: Job/step already completing or completed
weismanal@cn0610:~ $ mpirun a.out
^Cweismanal@cn0610:~ $ 
```

Notes:

* Original command that used to work (hello world jobs would not hang): `sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell`
* Swapping out `sinteractive` for `salloc.exe` throws `salloc.exe: unrecognized option '--no-gres-shell'`
* Removing `--no-gres-shell` throws `salloc.exe: error: QOSMaxNodePerJobLimit`
* Adding `--partition=gpu` seems to allocate the job (id 63184253) successfully
* Running `srun --no-gres-shell --mem=0 --jobid=63184253 --pty /bin/bash` throws `srun: unrecognized option '--no-gres-shell'`
* Removing `--no-gres-shell` seems to enter the job allocation successfully
* Running my MPI hello world program still hangs; here's exactly what I'm doing once I enter the allocation's shell using `srun`:
  * `module load openmpi/4.0.4/cuda-10.2/gcc-9.2.0`
  * `mpicc /data/BIDS-HPC/public/software/distributions/candle/dev_2/wrappers/test_files/hello.c`
  * `mpirun` (or `mpiexec` or `srun`) ` a.out`
  * All of these hang, even when adding `unset SLURM_MEM_PER_NODE` prior to the `mpicc` command
  * While one of the jobs is hanging (e.g., `mpirun a.out`), if I run on Biowulf `squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63184253` I get:

```
weismanal@biowulf:~ $ squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j 63184253
JOB_ID              STEPID              PARTITION           NAME                STATE               
63184253            63184253.0          gpu                 bash                RUNNING             
63184253            63184253.Extern     gpu                 extern              RUNNING
```

## Email to staff on 8/16/20

I appreciate your explanations of what may be going on. I think I get it but experimenting more on Monday will help me to digest what you're saying.

For now, just a few quick notes:

* I will indeed try what you're suggesting ("squeue --steps-- ...") to see if that's indeed what's going on.
* To get interactive mode to work, I just want to confirm that you're suggesting to try “srun -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=0 --no-gres-shell”?
* You asked to give me more details on the hello world example I'm running, and I just want to make sure we're on the same page... everything I've been trying is hello world, I'm not trying to run CANDLE yet.
* I was able to get hello world to work in batch mode... are you saying you want me to elaborate exactly what I'm trying for hello world in interactive mode?
* Definitely will do regarding not using --partition=gpu in interactive mode... that makes sense, that basically interactive jobs have their own partitions.

## Email from staff on 8/15/20

The underlying issue is an extremely subtle point in how Slurm allocates jobs to batch vs. interactive jobs, specifically, what resources are assigned to different "parts" of the job (what slurm calls steps). Please try step 1 again, and while the job is "hung", run:

```
squeue --steps --Format=jobid,stepid,partition,stepname,stepstate -j <your job ID>
```

You will probably see a "bash" step running and your "srun" step waiting for resources. This is because Slurm thinks that all the resources in your job "belong" to your shell, not to the srun command, and therefore it thinks that the srun command does not have sufficient resources to run. Unfortunately, the way slurm does this accounting changes from release to release, sometimes in unexpected ways (and you can probably tell I'm not a big fan of the way Slurm breaks jobs into steps). One other possible way around this that I remember, in addition to --no-gres-shell, is to add "--mem=0" to your *srun* command (not sinteractive), which basically tells Slurm that you don't want any memory for it (but don't worry, as long as you don't exceed the overall job's allocation, this will be fine).

This doesn't happen in batch jobs because there's no "bash" step to eat the resources. I'm not sure why your simple "hello world" test is failing; can you send me exactly what you ran and what didn't work?

One other note - please don't specify a --partition flag to an sinteractive; interactive jobs run in their own partition, which has access to all resources.

## Email to staff on 8/15/20

So I did some experimenting today (my notes are [here](https://github.com/andrew-weisman/public_notebook/blob/master/2020-08-13/setting_up_candle_on_biowulf/notes.md) if you’re interested), and here’s a summary of overall what’s happened:

* I used to be able to run MPI hello world scripts (both basic bash ones and for Swift/T) on an interactive allocation, e.g., “sinteractive -n 3 -N 3 --ntasks-per-core=1 --cpus-per-task=16 --gres=gpu:k20x:1,lscratch:400 --mem=60G --no-gres-shell”. This no longer seems to work, as the jobs seem to hang now.
* The jobs seemed to hang even using a simpler sinteractive command, e.g., “sinteractive -n 2 -N 2 --gres=gpu:p100:1”.
* Then I tried to see if I could get MPI hello world working in batch mode, and I couldn’t! However, upon adding “--partition=gpu”, they seemed to work in batch mode.
* However, going back and running the scripts with the exact same calls in interactive mode (i.e., “sinteractive -n 2 -N 2 --gres=gpu:p100:1 --partition=gpu”), they still wouldn’t work and would hang at the mpirun and mpiexec commands.
* Now that I am able to at least run the testing scripts in batch mode, I am at least more hopeful that CANDLE can work in batch mode, the primary mode of usage.
* However, debugging will be difficult in batch mode (waiting in the queue multiple/many times), so it would help if we could figure out how to run MPI hello world in interactive mode using e.g. “sinteractive -n 2 -N 2 --gres=gpu:p100:1 --partition=gpu”. But again, not as major of an issue now that I have a workaround by running in batch mode.

Long story short, since something (SLURM?) changed on Biowulf in the past six? months or so, I can no longer run interactive tests on multiple nodes using GPUs. This is not the world’s biggest deal as I can work in batch mode for the time being, but if you had any thoughts on the fix, that would help my debugging (and you’d think it should work anyway since it works in batch mode). Unfortunately Tim your “unset SLURM_MEM_PER_NODE” fix doesn’t seem to help.

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
