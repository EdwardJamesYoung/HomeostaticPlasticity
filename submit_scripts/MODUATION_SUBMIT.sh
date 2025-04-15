#!/bin/bash
#!
#! SLURM job script for Ampere GPU Nodes
#! Last updated: 15/04/2025
#!
#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#! Can also be performed using #SBATCH -J 
#SBATCH --job-name=homeostastic_plasticity
#!
#! Which project should be charged (Yashar's project is TERHANI-SL3-GPU):
#! Can also be performed using #SBATCH -A
#SBATCH --account=TERHANI-SL3-GPU
#!
#! Which partition should be used:
#! Can also be performed using #SBATCH -p 
#SBATCH --partition ampere
#!
#! How many whole nodes should be allocated? A node is a single CPU with 76 cores
#! Can also be performed using #SBATCH -N
#SBATCH --nodes=1 
#! How many GPUs should be allocated per node? 
#SBATCH --gres=gpu:1
#!
#! How many (MPI) tasks will there be in total? (<= nodes*76) 
#! An MPI task is a single process (i.e. a single instance of the application)
#! The Ice Lake (icelake) nodes have 76 CPUs each
#! Can also be performed using #SBATCH -n
#SBATCH --ntasks=1
# How many tasks per node:
#SBATCH --ntasks-per-node=1
#!
#! How much wallclock time will be required? (format: hh:mm:ss)
#! Can also be performed using #SBATCH -t
#SBATCH --time=3:00:00 
#!
#! What types of email messages do you wish to receive?
#! Recieve mail when the job ends or fails:
#! Can also be performed using #SBATCH -M
#SBATCH --mail-type=END,FAIL 
#!
#! Create an array of jobs (e.g. for parameter sweeps):
#! Can also be performed using #SBATCH -a
#SBATCH --array=1-6
#! The value of the SLURM_ARRAY_TASK_ID variable will be used to index the array
#!
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#!
#! sbatch directives end here (put any additional directives above this line)
#! ############################################################
#! Notes:
#! Charging is determined by GPU number*walltime.
#!
#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#!
#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:
#!
#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load miniconda/3                    # python/3.10 cuda/11.7
source ~/.bashrc  # Required for conda things
conda deactivate
conda activate comp_neuro
#!
#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory in which sbatch is run.
#!
#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1
#!
#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]
#!
CONFIG_NAME="modulation_experiment_config_${SLURM_ARRAY_TASK_ID}.yaml"
PATH_TO_CONFIG="local/splintered_configs/${CONFIG_NAME}"
#!
#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="python sweep.py -c $PATH_TO_CONFIG"
#!
#!
###############################################################
### You should not have to change anything below this line ####
###############################################################
#!
cd $workdir
echo -e "Changed directory to `pwd`.\n"
#!
JOBID=$SLURM_JOB_ID
#!
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
#!
if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi
#!
echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
#!
echo -e "\nExecuting command:\n==================\n$CMD\n"
#!
eval $CMD  # This tells shell to run the command CMD


