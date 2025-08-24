#!/bin/bash
#!
#! SLURM job script for Ampere GPU Nodes
#!
#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH --job-name=__JOB_NAME__
#!
#! Which project should be charged:
#SBATCH --account=TERHANI-SL3-GPU
#!
#! Which partition should be used:
#SBATCH --partition ampere
#!
#! How many whole nodes should be allocated?
#SBATCH --nodes=1 
#! How many GPUs should be allocated per node? 
#SBATCH --gres=gpu:1
#!
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! How many tasks per node:
#SBATCH --ntasks-per-node=1
#!
#! How much wallclock time will be required? (format: hh:mm:ss)
#SBATCH --time=08:00:00 
#!
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END,FAIL 
#!
#! Create an array of jobs (e.g. for parameter sweeps):
#SBATCH --array=__ARRAY_RANGE__
#!
#! Uncomment this to prevent the job from being requeued:
#SBATCH --no-requeue
#!
#! sbatch directives end here
#! ############################################################
#!
#! Number of nodes and tasks per node allocated by SLURM:
numnodes=${SLURM_JOB_NUM_NODES:-1}  # Default to 1 if not set
numtasks=${SLURM_NTASKS:-1}  # Default to 1 if not set
mpi_tasks_per_node=${SLURM_TASKS_PER_NODE:-1}  # Default to 1 if not set
mpi_tasks_per_node=$(echo "$mpi_tasks_per_node" | sed -e 's/^\([0-9][0-9]*\).*$/\1/')
#!
#! ############################################################
#!
#! Optionally modify the environment seen by the application
#! Optionally modify the environment seen by the application
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load miniconda/3                    # python/3.10 cuda/11.7
source ~/.bashrc  # Required for conda things
conda deactivate
conda activate RL
#!
#! Work directory:
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory in which sbatch is run.
#!
#! Are you using OpenMP? If so increase this:
export OMP_NUM_THREADS=1
#!
#! Number of MPI tasks (safely handle empty variables):
np=$(( ${numnodes:-1} * ${mpi_tasks_per_node:-1} ))
#!
# Make sure SLURM_ARRAY_TASK_ID is set, default to 1 if running outside array
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
CONFIG_NAME="__CONFIG_PREFIX__$(printf "%03d" ${TASK_ID}).yaml"

# Set the path to the config file - this will be in the same directory as this script
CONFIG_DIR="__CONFIG_DIR__"
PATH_TO_CONFIG="${CONFIG_DIR}/${CONFIG_NAME}"

CMD="python run_experiments.py $PATH_TO_CONFIG"
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
echo "Config file: $PATH_TO_CONFIG"
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