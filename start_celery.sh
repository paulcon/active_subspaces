#! /bin/bash
# Note: to run on the a cluster with slurm, run:
# > sbatch start_celery.sh

##SBATCH -o outz-%j
##SBATCH -e errz-%j
#SBATCH --job-name="celery"
#SBATCH --uid=hokanson
#SBATCH -p pconstan
#SBATCH --nodes=1	## The max is 2 for Paul's cluster
#SBATCH --ntasks-per-node=1 	## the max is 16 for Paul's cluster
#SBATCH --exclusive
#SBATCH --share

# Note, you will need to run this script in the active_subspaces root 
# directory so that it can pull from the right celeryconfig
cd $HOME/active_subspaces
export TMPDIR=$SCRATCH
srun celery -A active_subspaces.celery worker
