#!/bin/bash -l
#SBATCH --job-name=llr40-c-debug
#SBATCH --partition=debug
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=logs/regrade.%x.%j.out
#
# The C backend on the debug partition (30 min cap): 10 nodes = 40 ranks = one kernel per rank, into
# results_debug/ (a 40-rank split, kept apart from the 16-rank results/). Resubmitted by
# run_debug_chain.sh until every rank has graded its worklist; each job resumes past graded rows.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
export LLR40_BACKEND=c LLR40_RESULTS=$SLURM_SUBMIT_DIR/results_debug
echo "backend=c nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS $(date -Is)"
srun --cpu-bind=verbose,rank_ldom --hint=nomultithread bash "$SLURM_SUBMIT_DIR/regrade_rank.sh"
echo "DONE $(date -Is)"
