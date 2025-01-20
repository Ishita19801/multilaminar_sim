#!/bin/bash

# ----- #
# Job name
#SBATCH --job-name=2src_snr


# ----- #
# Jobs to run; each element corresponds to a subject.
# Only run n at a time with (%n) at the end of the command!
# Counting the python way!
# SBATCH --array=0-9899%200
# SBATCH --array=0-999%200
# SBATCH --array=1000-1999%200
# SBATCH --array=2000-2999%200
# SBATCH --array=3000-3999%200
# SBATCH --array=4000-4999%200
# SBATCH --array=5000-5999%200
# SBATCH --array=6000-6999%200
# SBATCH --array=7000-7999%200
# SBATCH --array=8000-8999%200
#SBATCH --array=9000-9900%200


# ----- #
# Computational resources.
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# Instead of specifying a nodelist which will ask for all the nodes to be
# available for each job, exclude the nodes that are not contained in the
# nodelist. Each job will only occupy one node this way.
# SBATCH --nodelist=node[13-21]
# SBATCH --exclude=node[2-12]


# ----- #
# Task time limit (D-HH:MM:SS)
#SBATCH --time=7-00:00:00


# ----- #
# Output and error filenames.
# Currently skipped and instead used directly when calling the python script.
# --output=fichier_de_sortie${SLURM_ARRAY_TASK_ID}.txt
# --error=sortie_erreur.err


# ----- #
# Python activation.
module add Programming_Languages/anaconda/3.11

# Activation of virtual python environment.
conda activate lameg

#SBATCH --licenses=sps

# ----- #
# Run script.
# Standard output and standard error are NOT redirected to the same file.
python -u /pbs/home/b/bonaiuto/laminar_erf/pipeline_23_2source_snr_simulations.py > /sps/isc/bonaiuto/laminar_erf/output/output_2src_snr_$SLURM_ARRAY_TASK_ID.txt 2> /sps/isc/bonaiuto/laminar_erf/output/error_2src_snr_$SLURM_ARRAY_TASK_ID.txt ${SLURM_ARRAY_TASK_ID}

