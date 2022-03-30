#! /bin/csh
#SBATCH --ntasks=4
#SBATCH --partition=computeq
#SBATCH --job-name=M1_lig_A_loopmodel

module load gcc/8.2.0

/public/apps/rosetta/2017.29.59598/main/source/bin/loopmodel.static.linuxgccrelease @kic_with_frags.flags >loops.log
