#! /bin/csh
#SBATCH --ntasks=4
#SBATCH --partition=computeq
#SBATCH --job-name=dock

/public/apps/moe/moe2020/bin-lnx64/moebatch -exec "run 'run.sh'"