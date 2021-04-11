###
#   Script for launching many Himmelblaus
###

numDims = 1:30

script = ""
for num in numDims

    script = 
"#!/bin/bash -l
#SBATCH -p phi
#SBATCH -e errors_$num
#SBATCH -o output_$num
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -n 10
#SBATCH -t 24:00:00

module purge
module load julia

#
# Should not need to edit below this line
#
echo ========================================================= echo SLURM job: submitted date = `date`
date_start=`date +%s`
echo ========================================================= echo Job output begins
echo -----------------
echo
hostname

julia Nd_gaussians.jl $num
echo
echo ---------------
echo Job output ends
echo =========================================================
echo SLURM job: finished date = `date`
    "

    #println(script)
    open("runSim_$num.sh","w") do io
        print(io, script)
    end
    run(`sbatch runSim_$num.sh`);
end
