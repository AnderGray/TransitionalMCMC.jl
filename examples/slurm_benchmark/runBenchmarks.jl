###
#   Script for launching many Himmelblaus
###

numProcs = [1, 5, 10, 20, 50, 80, 100, 120, 150, 180]

script = ""
for num in numProcs

    script = 
"#!/bin/bash -l
#SBATCH -p phi
#SBATCH -e errors_$num
#SBATCH -o output_$num
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -n $num
#SBATCH -t 24:00:00

module purge
#module load neutronics
#module load libs/qt/5.9.1/gcc-5.5.0
module load python/3.6
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

julia tmcmcHimmelblau_par_Slurm.jl $num
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
