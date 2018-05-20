#!/bin/bash
#COBALT -n 512 
#COBALT -q default
#COBALT -A CSC249ADOA01 
#COBALT -t 03:00:00

# For debugging:
# -n 8
# -q debug-cache-quad
# -t 01:00:00
# aprun -N 1 -n 8 ...

set -x

CONDA_ENV=py3.6_tf1.4

source activate $CONDA_ENV
module load darshan

export KMP_BLOCKTIME=0
export KMP_SETTINGS=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=128
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=128

cache=$( hostname )
cache=$COBALT_JOBID"_"$cache

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

aprun -N 1 -n 512 -cc none -b python $DIR/uno_baseline_keras2_hvd.py --epochs 5 --cache 217024_thetamom3_cache -v -l log.$COBALT_JOBID --use_landmark_genes

source deactivate