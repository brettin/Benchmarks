OBALT -n 64 
#COBALT -q default
#COBALT -A Candle_ECP
#COBALT -t 3:00:00

set -x

CONDA_ENV=tf-1.3.0_eigen_hv-0.9.6
CONDA_ENV=py3.6_tf1.4

source activate $CONDA_ENV
export LD_LIBRARY_PATH=/soft/datascience/conda/envs/py3.6_tf1.4/lib/:$LD_LIBRARY_PATH
module load darshan

export KMP_BLOCKTIME=0
export KMP_SETTINGS=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=128
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=128



DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

aprun -N 1 -cc none -b python $DIR/uno_baseline_keras2.py --cache cache.t

source deactivate
