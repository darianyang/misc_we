#!/bin/bash
#SBATCH --job-name=bn_bs
#SBATCH --output=job_logs/slurm.out
#SBATCH --error=job_logs/slurm.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cluster=gpu
#SBATCH --partition=a100_multi
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
##SBATCH --mail-user=jml230@pitt.edu
##SBATCH --mail-type=END,FAIL,BEGIN

set -x
cd $SLURM_SUBMIT_DIR
source env.sh || exit 1

env | sort

cd $WEST_SIM_ROOT
SERVER_INFO=$WEST_SIM_ROOT/west_zmq_info-$SLURM_JOBID.json

# if [ ! -f "west_init.h5" ]; then
#     cp west.h5 west_init.h5
# fi

# start server
w_run --work-manager=zmq --n-workers=0 --zmq-mode=master --zmq-write-host-info=$SERVER_INFO --zmq-comm-mode=ipc \
	--zmq-master-heartbeat 3 --zmq-worker-heartbeat 120 --zmq-startup-timeout 3600 --zmq-shutdown-timeout 360 --zmq-timeout-factor 240 &> west-$SLURM_JOBID.log &

# wait on host info file up to one minute
for ((n=0; n<180; n++)); do
    if [ -e $SERVER_INFO ] ; then
        echo "== server info file $SERVER_INFO =="
        cat $SERVER_INFO
        break
    fi
    sleep 1
done

# exit if host info file doesn't appear in one minute
if ! [ -e $SERVER_INFO ] ; then
    echo 'server failed to start'
    exit 1
fi

# start clients, with the proper number of cores on each

scontrol show hostname $SLURM_NODELIST >& SLURM_NODELIST.log

for node in $(scontrol show hostname $SLURM_NODELIST); do
    ssh -o StrictHostKeyChecking=no $node $PWD/node.sh $SLURM_SUBMIT_DIR $SLURM_JOBID $node $CUDA_VISIBLE_DEVICES --work-manager=zmq --n-workers=4 --zmq-mode=client --zmq-read-host-info=$SERVER_INFO --zmq-comm-mode=tcp &
done


wait
