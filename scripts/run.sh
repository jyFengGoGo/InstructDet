
config="configs/instructdet.yaml"
startidx=$1
stride=$2
export PYTHONPATH=modules/minigpt4:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0
python3 -u main.py \
     -c ${config} \
     --startidx ${startidx} \
     --stride ${stride}
