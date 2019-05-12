#!/usr/bin

python ./BAN/pipeline.py \
--memo exp1 \
--data-dir ./data \
--model resnet \
--depth 32 \
-c 1 \
--born-time 3 \
--nb-classes 100 \
--devices 1 \
--mode common\

python ./BAN/pipeline.py --memo exp-cifar100 --dataset cifar100 --data-dir ./data --weight_decay 5e-4 --model plaincnn --depth 6 -c 1 --born-time 3 --nb-classes 100  --mode common
python ./BAN/pipeline.py --memo exp-cifar10 --dataset cifar10 --data-dir ./data --weight_decay 5e-4 --model plaincnn --depth 6 -c 1 --born-time 3 --nb-classes 10  --mode common

python ./BAN/pipeline.py --memo exp-cifar100 --data-dir ./data --model wrn --depth 28 -c 1 --born-time 3 --nb-classes 100 --devices 1 --mode common --gamma 0.2 --max-epoch 200 --milestones 60,120,160 --weight_decay 5e-4
python ./BAN/pipeline.py --memo exp-cifar10 --data-dir ./data --model wrn --depth 28 -c 1 --born-time 3 --nb-classes 10 --devices 1 --mode common --gamma 0.2 --max-epoch 200 --milestones 60,120,160 --weight_decay 5e-4
python ./BAN/pipeline.py --memo exp-cifar10 --data-dir ./data --model resnet --depth 32 -c 1 --born-time 3 --nb-classes 100 --devices 1 --mode common --weight_decay 1e-4
