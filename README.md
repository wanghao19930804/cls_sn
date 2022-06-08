## download data to input


## eca_nfnet_l2
### Train
#### step1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --backbone 'eca_nfnet_l2' --max_epoch 25 --max_lr 1e-3 --image_size 576
#### step2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint_path 'step_1_checkpoint' --backbone 'eca_nfnet_l2' --use_meta --fix_backbone --max_epoch 10 --max_lr 1e-3 --image_size 576
### Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 submit.py --checkpoint_path 'step_2_checkpoint' --backbone 'eca_nfnet_l2' --use_meta --image_size 736



<br/>

## eca_nfnet_l1
### Train
#### step1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --backbone 'eca_nfnet_l1' --max_epoch 25 --max_lr 1e-3 --image_size 576
#### step2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint_path 'step_1_checkpoint' --backbone 'eca_nfnet_l1' --use_meta --fix_backbone --max_epoch 10 --max_lr 1e-3 --image_size 576

### Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 submit.py --checkpoint_path 'step_2_checkpoint' --backbone 'eca_nfnet_l1' --use_meta --image_size 736

<br/>

## tf_efficientnet_b7_ns
### Train
#### step1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --backbone 'tf_efficientnet_b7_ns' --max_epoch 25 --max_lr 1e-3 --image_size 736 --samples_per_gpu 12
#### step2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint_path 'step_1_checkpoint' --backbone 'tf_efficientnet_b7_ns' --use_meta --fix_backbone --max_epoch 10 --max_lr 1e-3 --image_size 736

### Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 submit.py --checkpoint_path 'step_2_checkpoint' --backbone 'tf_efficientnet_b7_ns' --use_meta --image_size 736


<br/><br/>

### ensemble
Modify the backbones and weights to be merged in the file ensemble_submit.py



