## 8 x Tesla V100(32GB)
<br/>

## download data to input
input/SnakeCLEF2022-large_size<br/>
input/SnakeCLEF2022-test_images<br/>
input/SnakeCLEF2022-TestMetadata.csv<br/>
...

<br/>

### requirements
torch==1.9.1+cu111<br/>
torchvision==0.10.1+cu111
numpy==1.22.2<br/>
pandas==1.4.1<br/>
opencv-python==4.5.5<br/>
albumentations==1.1.0<br/>
timm==0.5.4<br/>
torch-scatter==2.0.9<br/>
tqdm==4.63.0<br/>
...
If there is a library that is not in the list, you just need to pip install it.

<br/>



## eca_nfnet_l2
### Train
#### step1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --backbone 'eca_nfnet_l2' --max_epoch 25 --max_lr 1e-3 --image_size 576
#### step2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint_path 'step_1_checkpoint' --backbone 'eca_nfnet_l2' --use_meta --fix_backbone --max_epoch 10 --max_lr 1e-4 --image_size 576
### Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 submit.py --checkpoint_path 'step_2_checkpoint' --backbone 'eca_nfnet_l2' --use_meta --image_size 736


step1 (~ 13 hours)
step2 (~ 2 hours)
test (~ 0.5 hours)

<br/>
submit_file: submits/submit_eca_nfnet_l2.csv



<br/>

## eca_nfnet_l1
### Train
#### step1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --backbone 'eca_nfnet_l1' --max_epoch 25 --max_lr 1e-3 --image_size 640
#### step2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint_path 'step_1_checkpoint' --backbone 'eca_nfnet_l1' --use_meta --fix_backbone --max_epoch 10 --max_lr 1e-4 --image_size 640

### Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 submit.py --checkpoint_path 'step_2_checkpoint' --backbone 'eca_nfnet_l1' --use_meta --image_size 736

<br/>
submit_file: submits/submit_eca_nfnet_l1.csv



<br/>

## tf_efficientnet_b7_ns
### Train
#### step1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --backbone 'tf_efficientnet_b7_ns' --max_epoch 25 --max_lr 1e-3 --image_size 736 --samples_per_gpu 12
#### step2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint_path 'step_1_checkpoint' --backbone 'tf_efficientnet_b7_ns' --use_meta --fix_backbone --max_epoch 10 --max_lr 1e-4 --image_size 736

### Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 submit.py --checkpoint_path 'step_2_checkpoint' --backbone 'tf_efficientnet_b7_ns' --use_meta --image_size 736


<br/><br/>

### ensemble
Modify the backbones and weights to be merged in the file ensemble_submit.py<br/>
python ensemble_submit.py



## final submit

| backbone | input size |public score | private score |
| :---:      | :---:        |     :---:     | :---:           |
|eca-nfnet-l2 + TTA|train: 576 test: 736| 0.904 | 0.870 |
|eca-nfnet-l1 + TTA|train: 640 test: 736| 0.898 | 0.871 |
|tf-efficientnet-b7-ns|736|0.887|0.863|
eca-nfnet-l1+TTA+eca-nfnet-l2+TTA|test: 736| 0.913 | 0.887|
eca-nfnet-l2+TTA+tf-efficientnet-b7-ns+TTA|test: 736| 0.910 | 0.891|

