# Steps:
ssh username@perlmutter.nersc.gov（username+SSL+OTP)  
module load pytorch/1.13.1
pip3 install torchinfo
pip install -e .
salloc --nodes 1 --ntasks-per-node=4 --ntasks=4 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4431
python -m torch.distributed.launch --nproc_per_node=4 examples/torch_cifar10_resnet.py --base-lr 0.1 --epochs 10 --model resnet32 --lr-decay 35 75 90



## Some Error Occurred
sbatch: error: Job request does not match any supported policy.
sbatch: error: Batch job submission failed: Unspecified error