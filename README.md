CrossMol: Cross-Modal Mask-Predict Pre-training For 3D Molecular Data
===================================================================
This is an official implement for CrossMol.

**Abstract**:
Self-supervised pre-training models for molecular data are attracting increasing attention and have demonstrated notable results across many downstream tasks. Moreover, the inherent multimodal properties of molecules have led to extensive efforts to capture the multimodal information contained in molecular data. However, current multimodal molecular pre-training models usually treat inputs from different modalities as equal and independent. Yet, the knowledge and information contained in different modalities can differ significantly. For example, 3D molecular structure data generally contain more and finer-grained knowledge than SMILES data and the SMILES data primarily contains higher-level semantic information, such as the molecular topology. In light of this, we designed a cross-modal mask-predict pre-training model, CrossMol, to better capture the semantic associations between modalities with unequal information volumes. In this approach, the model completes missing 3D structure using higher-level semantic information from different modality, such as SMILES. In this way, the model can learn cross-modal associations and further assist the model in better understanding finer-grained 3D structural information from multiple perspectives. Additionally, to improve the modeling ability of short-range structural information within 3D data, we introduce a new reweighted distance prediction loss function for structural pre-training. Our experiments show that the CrossMol achieves a large margin performance gain on multiple downstream molecular tasks, achieving state-of-the-art results.

Pre-trained Data 
------------------------------

We adopt the same training data from [Uni-Mol](https://openreview.net/forum?id=6K2RM6wVqKu).


| Data                     | File Size  | Update Date | Download Link                                                                                                             | 
|--------------------------|------------| ----------- |---------------------------------------------------------------------------------------------------------------------------|
| molecular pretrain       | 114.76GB   | Jun 10 2022 |https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz                                |

Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2022.9.3, install via `pip install rdkit-pypi==2022.9.3`

To use GPUs within docker you need to [install nvidia-docker-2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) first. Use the following command to pull the docker image:

```bash
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
```

Pre-training Command 
------------------------------

```shell
data_path="/path/to/your/data"  # replace to your data path
save_dir=/path/to/save/data # replace to your save path
logfile=${save_dir}/train.log
n_gpu=4
MASTER_PORT=52088
lr=1e-4
wd=1e-4
batch_size=32
update_freq=1
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

mkdir -p ${save_dir}
cp $0 ${save_dir}

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 32 --ddp-backend=c10d \
       --task unimol_smi2struct_origin_data_2d --loss smi2struct --arch smi2struct_2d_base  \
       --decoder-no-pe \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --validate-interval-updates 10000 \
       --save-interval-updates 10000 --keep-interval-updates 100 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --decoder-x-norm-loss $x_norm_loss --decoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --smi-mask-prob 0 --masked-smi-loss -1 --decoder-origin-pe --decoder-masked-3d-pe \
       --tmp-save-dir ${save_dir}/tmp \
       --max-source-positions 1024 \
       --atom-dict-name dict.txt --smi-dict-name smi_dict.txt \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --encoder-layers 6 --encoder-ffn-embed-dim 1024 --encoder-attention-heads 4 \
       --decoder-layers 15 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 64 \
       --save-dir $save_dir  --only-polar $only_polar > ${logfile} 2>&1
```

