# VGAE-L



Based on the code of CIKM 2021 paper "Variational Graph Normalized Auto-Encoders" (CIKM 2021).
> Variational Graph Normalized Auto-Encoders.  
> Seong Jin Ahn, Myoung Ho Kim.  
> CIKM '21: The 30th ACM International Conference on Information and Knowledge Management Proceedings.  
> Short paper link: https://arxiv.org/abs/2108.08046

## Installation Instructions

1. Create a conda environment with python=3.8.0
2. Install PyTorch v1.8.1 \
Windows install command for cuda 11: `conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge`
3. Install PyTorch Sparse (and PyTorch Scatter): \
`conda install pytorch-sparse -c pyg`
4. Install PyTorch Geometric: \
`pip install torch-geometric`

## Easy Run (from the paper authors)
`python main_authors.py --dataset=Cora --training_rate=0.2 --epochs=300`

## Easier Run
`python loop_eval_a1.py`\
Trains the model over multiple random data-splits and calculates average performance. \
Modify values in `loop_eval_a1.py` to evaluate `VGAE`, `GAE`, `GNAE` and `VGNAE` models
