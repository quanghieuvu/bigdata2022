## Installation
Setup and activate anaconda environment:
```bash
conda env update --file environment.yml --prune
conda activate dragonenv
```

## Source-code structure
+ archs: architecture designs
+ evals: most of evaluation metrics
+ loaders: data preprocessing
+ models: implementation of model functions like training, testing, visualization
+ runs: calling functions written in archs, loaders, models
+ utils: supportive functions

## Concepts
+ task_name: includes three tasks S1, S2, S3.
+ arch_id: an architecture design can see various versions with minor changes. Each design is labelled with an "arch_id".
+ model_id: for each design, a model can be trained with different hyper-parameters such as learning rate, weights of loss components, leading to slightly different results. Each model is encoded by a "model_id".

## Basic operations
The following commands should be operated under the <b>src</b> directory.

Generate training and validation files:
```bash
python -W ignore main.py helper generate_train_val 0 0 "task_name"
```
Train the a dragon model (format and example):
```bash
python -W ignore main.py dragon train "arch_id" "model_id" "task_name"
python -W ignore main.py dragon train 0 0 S2
```
Save decoded maps (format and example):
```bash
python -W ignore main.py dragon save_decoded_map "arch_id" "model_id" "task_name"
python -W ignore main.py dragon save_decoded_map 0 0 S2
```


## Related work
1. Arnold's cat map ([url](https://en.wikipedia.org/wiki/Arnold's_cat_map))
1. Chaos-based image encryption algorithm ([url](https://www.sciencedirect.com/science/article/pii/S0375960105011904?via%3Dihub))
1. Dynamics analysis of chaotic maps: From perspective on parameter estimation by meta-heuristic algorithm ([url](https://iopscience.iop.org/article/10.1088/1674-1056/ab695c))
1. A Deep Learning Based Attack for The Chaos-based Image Encryption ([url](https://arxiv.org/pdf/1907.12245v1.pdf))
1. Image similarity using Triplet Loss ([url](https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973))
1. Triplet Loss ([url](https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905))

## Others
+ The trained weights can be downloaded from [here](https://drive.google.com/drive/folders/1eONm0PQjFpRDiNRditIWv45L5BxTR2KO?usp=sharing) and should be put under the <b>ckpt</b> directory.