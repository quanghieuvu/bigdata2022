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
Train the a buffalo model (format and example):
```bash
python -W ignore main.py buffalo train "arch_id" "model_id" "task_name"
python -W ignore main.py buffalo train 0 0 S2
```
Evaluate the discrimination performance on the validation set:
```bash
python -W ignore main.py buffalo eval_discrimination "arch_id" "model_id" "task_name"
python -W ignore main.py buffalo eval_discrimination 0 0 S2
```



## Related work
1. Arnold's cat map ([url](https://en.wikipedia.org/wiki/Arnold's_cat_map))
1. Chaos-based image encryption algorithm ([url](https://www.sciencedirect.com/science/article/pii/S0375960105011904?via%3Dihub))
1. Dynamics analysis of chaotic maps: From perspective on parameter estimation by meta-heuristic algorithm ([url](https://iopscience.iop.org/article/10.1088/1674-1056/ab695c))
1. A Deep Learning Based Attack for The Chaos-based Image Encryption ([url](https://arxiv.org/pdf/1907.12245v1.pdf))
1. Image similarity using Triplet Loss ([url](https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973))
1. Triplet Loss ([url](https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905))
1. Fully Convolutional Networks for Semantic Segmentation ([url](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf))
1. Feature Pyramid Networks for Object Detection ([url](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf))
1. GAN ([url](https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09))
1. Auto-encoder ([url](https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726))
1. SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification ([url](https://arxiv.org/pdf/1707.02131.pdf))
1. Membership Inference Attacks Against Machine Learning Models ([url](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf))

## Others
+ The trained weights can be downloaded from [here](https://drive.google.com/drive/folders/1eONm0PQjFpRDiNRditIWv45L5BxTR2KO?usp=sharing) and should be put under the <b>ckpt</b> directory.