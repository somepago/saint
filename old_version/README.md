This repository is the official PyTorch implementation of SAINT. Find the paper on [arxiv](https://arxiv.org/abs/2106.01342) 

# SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training


![Overview](pipeline.png)



## Requirements

We recommend using `anaconda` or `miniconda` for python. Our code has been tested with `python=3.8` on linux.

To create a new environment with conda

```
conda create -n saint_env python=3.8
conda activate saint_env
```

We recommend installing the latest pytorch, torchvision, einops, pandas, wget, sklearn packages.

You can install them using 

```
conda install pytorch torchvision -c pytorch
conda install -c conda-forge einops 
conda install -c conda-forge pandas 
conda install -c conda-forge python-wget 
conda install -c anaconda scikit-learn 
``` 

Make sure the following requirements are met

* torch>=1.8.1
* torchvision>=0.9.1

### Optional
We used wandb to update our logs. But it is optional.
```
conda install -c conda-forge wandb 
```


## Training & Evaluation

In each of our experiments, we use a single Nvidia GeForce RTX 2080Ti GPU.

First download the processed datasets from [this link](https://drive.google.com/file/d/1mJtWP9mRP0a10d1rT6b3ksYkp4XOpM0r/view?usp=sharing) into the folder `./data`

To train the model(s) in the paper, run this command:

```
python train.py  --dataset <dataset_name> --attentiontype <attention_type> 
```

Pretraining is useful when there are few training data samples. Sample code looks like this
```
python train.py  --dataset <dataset_name> --attentiontype <attention_type> --pretrain --pt_tasks <pretraining_task_touse> --pt_aug <augmentations_on_data_touse> --ssl_avail_y <Number_of_labeled_samples>
```

Train all 16 datasets by running bash files. `train.sh` for supervised learning and `train_pt.sh` for pretraining and semi-supervised learning

```
bash train.sh
bash train_pt.sh
```

### Arguments
* `--dataset` : Dataset name. We support only the 16 datasets discussed in the paper. Supported datasets are `['1995_income','bank_marketing','qsar_bio','online_shoppers','blastchar','htru2','shrutime','spambase','philippine','mnist','arcene','volkert','creditcard','arrhythmia','forest','kdd99']`
* `--embedding_size` : Size of the feature embeddings
* `--transformer_depth` : Depth of the model. Number of stages.
* `--attention_heads` : Number of attention heads in each Attention layer.
* `--cont_embeddings` : Style of embedding continuous data.
* `--attentiontype` : Variant of SAINT. 'col' refers to SAINT-s variant, 'row' is SAINT-i, and 'colrow' refers to SAINT.
* `--pretrain` : To enable pretraining
* `--pt_tasks` : Losses we want to use for pretraining. Multiple arguments can be passed.
* `--pt_aug` : Types of data augmentations used in pretraining. Multiple arguments are allowed. We support only mixup and CutMix right now.
* `--ssl_avail_y` : Number of labeled samples used in semi-supervised experiments. Default is 0, which means all samples are labeled and is supervised case.
* `--pt_projhead_style` : Projection head style used in contrastive pipeline.
* `--nce_temp` : Temperature used in contrastive loss function.
* `--active_log` : To update the logs onto wandb. This is optional

### Evaluation

We choose the best model by evaluating the model on validation dataset. The AUROC(for binary classification datasets) and Accuracy (for multiclass classification datasets) of the best model on test datasets is printed after training is completed. If wandb is enabled, they are logged to 'test_auroc_bestep', 'test_accuracy_bestep'  variables.


## Acknowledgements

We would like to thank the following public repo from which we borrowed various utilites.
- https://github.com/lucidrains/tab-transformer-pytorch

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Cite us

```
@article{somepalli2021saint,
  title={SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training},
  author={Somepalli, Gowthami and Goldblum, Micah and Schwarzschild, Avi and Bruss, C Bayan and Goldstein, Tom},
  journal={arXiv preprint arXiv:2106.01342},
  year={2021}
}

```