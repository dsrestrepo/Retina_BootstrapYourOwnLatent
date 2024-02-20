# Bootstrap Your Own Latent Retina Model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


This github repository contains code and data to pre-train and fine tune an embedding generation models for retinal images embeddings. The code is based on the [BYOL](https://arxiv.org/pdf/2006.07733.pdf) paper and implementing the model in pytorch.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Usage](#usage)
- [Modeling](#modeling)
- [Citation](#citation)
- [License](#license)

## Introduction
In this case we'll pre train a ConvNextV2 model using the BYOL method and then fine tune it using the BRSET dataset. The BRSET dataset is a publicly available dataset of retinal images with multiple labels. The dataset is available on PhysioNet and can be accessed through the following DOI link:

## Dataset Description
The BRSET dataset is publicly available on PhysioNet, and you can access it through the following DOI link:

- **PhysioNet:** [A Brazilian Multilabel Ophthalmological Dataset (BRSET)](https://doi.org/10.13026/xcxw-8198)

Please refer to the PhysioNet page for detailed information on the dataset structure, contents, and citation guidelines.

## Usage
Firts, you need to clone this repository to your local machine and install the required libraries. Then, you can explore the dataset and pretrain your own BYOL model using the BRSET dataset:

1. Clone this repository to your local machine:
```
git clone https://github.com/dsrestrepo/Retina_BootstrapYourOwnLatent.git
```

2. Set up your Python environment and install the required libraries by running:

The Python version used here is `Python 3.8.17`
```
pip install -r requirements.txt
```

3. Explore the dataset and access the data for your analysis.

## Data Analysis
The data analysis for the BRSET can be found in the `eda.ipynb` notebook. It includes exploratory data analysis, plots, distributions, and an overview of the dataset. Feel free to use this notebook as a starting point for your own analysis.

# Bootstrap Your Own Latent
To pretrain your own BYOL model using the BRSET dataset, you can use the `train_byol.ipynb` notebook. This notebook demonstrates how to load the dataset, preprocess the images, and train a BYOL model using the BRSET dataset. You can use this notebook as a starting point to pretrain your own BYOL model and then fine-tune it using the BRSET dataset.

## Fine-tuning the model
To fine-tune the pre-trained BYOL model using the BRSET dataset, you can use any of the notebook examples of each one of the downstream tasks. The notebooks are those ended with `..._byol.ipynb` notebooks. These notebooks demonstrate how to load the pre-trained BYOL model, fine-tune it using the BRSET dataset, and evaluate the model's performance on the downstream task.

## Citation
If you use the BRSET dataset in your research, please cite the following publication:

TODO: Add citation

For more information about the BRSET dataset, please refer to the [PhysioNet link](https://physionet.org/content/brazilian-ophthalmological/1.0.0/).