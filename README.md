# Image-Similarity-Classifier
Repository for training and evaluating an image similarity classifier on the Tiny ImageNet dataset. 

## Data

This repository is designed to work with images from the Tiny ImageNet dataset, containing 64x64x3 images,  however the code is able to process images of any size. If you want to train the classifier on your own images you must either change the folder name from which the images are loaded in [load_process_data.py](load_process_data.py). The data structure needs to be:
```
data -> {label1_folder, label2_folder, ..., labeln_folder} -> \*.JPEG
```
So that the data folder contains folders with images from different classes.

To download the Tiny ImageNet dataset use:
```{bash}
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```
Since the validation (val) folder in the tiny-imagenet-200 has a different structure, in order for the code to work properly we need to restructure it to the structure given above:
```{bash}
chmod u+x bash/restructure.sh
./bash/restructure.sh
```

## Training

### Classification

Training is performed in two steps. First we train the classification network and that is controled by the contents of the [config.json](config.json) file. In the model section write the name of the model as one of the defined models in [model_torch.py](model_torch.py). Some of the implemented models are:
1. [VGG16](https://arxiv.org/pdf/1409.1556.pdf)
2. [ResNet50](https://arxiv.org/pdf/1512.03385.pdf)
3. Simple 3 Convolutional layer network with batch-norm (L3BNConvNet)

In the [config.json](config.json) you can control learning rate decay, batch size, number of epochs, the optimizer, how many classes to train on and many more training hyperparameters. To train the classification network type:
```{bash}
python train.py -d experiment_name
```
and experiment_name will be the directory in the experiments/ folder which the training logs and the final models will be saved.

### Similarity

The training of the similarity network is controlled by [config_similarity.json](config_similarity.json). Many of the parameters there will be unused as we only train a binary classifier now, and the code automatically loads the old config file used to train the classification network. The models for the similarity network consist only of fully-connected layers and can be viewed in [model_torch.py](model_torch.py). Some of them are:
1. simpleSimilarity0L - no hidden layers network.
2. simpleSimilarity1L - 1 hidden layer network.
3. simpleSimilarity2L - 2 hidden layers network.

Training is performed by:
```{bash}
python train_similarity.py -d experiment_destination -f folder_with_classification_model
```
and the model and training logs will be saved in experiments_similarity/experiment_destination and the model will be trained using the embeddings created by the best model sitting in experiments/folder_with_classification_model.

## Evaluation

To evaluate your model on the test data (which must sit in tiny-imagenet-200/val) type:
```{bash}
python evaluate_model.py -c folder_with_classification_model -s folder_with_similarity_model
```
and not supplying the -s parameter will evaluate the classification model only.

More details of the actual implementation can be extracted from the paper-like summary (mini_project2.pdf)[mini_project2.pdf]. 





