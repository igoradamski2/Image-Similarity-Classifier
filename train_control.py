import argparse
import json
import logging
from load_process_data import *
from model_torch import *
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models 
import torchvision.transforms as transforms
from sys import stdout
from utilities import *
import os
from time import sleep
from sklearn.metrics import roc_auc_score


def load_data(config):

    data = LoadData('tiny-imagenet-200/train', config.data.num_classes)

    train, validation = data.splitValidation(config.data.validation_split)

    return train, validation

def get_augmentation(config):

    print("Defining augmentation procedure...")

    if config.model.name == "ResNet18" or config.model.name == "ResNet50" or config.model.name == "newVGG16":

        valid_transform = transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.39731373, 0.44799216, 0.48021961), 
                                  (0.28160392, 0.26892941, 0.27651765))
            ]
        )

    else:

        valid_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.39731373, 0.44799216, 0.48021961), 
                                  (0.28160392, 0.26892941, 0.27651765))
            ]
        )

    if config.data.augment == 0:

        train_transform = valid_transform

    elif config.data.augment == 1:

        train_transform = transforms.Compose(
            [transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(30)]),
            transforms.ToTensor(),
            transforms.Normalize((0.39731373, 0.44799216, 0.48021961), 
                                  (0.28160392, 0.26892941, 0.27651765))
            ]
        )

        if config.model.name == "ResNet18" or config.model.name == "ResNet50" or config.model.name == "newVGG16":

            train_transform = transforms.Compose(
            [transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(30)]),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.39731373, 0.44799216, 0.48021961), 
                                  (0.28160392, 0.26892941, 0.27651765))
            ]
        )

    return train_transform, valid_transform


def get_model(config, old_config = None):

    print("Defining model...")

    # Define model
    if config.model.name == 'L3BNConvNet':
        model = L3BNConvNet(config.data.num_classes, config.model.initialization)

    elif config.model.name == "newVGG16":
        model = newVGG16(config.data.num_classes)
    elif config.model.name == 'VGG16':
        model = myVGG16(config.data.num_classes, config.model.initialization)
    elif config.model.name == "ResNet18":
        model = myResNet18(config.data.num_classes, 
            config.model.initialization,
            bool(config.model.pretrained),
        )
    elif config.model.name == "ResNet50":
        model = myResNet50(config.data.num_classes, 
            config.model.initialization,
            bool(config.model.pretrained),
        )
    elif config.model.name == "simpleSimilarity0L":
        if old_config.model.name == "ResNet50":
            model = simpleSimilarity0L(2048,
                initialization = config.model.initialization,
            )
        elif old_config.model.name == "newVGG16":
            model = simpleSimilarity0L(4096,
                initialization = config.model.initialization,
            )
    elif config.model.name == "simpleSimilarity1L":
        if old_config.model.name == "ResNet50":
            model = simpleSimilarity1L(2048,
                hidden_size = config.model.hidden_size,
                initialization = config.model.initialization,
            )
        elif old_config.model.name == "newVGG16":
            model = simpleSimilarity1L(4096,
                hidden_size = config.model.hidden_size,
                initialization = config.model.initialization,
            )
    elif config.model.name == "simpleSimilarity2L":
        if old_config.model.name == "ResNet50":
            model = simpleSimilarity2L(2048,
                hidden_size = config.model.hidden_size,
                initialization = config.model.initialization,
            )
        elif old_config.model.name == "newVGG16":
            model = simpleSimilarity2L(4096,
                hidden_size = config.model.hidden_size,
                initialization = config.model.initialization,
            )



    return model

def get_optimizer(model, config):

    print("Defining optimizer...")

    if config.optimizer.name == "Adam":

        optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

    elif config.optimizer.name == "SGD":
        if config.optimizer.nesterov == 1:

            optimizer = optim.SGD(model.parameters(), 
                lr=config.optimizer.learning_rate, 
                momentum=config.optimizer.momentum,
                nesterov = True,
            )
        else:

            optimizer = optim.SGD(model.parameters(), 
                lr=config.optimizer.learning_rate, 
                momentum=config.optimizer.momentum,
                nesterov = False,
            )

    elif config.optimizer.name == "RMSProp":

        optimizer = optim.RMSprop(model.parameters(), lr=config.optimizer.lr)


    if config.optimizer.decay == 0:

        lr_decay = LearningRateDecay(optimizer, amount = 0, 
            tolerance = 0, 
            min_epoch = config.training.num_epoch + 10, 
            wait = 0,
        )
    
    else:

        lr_decay = LearningRateDecay(optimizer, amount = config.optimizer.decay_by, 
            tolerance = config.optimizer.decay_tolerance, 
            min_epoch = config.optimizer.min_epoch, 
            wait = config.optimizer.wait,
        )

    return optimizer, lr_decay


# ------------

def load_model(folder):

    config = Config(LoggerUtils.read_config_file('experiments/'+folder+'/config.log'))

    if config.model.name == 'L3BNConvNet':

        model = L3BNConvNet(config.data.num_classes, config.model.initialization)
        model.load_state_dict(torch.load('experiments/'+folder+'/best_val_model.pth'))

    elif config.model.name == 'VGG16':

        model = myVGG16(config.data.num_classes, config.model.initialization)
        model.load_state_dict(torch.load('experiments/'+folder+'/best_val_model.pth'))

        model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])

    elif config.model.name == "newVGG16":

        model = newVGG16(config.data.num_classes, config.model.initialization)
        model.load_state_dict(torch.load('experiments/'+folder+'/best_val_model.pth'))

        model.model.classifier = nn.Sequential(*list(model.model.classifier.children())[:-3])

    elif config.model.name == "ResNet18":
        
        model = myResNet18(config.data.num_classes, 
            config.model.initialization,
            False,
        )

    elif config.model.name == "ResNet50":
        
        model = myResNet50(config.data.num_classes, 
            config.model.initialization,
            False,
        )
        model.load_state_dict(torch.load('experiments/'+folder+'/best_val_model.pth'))
        model.features = nn.Sequential(*list(model.features.children())[:-1])

    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model

def remove_intersection(train, validation):

        train_pairs = ['-'.join(link) for link in train.xdata]
        val_pairs   = ['-'.join(link) for link in validation.xdata]

        set_train_pairs = set(train_pairs)
        set_val_pairs   = set(val_pairs)
        
        intersection = set_train_pairs.intersection(set_val_pairs)

        train_pairs = np.array(train_pairs)
        val_pairs   = np.array(val_pairs)

        train_pairs_new = np.setdiff1d(train_pairs, np.array(list(intersection)))

        keep_idx = np.in1d(train_pairs,train_pairs_new)
        keep_idx = np.where(keep_idx == True)

        train.xdata = list(itemgetter(*list(keep_idx[0]))(train.xdata))
        train.ydata = train.ydata[keep_idx[0]]

        return train


def roc_auc_compute_fn(y_preds, y_targets):
    
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()

    return roc_auc_score(y_true, y_pred)






