import numpy as np
import pickle
import os
from PIL import Image
import random
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from operator import itemgetter 

class DataObject(object):

    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def __getitem__(self, arg):
        return DataObject(self.xdata[arg], self.ydata[arg])

    def __len__(self):
        return len(self.xdata)

    def augment(self, func = None, model = None, emb_dict = None):

        if emb_dict is not None:

            right_links = [self.xdata[i][0] for i in range(len(self))]
            left_links  = [self.xdata[i][1] for i in range(len(self))]

            inputs_right = torch.stack([emb_dict[link] for link in right_links])
            inputs_left  = torch.stack([emb_dict[link] for link in left_links])

            inputs = torch.abs(inputs_right-inputs_left)

        else:

            if isinstance(self.xdata[0], str):

                inputs  = torch.stack([func(Image.open(self.xdata[i]).convert('RGB')) for i in range(len(self))])
                targets = self.ydata

            else:

                inputs_left  = torch.stack([func(Image.open(self.xdata[i][0]).convert('RGB')) for i in range(len(self))])
                inputs_right = torch.stack([func(Image.open(self.xdata[i][1]).convert('RGB')) for i in range(len(self))])

                inputs  = [DataObject.extract_features(inputs_left, model), 
                           DataObject.extract_features(inputs_right, model),
                ]

                inputs = torch.abs(inputs[0]-inputs[1])
        
        targets = self.ydata

        return inputs, targets

    def splitValidation(self, v_size):

        size       = len(self.xdata)
        random.seed(2888)
        idx        = set(random.sample(range(size), int(round(v_size*size))))
        not_idx    = set(range(size)) - idx

        train      = DataObject(list(itemgetter(*not_idx)(self.xdata)),
                                self.ydata[np.array(list(not_idx))],
        )
        
        validation = DataObject(list(itemgetter(*idx)(self.xdata)), 
                                self.ydata[np.array(list(idx))],
        )
        
        return train, validation

    @staticmethod
    def extract_features(inputs, model):

        out = model(inputs)
        return out

class Batch(object):

    def __init__(self, data_object, batch_size):

        assert isinstance(data_object, DataObject), "Pass an instance of DataObject to Batch"
        assert batch_size < len(data_object), "Batch size must be smaller than data length"
        
        self.all_data     = data_object
        self.batch_number = 0
        self.batch_size   = batch_size
        self.data         = data_object[0:batch_size]

        assert len(self.data) == self.batch_size, "Length of batch object is not batch_size"
        
    def __next__(self):
        
        self.data = self.all_data[(self.batch_number*self.batch_size):(self.batch_number*self.batch_size + self.batch_size)]
        self.batch_number += 1

        if len(self.data) == 0:
            raise StopIteration
        
        return self.data

    def __iter__(self):
        return self

    def __len__(self):
        return np.ceil(len(self.all_data)/self.batch_size)
    
    def reset(self):
        self.batch_number = 0
        self.data = self.all_data[0:self.batch_size]



class LoadData(DataObject):
    
    def __init__(self, folder, 
        num_classes = 300, pairs = False,
        ex_per_pair = 10, all_files_dict = None, from_file = None,
    ):
        
        self.folder  = folder
        self.pairs = pairs
        self.ex_per_pair = ex_per_pair
        self.from_file = from_file

        if all_files_dict is None:
            all_files_dict = {}

        if self.pairs == False:
            print("Loading just images...")
            images, labels = LoadData.loadImages(self, num_classes)
        else:
            print("Loading pairs of images...")
            images, labels = LoadData.loadImagePairs(self, all_files_dict)
        random.seed(2888)
        shfl = np.random.permutation(len(images))
        images = list(itemgetter(*shfl)(images))
        if self.pairs == False:
            pass
        else:
            self.class_pairs = np.array(list(itemgetter(*shfl)(self.class_pairs)))
        #images = [images[i] for i in shfl]
        labels = labels[shfl]
        super(LoadData, self).__init__(images, labels)

    def loadImagePairs(self, all_files_dict):
        images      = []
        labels      = []
        class_pairs = []
        print("Loading data...")

        if self.from_file is not None:

            with open(self.from_file, 'r') as f:
                loading_labels = []
                for line in f:
                    loading_labels.append(line.rstrip())

        else:

            loading_labels = []
            for d in os.listdir(self.folder):
                if os.path.isdir(os.path.join(self.folder, d)):
                    loading_labels.append(d)


        #all_files_dict = {}
        for im_class1 in tqdm(loading_labels):

            for im_class2 in loading_labels:

                if im_class1 == im_class2:
                    path = os.path.join(self.folder, im_class1, "images")
                    try:
                        all_files = all_files_dict[im_class1]
                    except:
                        all_files = []
                        path = os.path.join(self.folder, im_class1, "images")
                        for file in os.listdir(path):
                            if file.endswith(".JPEG"):
                                all_files.append(file)
                        all_files_dict[im_class1] = all_files

                    #np.random.seed(2888)
                    indices1 = np.random.randint(0,len(all_files)-1, (len(loading_labels)-1)*self.ex_per_pair)
                    indices2 = np.random.randint(0,len(all_files)-1, (len(loading_labels)-1)*self.ex_per_pair)

                    for i, idx in enumerate(indices1):
                        images.append([os.path.join(path, all_files[idx]), os.path.join(path, all_files[indices2[i]])])
                        labels.append(1)
                        class_pairs.append((im_class1, im_class2))

                else:

                    path1 = os.path.join(self.folder, im_class1, "images")
                    path2 = os.path.join(self.folder, im_class2, "images")

                    try:
                        all_files1 = all_files_dict[im_class1]
                    except:
                        all_files1 = [] 
                        for file in os.listdir(path1):
                            if file.endswith(".JPEG"):
                                all_files1.append(file)
                        all_files_dict[im_class1] = all_files1
                    
                    try:
                        all_files2 = all_files_dict[im_class2]
                    except:
                        all_files2 = []
                        for file in os.listdir(path2):
                            if file.endswith(".JPEG"):
                                all_files2.append(file)
                        all_files_dict[im_class2] = all_files2

                    #np.random.seed(2888)
                    indices1 = np.random.randint(0,len(all_files1)-1, self.ex_per_pair)
                    indices2 = np.random.randint(0,len(all_files2)-1, self.ex_per_pair)

                    for i, idx in enumerate(indices1):
                        images.append([os.path.join(path1, all_files1[idx]), os.path.join(path2, all_files2[indices2[i]])])
                        labels.append(0)
                        class_pairs.append((im_class1, im_class2))

        self.all_files_dict = all_files_dict
        self.class_pairs = class_pairs

        labels = torch.from_numpy(np.array(labels))

        return images, labels


    def loadImages(self, num_classes):
        images = []
        labels = []
        idx    = 0
        print("Loading data...")

        if self.from_file is not None:

            with open(self.from_file, 'r') as f:
                loading_labels = []
                for line in f:
                    loading_labels.append(line.rstrip())

            for d in tqdm(loading_labels):
                if os.path.isdir(os.path.join(self.folder, d)):
                    path = os.path.join(self.folder, d, "images")
                    for file in os.listdir(path):
                        images.append(os.path.join(path, file))
                        labels.append(d)
        
        else:

            for d in tqdm(os.listdir(self.folder)):
                if os.path.isdir(os.path.join(self.folder, d)):
                    path = os.path.join(self.folder, d, "images")
                    for file in os.listdir(path):
                        #img = Image.open(os.path.join(path, file)).convert('RGB')
                        #img = np.array(img)
                        #img = np.moveaxis(img, -1, 0)
                        images.append(os.path.join(path, file))
                        labels.append(d)
                    idx += 1
                if idx == num_classes:
                    break

        #images = np.array(images)
        labels = np.array(labels)

        # Check how many distinct categories we have
        real_num_classes = np.shape(np.unique(labels))[0]

        print("Loaded {} categories, should load {}".format(real_num_classes, num_classes))

        # Represent labels as integers
        labels, lab_dict = LoadData.create_one_hot_rep(labels)
        self.label_dictionary = lab_dict

        #images = images.astype('uint8')
        #labels = LoadData.create_one_hot_vecs(labels)
        #images = torch.from_numpy(images).float()
        labels = torch.from_numpy(labels)

        #images = images.float()
        return images, labels

    @staticmethod
    def create_one_hot_rep(labels):
        unique_labels = np.unique(labels)
        lab_dict = {label:idx for idx, label in enumerate(unique_labels)}
        new_labels = [lab_dict[label] for label in labels]

        return np.array(new_labels), lab_dict

    @staticmethod
    def create_one_hot_vecs(labels):
        unique_labels = np.unique(labels)
        new_labels    = np.zeros((len(labels), len(unique_labels)))
        for idx, label in enumerate(labels):
            new_labels[idx, label] = 1

        return new_labels


