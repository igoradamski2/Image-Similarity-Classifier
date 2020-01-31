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
from train_control import *

# Inputs: folder, 

def train_similarity(config, folder, log_dir):

    classification_model = load_model(folder)

    old_config = Config(LoggerUtils.read_config_file('experiments/'+folder+'/config.log'))

    data = LoadData('tiny-imagenet-200/train', old_config.data.num_classes,
        pairs = True, ex_per_pair = 20, from_file = 'trained_labels.txt',
    )

    train, validation = data.splitValidation(config.data.validation_split)

    # Now we need to remove intersection
    train = remove_intersection(train, validation)


    train_transform, valid_transform = get_augmentation(old_config)

    # Define optimizer
    criterion = nn.CrossEntropyLoss()

    # Define new model
    model = get_model(config, old_config)

    optimizer, lr_decay = get_optimizer(model, config)

    # params
    batch_size      = config.training.batch_size
    num_epoch       = config.training.num_epoch
    batch_per_epoch = int(np.ceil(len(train)/batch_size))

    # Train model
    validation_batch = Batch(validation, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() and config.model.process == "GPU" else "cpu")

    message = "============ Using device {} ============".format(device)
    print(message)
    LoggerUtils.write_message(message, os.path.join(log_dir, 'training.log'))

    model.to(device)
    classification_model.to(device)

    validation_acc  = []

    # If not augmenting every image, for efficiency store embeddings in a dict
    if config.data.augment == 0 and config.data.precompute_embeddings == 1:        

        if os.path.isfile('experiments/'+folder+'/train_embeddings_idx.json') is True:

            json_embedding_dict = LoggerUtils.read_config_file('experiments/'+folder+'/train_embeddings_idx.json')
            embeddings_mat = np.load('experiments/'+folder+'/train_embeddings.npy')
            embeddings_dict = 'I exist'

        else:

            json_embedding_dict = {}
            if old_config.model.name == "ResNet50":
                embeddings_mat = np.zeros((100000, 2048))
            elif old_config.model.name == "newVGG16":
                embeddings_mat = np.zeros((100000, 4096))
            
            device_cpu = torch.device('cpu')
            print("Calculating embeddings for the entire dataset...")
            all_data = LoadData('tiny-imagenet-200/train')

            all_data_batch = Batch(all_data, 128)
            all_batches = int(np.ceil(len(all_data)/128))

            del all_data

            for i, all_data_curr_batch in enumerate(all_data_batch, 1):

                inputs, labels = all_data_curr_batch.augment(valid_transform)

                inputs = inputs.to(device)

                out = classification_model(inputs).to(device_cpu)
                if old_config.model.name == "ResNet50":
                    out = out.view(torch.Size([inputs.shape[0], 2048]))
                elif old_config.model.name == "newVGG16":
                    out = out.view(torch.Size([inputs.shape[0], 4096]))

                for j, link in enumerate(all_data_curr_batch.xdata, 0):
                    index = (i-1)*128 + j
                    json_embedding_dict[link] = index
                    embeddings_mat[index,:] = out[j].numpy()  

                del inputs, labels, out
                torch.cuda.empty_cache()

                stdout.write('\rGetting {}/{} of data embeddings'.format(i, all_batches))
                stdout.flush()

            print('Saving embeddings dict to json file')
            json_dict = json.dumps(json_embedding_dict)
            with open('experiments/'+folder+'/train_embeddings_idx.json', 'w+') as f:

                f.write(json_dict)
            
            np.save('experiments/'+folder+'/train_embeddings.npy', embeddings_mat)

            embeddings_dict = 'I exist'

    else:

        embeddings_dict = None


    for epoch in range(num_epoch):  # loop over the dataset multiple times

        # Load data again
        print("Loading new pairs of images...")
        data = LoadData('tiny-imagenet-200/train', old_config.data.num_classes,
            pairs = True, ex_per_pair = 10, all_files_dict = data.all_files_dict,
            from_file = 'trained_labels.txt', 
        )

        data        = remove_intersection(data, validation)
        train_batch = Batch(data, batch_size)

        print("Image pairs in this epoch : {}".format(len(data)))

        batch_this_epoch = int(np.ceil(len(data)/batch_size))
        
        epoch += 1
        running_loss = 0.0
        correct = 0
        num_examples = 0
        auc = 0
        auc_i = 0

        if embeddings_dict is not None:
            embeddings_dict = {}
            for link_pair in data.xdata:
                embeddings_dict[link_pair[0]] = torch.from_numpy(embeddings_mat[json_embedding_dict[link_pair[0]]]).float()
                embeddings_dict[link_pair[1]] = torch.from_numpy(embeddings_mat[json_embedding_dict[link_pair[1]]]).float()
            for link_pair in validation.xdata:
                embeddings_dict[link_pair[0]] = torch.from_numpy(embeddings_mat[json_embedding_dict[link_pair[0]]]).float()
                embeddings_dict[link_pair[1]] = torch.from_numpy(embeddings_mat[json_embedding_dict[link_pair[1]]]).float()

        for i, curr_batch in enumerate(train_batch, 1):

            inputs, labels = curr_batch.augment(valid_transform, classification_model, embeddings_dict)

            if old_config.model.name == "ResNet50":
                inputs = inputs.view(torch.Size([inputs.shape[0], 2048]))
            elif old_config.model.name == "newVGG16":
                inputs = inputs.view(torch.Size([inputs.shape[0], 4096]))

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            while True:
                try:

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    break
                
                except:

                    print("Sleeping and waiting for GPU...")
                    sleep(60)
                    continue

            
            
            with torch.no_grad():
                predictions   = torch.max(F.softmax(outputs, dim=1), 1)[1]
                correct += (predictions == labels).float().sum()
                running_loss += loss
                num_examples += inputs.shape[0]
                acc = correct/num_examples

                try:
                    auc += roc_auc_compute_fn(F.softmax(outputs, dim=1)[:,1].cpu(), labels.cpu())
                    auc_i +=1
                except:
                    pass

                message = '\rEpoch: {}/{}, batch: {}/{}, ROC-AUC = {:.3f}, train_acc = {:.5f}, loss = {:.5f}'.format(epoch, num_epoch, i,
                                                                                        batch_this_epoch, auc/i, acc, running_loss/num_examples)
                stdout.write(message)
                stdout.flush()
                LoggerUtils.write_message(message, os.path.join(log_dir, 'training.log'))
            
            
        train_batch.reset()
        validation_batch.reset()
            
        with torch.no_grad():

            val_loss = 0.0
            val_correct = 0
            val_acc = 0.0
            num_inputs = 0

            for i, curr_v_batch in enumerate(validation_batch, 1):

                val_inputs, val_labels = curr_v_batch.augment(valid_transform, classification_model, embeddings_dict)

                if old_config.model.name == "ResNet50":
                    val_inputs = val_inputs.view(torch.Size([val_inputs.shape[0], 2048]))
                elif old_config.model.name == "newVGG16":
                    val_inputs = val_inputs.view(torch.Size([val_inputs.shape[0], 4096]))

                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                while True:
                    try:

                        val_outputs = model(val_inputs)
                        break

                    except:

                        print("Sleeping and waiting for GPU...")
                        sleep(60)
                        continue

                val_loss    += criterion(val_outputs, val_labels)
                predictions  = torch.max(F.softmax(val_outputs, dim=1), 1)[1]

                val_correct += (predictions == val_labels).float().sum()
                num_inputs  += val_inputs.shape[0]
                

            val_acc = val_correct/num_inputs
            validation_acc.append(val_acc.cpu().numpy())

            message = "\n\tAfter Epoch: {}/{}, validation acc = {:.5f}, validation loss = {:.5f}".format(epoch,num_epoch,
                                                                                                       val_acc, val_loss/num_inputs)
            print(message)
            
            LoggerUtils.write_message(message, os.path.join(log_dir, 'training.log'))

            optimizer, res = lr_decay.progress(validation_acc, epoch)
            message = "\n\tLearning rate decayed: {}".format(res)
            print(message)

            LoggerUtils.write_message(message, os.path.join(log_dir, 'training.log'))

        # Save the best running model (best on validation accuracy)
        if epoch == 1:
            PATH = os.path.join(log_dir, 'best_val_model.pth')
            torch.save(model.state_dict(), PATH)
            saved_accuracy_val = val_acc
            saved_epoch_val    = epoch

        if epoch > 1 and val_acc > saved_accuracy_val:
            PATH = os.path.join(log_dir, 'best_val_model.pth')
            torch.save(model.state_dict(), PATH)
            saved_accuracy_val = val_acc
            saved_epoch_val    = epoch


        # Save the best running model (best on training accuracy)
        if epoch == 1:
            PATH = os.path.join(log_dir, 'best_train_model.pth')
            torch.save(model.state_dict(), PATH)
            saved_accuracy_train = acc
            saved_epoch_train    = epoch

        if epoch > 1 and acc > saved_accuracy_train:
            PATH = os.path.join(log_dir, 'best_train_model.pth')
            torch.save(model.state_dict(), PATH)
            saved_accuracy_train = acc
            saved_epoch_train    = epoch



    if config.optimizer.decay == 1:
        lr_decay.write_history(os.path.join(log_dir, 'lr_decay.log'))

    PATH = os.path.join(log_dir, 'final_model.pth')
    torch.save(model.state_dict(), PATH)
            
    print('Finished Training')

if __name__ == '__main__':


    # ======== PARSE ARGUMENTS =======

    parser = argparse.ArgumentParser(description='Train Tiny-ImageNet Similarity Models')

    parser.add_argument('-d','--dest', help='LOG destination')
    parser.add_argument('-f','--fold', help='folder of classification model')

    args = vars(parser.parse_args())

    log_dir = './experiments_similarity/'+args['dest']

    assert os.path.isdir(log_dir) is not True, "Directory {} already exists".format(args['dest'])
    assert os.path.isdir('./experiments/'+args['fold']) is True, "Folder with classification model doesnt exist"


    os.mkdir(log_dir)
    print("Created directory {}".format(log_dir))

    
    config = Config(LoggerUtils.read_config_file('configs/config_similarity.json'))


    with open("configs/config_similarity.json") as f:
        lines = f.readlines()
        lines = [l for l in lines]
        with open(os.path.join(log_dir, "similarity_config.log"), "w+") as f1:
            f1.writelines(lines)

    with open('./experiments/'+args['fold']+"/config.log") as f:
        lines = f.readlines()
        lines = [l for l in lines]
        with open(os.path.join(log_dir, "classification_config.log"), "w+") as f1:
            f1.writelines(lines)

    with open('./experiments_similarity/'+args['dest']+'/classification_folder.log', 'w+') as f:
        f.writelines(args['fold'])



    train_similarity(config, args['fold'], log_dir)


