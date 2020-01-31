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

def evaluate_model(log_dir, class_folder, sim_folder = None):

    if sim_folder is None:

        config = Config(LoggerUtils.read_config_file('experiments/'+class_folder+'/config.log'))
        model  = get_model(config)
        model.load_state_dict(torch.load('experiments/'+class_folder+'/best_val_model.pth'))
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        data = LoadData('tiny-imagenet-200/val', from_file = 'trained_labels.txt')

        accuracy, loss = evaluate_non_similarity(config, model, data)

        message = 'Evaluated {} test examples: accuracy={}, loss={}'.format(len(data), accuracy, loss)
        LoggerUtils.write_message(message, os.path.join(log_dir, 'test_eval.log'))


    else:

        config     = Config(LoggerUtils.read_config_file('experiments_similarity/'+sim_folder+'/similarity_config.log'))
        old_config = Config(LoggerUtils.read_config_file('experiments/'+class_folder+'/config.log'))

        
        classification_model = load_model(class_folder)
        model = get_model(config, old_config)
        model.load_state_dict(torch.load('experiments_similarity/'+sim_folder+'/best_val_model.pth'))
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        # load test data

        data = LoadData('tiny-imagenet-200/val',
            pairs = True, ex_per_pair = 50, 
        )

        accuracy, loss, cases_dict = evaluate_similarity(config, old_config, class_folder,
            log_dir, model, classification_model, data,
        )

        message = 'Evaluated {} test examples: accuracy={}, loss={}'.format(len(data), accuracy, 
            loss,
        )

        LoggerUtils.write_message(message, os.path.join(log_dir, 'test_eval.log'))
        message = '\nAccuracy of Case1={}, Case2={}, Case3={}, Case4={}, Case5={}'.format(cases_dict[1],
            cases_dict[2], cases_dict[3], cases_dict[4], cases_dict[5])

        LoggerUtils.write_message(message, os.path.join(log_dir, 'test_eval.log'))



def evaluate_non_similarity(config, model, data):

    criterion  = nn.CrossEntropyLoss()
    batch_size = 128

    test_batch = Batch(data, batch_size)

    train_transform, valid_transform = get_augmentation(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    running_loss = 0.0
    correct = 0
    num_examples = 0

    all_batches = int(np.ceil(len(data)/batch_size))

    for i, curr_batch in enumerate(test_batch, 1):

        inputs, labels = curr_batch.augment(valid_transform)

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        predictions   = torch.max(F.softmax(outputs, dim=1), 1)[1]

        correct += (predictions == labels).float().sum()
        running_loss += loss
        num_examples += inputs.shape[0]

        message = '\rBatch: {}/{}, train_acc = {:.5f}, loss = {:.5f}'.format(i,all_batches, 
            correct/num_examples, running_loss/i)
        stdout.write(message)
        stdout.flush()

    return correct/num_examples, running_loss/i

def get_which_test_case(class_pairs, training_labels):

    cases = np.zeros(len(class_pairs))
    for idx, pair in enumerate(class_pairs):
        if pair[0] == pair[1] and pair[0] in training_labels:
            cases[idx] = 1
            continue
        if pair[0] != pair[1] and pair[0] in training_labels and pair[1] in training_labels:
            cases[idx] = 2
            continue
        if pair[0] != pair[1] and pair[0] not in training_labels and pair[1] in training_labels:
            cases[idx] = 3
            continue
        if pair[0] != pair[1] and pair[0] in training_labels and pair[1] not in training_labels:
            cases[idx] = 3
            continue
        if pair[0] == pair[1] and pair[0] not in training_labels and pair[1] not in training_labels:
            cases[idx] = 4
            continue
        if pair[0] != pair[1] and pair[0] not in training_labels and pair[1] not in training_labels:
            cases[idx] = 5
            continue

    return cases

def get_case_idx(cases, which_case):
    return np.where(cases == which_case)

def evaluate_similarity(config, old_config, class_folder, log_dir, model, classification_model, data):

    criterion  = nn.CrossEntropyLoss()
    batch_size = 512

    test_batch = Batch(data, batch_size)

    train_transform, valid_transform = get_augmentation(old_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    classification_model.to(device)

    embeddings_dict = {}
    device_cpu = 'cpu'
    print("Calculating embeddings for the entire dataset...")
    
    if os.path.isfile('experiments/'+class_folder+'/val_embeddings_idx.json') is True:

        json_embedding_dict = LoggerUtils.read_config_file('experiments/'+class_folder+'/val_embeddings_idx.json')
        embeddings_mat = np.load('experiments/'+class_folder+'/val_embeddings.npy')
        embeddings_dict = 'I exist'

    else:

        json_embedding_dict = {}
        if old_config.model.name == "ResNet50":
            embeddings_mat = np.zeros((100000, 2048))
        elif old_config.model.name == "newVGG16":
            embeddings_mat = np.zeros((100000, 4096))
            
        device_cpu = torch.device('cpu')
        print("Calculating embeddings for the entire dataset...")
        all_data = LoadData('tiny-imagenet-200/val')

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
        with open('experiments/'+class_folder+'/val_embeddings_idx.json', 'w+') as f:

            f.write(json_dict)
            
        np.save('experiments/'+class_folder+'/val_embeddings.npy', embeddings_mat)

        embeddings_dict = 'I exist'

    embeddings_dict = {}
    for link_pair in data.xdata:
        embeddings_dict[link_pair[0]] = torch.from_numpy(embeddings_mat[json_embedding_dict[link_pair[0]]]).float()
        embeddings_dict[link_pair[1]] = torch.from_numpy(embeddings_mat[json_embedding_dict[link_pair[1]]]).float()


    with open('trained_labels.txt', 'r') as f:
        training_labels = []
        for line in f:
            training_labels.append(line.rstrip())

    running_loss = 0.0
    correct = 0
    num_examples = 0

    all_batches = int(np.ceil(len(data)/batch_size))

    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    correct_5 = 0

    message = 'label,class_pair,case,prob'
    LoggerUtils.write_message(message, os.path.join(log_dir, 'test_eval_predictions.log'))

    val_loss = 0.0
    val_correct = 0
    val_acc = 0.0
    num_inputs = 0

    for i, curr_v_batch in enumerate(test_batch, 1):

        val_inputs, val_labels = curr_v_batch.augment(valid_transform, classification_model, embeddings_dict)

        if old_config.model.name == "ResNet50":
            val_inputs = val_inputs.view(torch.Size([val_inputs.shape[0], 2048]))

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

        curr_class_pairs = data.class_pairs[((i-1)*batch_size):((i-1)*batch_size + batch_size)]
        curr_cases = get_which_test_case(curr_class_pairs, training_labels)

        correct_1_r = (predictions[get_case_idx(curr_cases, 1)] == val_labels[get_case_idx(curr_cases, 1)]).float().sum()
        correct_2_r = (predictions[get_case_idx(curr_cases, 2)] == val_labels[get_case_idx(curr_cases, 2)]).float().sum()
        correct_3_r = (predictions[get_case_idx(curr_cases, 3)] == val_labels[get_case_idx(curr_cases, 3)]).float().sum()
        correct_4_r = (predictions[get_case_idx(curr_cases, 4)] == val_labels[get_case_idx(curr_cases, 4)]).float().sum()
        correct_5_r = (predictions[get_case_idx(curr_cases, 5)] == val_labels[get_case_idx(curr_cases, 5)]).float().sum()

        correct_1 += correct_1_r/len(get_case_idx(curr_cases, 1)[0])
        correct_2 += correct_2_r/len(get_case_idx(curr_cases, 2)[0])
        correct_3 += correct_3_r/len(get_case_idx(curr_cases, 3)[0])
        correct_4 += correct_4_r/len(get_case_idx(curr_cases, 4)[0])
        correct_5 += correct_5_r/len(get_case_idx(curr_cases, 5)[0])

        
        message = '\rBatch: {}/{}, train_acc = {:.5f}, loss = {:.5f}'.format(i,all_batches, 
            val_correct/num_inputs, val_loss/i)
        stdout.write(message)
        stdout.flush()
                

        for idx, pair in enumerate(curr_class_pairs):
            message = '{},{},{},{}'.format(val_labels[idx], str(pair), curr_cases[idx], 
                F.softmax(val_outputs, dim=1)[idx,1])
            LoggerUtils.write_message(message, os.path.join(log_dir, 'test_eval_predictions.log'))
        

    cases_dict = {1:correct_1/i, 2:correct_2/i, 3:correct_3/i, 4:correct_4/i, 5:correct_5/i}
    
    return val_correct/num_inputs, running_loss/i, cases_dict

if __name__ == '__main__':


    # ======== PARSE ARGUMENTS =======

    parser = argparse.ArgumentParser(description='Train Tiny-ImageNet Models')

    parser.add_argument('-c','--clas', help='Folder with classification model')
    parser.add_argument('-s','--simi', help='Folder with similrity model')

    args = vars(parser.parse_args())

    if args['simi'] is None:
        log_dir = './experiments/'+args['clas']
    else:
        log_dir = './experiments_similarity/'+args['simi']

    assert os.path.isdir(log_dir) is True, "Directory {} doesnt exists".format(args['fold'])



    evaluate_model(log_dir, args['clas'], args['simi'])






