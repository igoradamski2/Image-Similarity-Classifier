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

def train(config, log_dir):

    # Load data
    train, validation = load_data(config)

    # Define augmentation
    train_transform, valid_transform = get_augmentation(config)

    # Define model
    model = get_model(config)

    print(model)

    # Define optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer, lr_decay = get_optimizer(model, config)


    # params
    batch_size      = config.training.batch_size
    num_epoch       = config.training.num_epoch
    batch_per_epoch = int(np.ceil(len(train)/batch_size))

    # Train model
    train_batch      = Batch(train, batch_size)
    validation_batch = Batch(validation, 16)

    device = torch.device("cuda:0" if torch.cuda.is_available() and config.model.process == "GPU" else "cpu")

    message = "============ Using device {} ============".format(device)
    print(message)
    LoggerUtils.write_message(message, os.path.join(log_dir, 'training.log'))

    model.to(device)

    validation_acc  = []

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        
        epoch += 1
        running_loss = 0.0
        correct = 0
        num_examples = 0

        for i, curr_batch in enumerate(train_batch, 1):

            inputs, labels = curr_batch.augment(train_transform)

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
                

                message = '\rEpoch: {}/{},\t batch: {}/{},\t train_acc = {:.5f},\t loss = {:.5f}'.format(epoch, num_epoch, i,
                                                                                        batch_per_epoch, acc, running_loss/num_examples)
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

                val_inputs, val_labels = curr_v_batch.augment(valid_transform)

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

    parser = argparse.ArgumentParser(description='Train Tiny-ImageNet Models')

    parser.add_argument('-d','--dest', help='LOG destination')

    args = vars(parser.parse_args())

    log_dir = './experiments/'+args['dest']

    assert os.path.isdir(log_dir) is not True, "Directory {} already exists".format(args['dest'])


    os.mkdir(log_dir)
    print("Created directory {}".format(log_dir))

    
    config = Config(LoggerUtils.read_config_file('configs/config.json'))


    with open("configs/config.json") as f:
        lines = f.readlines()
        lines = [l for l in lines]
        with open(os.path.join(log_dir, "config.log"), "w+") as f1:
            f1.writelines(lines)


    train(config, log_dir)


