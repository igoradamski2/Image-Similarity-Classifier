import json
import numpy as np


class Config(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, Config(b) if isinstance(b, dict) else b)

class LoggerUtils(object):

    def __init__(self):
        pass

    @staticmethod
    def read_config_file(name):

        config_dict = {}
        with open(name) as f:
            data = json.load(f)
            for p in data:
                config_dict[p] = data[p]

        return config_dict

    @staticmethod
    def write_message(message, log_file):

        with open(log_file, 'a+') as f:
            f.write(message + "\n")

class LearningRateDecay(object):

    def __init__(self, optimizer, amount, tolerance, min_epoch, wait):
        self.optimizer = optimizer
        self.amount    = amount
        self.tol       = tolerance
        self.min_epoch = min_epoch
        self.wait      = wait
        self.history   = {}
        self.last_dec  = 0

    def progress(self, prev_val, epoch):
        
        condition = self.condition(prev_val, epoch)
        if condition:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.amount
            
            self.last_dec = self.wait + 1
        
        if self.last_dec > 0:
            self.last_dec -= 1
            
        self.make_history(epoch)

        return self.optimizer, condition
                

    def condition(self, prev_val, epoch):
        
        if len(prev_val) < self.min_epoch:
            return False
        
        cmean0 = np.mean(np.array(prev_val))
        cmean1 = np.mean(np.array(prev_val[:-1]))
        cmean2 = np.mean(np.array(prev_val[:-2]))
        cmean3 = np.mean(np.array(prev_val[:-3]))

        avdiff = ((cmean0-cmean3) + (cmean0-cmean2) + (cmean0-cmean1))/3
   
        #diff1 = prev_val[-1] - prev_val[-2]
        #diff2 = prev_val[-2] - prev_val[-3]
        #if diff1 <= self.tol and diff2 <= self.tol and epoch > self.min_epoch and self.last_dec == 0:
        if avdiff < self.tol and self.last_dec == 0:
            return True
        else: 
            return False
    
    def make_history(self, epoch):
        
        for param_group in self.optimizer.param_groups:
                curr_lr = param_group['lr']
        
        self.history["epoch_{}".format(epoch)] = curr_lr

    def write_history(self, log_file):

        with open(log_file, 'a+') as f:
            for item in self.history:
                f.write(str(item)+'='+str(self.history[item])+"\n")







