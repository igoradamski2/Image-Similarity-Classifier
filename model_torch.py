import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models 


class L3BNConvNet(nn.Module):
    
    def __init__(self, num_classes, initialization = 'xavier_uniform'):
        super(L3BNConvNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(64, 64, kernel_size=7, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes))

        self.initialization = initialization

    def forward(self, x):
        self.apply_init_weights()
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights


class newVGG16(nn.Module):

    def __init__(self, num_classes, initialization = 'xavier_uniform'):
        super(newVGG16, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=True)

        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class myVGG16(nn.Module):

    def __init__(self, num_classes, initialization = 'xavier_uniform'):
        super(myVGG16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=False).features

        self.features = nn.Sequential(*list(self.features.children())[:-7])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(nn.Linear(25088, 2048*2),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(2048*2, 2048*2),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(2048*2, num_classes))

        self.initialization = initialization

    def forward(self, x):
        self.apply_init_weights()
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    #m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    #m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights


class myResNet18(nn.Module):

    def __init__(self, num_classes, initialization = 'xavier_uniform', pretrained = False):
        super(myResNet18, self).__init__()

        #self.features = nn.Sequential(*list(torchvision.models.resnet18(pretrained=False).children())[:-1])

        self.features = torchvision.models.resnet18(pretrained=pretrained)
        self.features.fc = nn.Linear(512, num_classes)

        #self.classifier = nn.Sequential(nn.Linear(512, num_classes))

        self.initialization = initialization

    def forward(self, x):
        self.apply_init_weights()
        x = self.features(x)
        #x = self.classifier(x)
        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)
        #self.classifier.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    #m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    #m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights 


class myResNet50(nn.Module):

    def __init__(self, num_classes, initialization = 'xavier_uniform', pretrained = False):
        super(myResNet50, self).__init__()

        #self.features = nn.Sequential(*list(torchvision.models.resnet18(pretrained=False).children())[:-1])

        self.features = torchvision.models.resnet50(pretrained=pretrained)
        self.features.fc = nn.Linear(2048, num_classes)

        #self.classifier = nn.Sequential(nn.Linear(512, num_classes))

        self.initialization = initialization

    def forward(self, x):
        self.apply_init_weights()
        x = self.features(x)
        #x = self.classifier(x)
        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)
        #self.classifier.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    #m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    #m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights 

class simpleSimilarity2L(nn.Module):

    def __init__(self, input_size, hidden_size, initialization = 'xavier_uniform'):
        super(simpleSimilarity2L, self).__init__()
        self.features = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(True),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(True),
            nn.Linear(hidden_size[1], 2))

        self.initialization = initialization

    def forward(self, x):
        self.apply_init_weights()
        x = self.features(x)
        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights

class simpleSimilarity1L(nn.Module):

    def __init__(self, input_size, hidden_size, initialization = 'xavier_uniform'):
        super(simpleSimilarity1L, self).__init__()
        #self.features = nn.Sequential(nn.Linear(input_size, hidden_size),
        #    nn.ReLU(True),
        #    nn.Linear(hidden_size, 2))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.initialization = initialization

    def forward(self, x):
        #self.apply_init_weights()
        #x = self.features(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights

class simpleSimilarity0L(nn.Module):

    def __init__(self, input_size, initialization = 'xavier_uniform'):
        super(simpleSimilarity0L, self).__init__()
        self.features = nn.Sequential(nn.Linear(input_size, 2))
        self.initialization = initialization

    def forward(self, x):
        self.apply_init_weights()
        x = self.features(x)
        return x

    def apply_init_weights(self):
        init_weights = self.get_init_function()
        self.features.apply(init_weights)

    def get_init_function(self):
        if self.initialization == 'xavier_uniform':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
        elif self.initialization == 'xavier_normal':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)
        else:
            def init_weights(m):
                pass

        return init_weights




