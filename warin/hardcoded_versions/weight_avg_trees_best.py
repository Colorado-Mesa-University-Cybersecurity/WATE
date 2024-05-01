import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
    
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        hidden_size = input_size*100
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
def weight_avg_2(epochs, train_loader:DataLoader, val_loader:Dataset, input_size, output_size, device_in_use, model:str = 'Linear'):
    epochs = epochs//3

    if model == 'Linear':
        model1 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion1 = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model2 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion2 = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model3 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion3 = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer3 = optim.SGD(model3.parameters(), lr=0.01)  # Stochastic Gradient Descent
    else:
        model1 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion1 = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer1 = optim.SGD(model1.parameters(), lr=0.00001)  # Stochastic Gradient Descent

        model2 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion2 = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer2 = optim.SGD(model2.parameters(), lr=0.00001)  # Stochastic Gradient Descent

        model3 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion3 = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer3 = optim.SGD(model3.parameters(), lr=0.00001)  # Stochastic Gradient Descent

    test_loss_dic = {}
    train_loss_dic = {}

    for _ in range(epochs):
        model1.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer1.zero_grad()  
            outputs = model1(inputs)  
            loss = criterion1(outputs, targets.unsqueeze(2))  
            loss.backward() 
            optimizer1.step() 
        train_loss_dic[f'm1_{_}'] = loss.item()

        model1.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model1(inputs)
                loss = criterion1(outputs, targets.unsqueeze(2))
        test_loss_dic[f'm1_{_}'] = loss.item()


    for _ in range(epochs):
        model2.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer2.zero_grad()  
            outputs = model2(inputs)  
            loss = criterion2(outputs, targets.unsqueeze(2))  
            loss.backward() 
            optimizer2.step()
        train_loss_dic[f'm2_{_}'] = loss.item()

        model2.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model2(inputs)
                loss = criterion2(outputs, targets.unsqueeze(2))
        test_loss_dic[f'm2_{_}'] = loss.item()

    for param1, param2, param_avg in zip(model1.parameters(), model2.parameters(), model3.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for _ in range(epochs):
        model3.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer3.zero_grad()  
            outputs = model3(inputs)  
            loss = criterion3(outputs, targets.unsqueeze(2))  
            loss.backward() 
            optimizer3.step()
        train_loss_dic[f'm3_{_}'] = loss.item()

        model3.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model3(inputs)
                loss = criterion3(outputs, targets.unsqueeze(2))
        test_loss_dic[f'm3_{_}'] = loss.item()

    
    return test_loss_dic, train_loss_dic


def weight_avg_2_classification(epochs, train_loader:DataLoader, val_loader:Dataset, input_size, output_size, device_in_use, model:str = 'Linear'):
    epochs = epochs//3

    criterion = nn.CrossEntropyLoss()

    if model == 'Linear':
        model1 = LinearModel(input_size, output_size).to(device_in_use)
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model2 = LinearModel(input_size, output_size).to(device_in_use) 
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model3 = LinearModel(input_size, output_size).to(device_in_use)
        optimizer3 = optim.SGD(model3.parameters(), lr=0.01)  # Stochastic Gradient Descent
    else:
        model1 = NN(input_size, output_size).to(device_in_use)
        optimizer1 = optim.SGD(model1.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model2 = NN(input_size, output_size).to(device_in_use) 
        optimizer2 = optim.SGD(model2.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model3 = NN(input_size, output_size).to(device_in_use)  
        optimizer3 = optim.SGD(model3.parameters(), lr=0.001)  # Stochastic Gradient Descent

    test_loss_dic = {}
    train_loss_dic = {}

    for _ in range(epochs):
        model1.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer1.zero_grad()  
            outputs = model1(inputs)  
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer1.step() 
        train_loss_dic[f'm1_{_}'] = loss.item()

        model1.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model1(inputs)
                loss = criterion(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm1_{_}'] = loss.item()


    for _ in range(epochs):
        model2.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer2.zero_grad()  
            outputs = model2(inputs)  
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer2.step()
        train_loss_dic[f'm2_{_}'] = loss.item()

        model2.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model2(inputs)
                loss = criterion(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm2_{_}'] = loss.item()

    for param1, param2, param_avg in zip(model1.parameters(), model2.parameters(), model3.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for _ in range(epochs):
        model3.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer3.zero_grad()  
            outputs = model3(inputs)  
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer3.step()
        train_loss_dic[f'm3_{_}'] = loss.item()

        model3.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model3(inputs)
                loss = criterion(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm3_{_}'] = loss.item()
    
    return test_loss_dic, train_loss_dic

def weight_avg_8_classification(epochs, train_loader: DataLoader, val_loader: DataLoader, input_size, output_size, device_in_use, model_type: str = 'Linear'):
    epochs = epochs // 15

    models = []
    optimizers = []
    test_loss_dic = {}
    train_loss_dic = {}

    # Initialize models, optimizers, and criteria
    for i in range(15):
        if model_type == 'Linear':
            mdl = LinearModel(input_size, output_size).to(device_in_use)
            opt = optim.SGD(mdl.parameters(), lr=0.01)
        else:
            mdl = NN(input_size, output_size).to(device_in_use)
            opt = optim.SGD(mdl.parameters(), lr=0.001)
        models.append(mdl)
        optimizers.append(opt)

    # Train each model and record losses
    for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(0), targets.squeeze(0))
                loss.backward()
                optimizer.step()
            train_loss_dic[f'm{idx+1}_{epoch}'] = loss.item()

            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                    outputs = model(inputs)
                    val_loss = criterion(outputs.squeeze(0), targets.squeeze(0))
            test_loss_dic[f'm{idx+1}_{epoch}'] = val_loss.item()

    # Averaging weights for models 1 and 2 to model 9, etc.
    for i in range(0, 8, 2):
        averaged_model = models[i//2 + 8]  # This assumes models 9-12 are averaged versions
        params1 = models[i].parameters()
        params2 = models[i+1].parameters()
        for param1, param2, param_avg in zip(params1, params2, averaged_model.parameters()):
            param_avg.data.copy_((param1.data + param2.data) / 2)

    return test_loss_dic, train_loss_dic


def regular_classification(epochs, train_loader, val_loader, input_size, output_size, device_in_use,  model:str = 'Linear'):
    if model == 'Linear':
        # Initialize the model
        model = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
    else:
        # Initialize the model
        model = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent

    test_loss_dic = {}
    train_loss_dic = {}

    for epoch in range(epochs):
        model.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer.zero_grad()  
            outputs = model(inputs) 
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer.step()  

        train_loss_dic[f'm_{epoch}'] = loss.item()


        model.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(0), targets.squeeze(0))

        test_loss_dic[f'm_{epoch}'] = loss.item()

    return test_loss_dic, train_loss_dic





























