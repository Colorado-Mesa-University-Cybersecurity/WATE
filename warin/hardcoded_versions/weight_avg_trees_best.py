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

def weight_avg_8_classification(epochs, train_loader:DataLoader, val_loader:Dataset, input_size, output_size, device_in_use, model:str = 'Linear'):
    epochs = epochs//15

    if model == 'Linear':
        model1 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion1 = nn.CrossEntropyLoss()  
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model2 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion2 = nn.CrossEntropyLoss()  
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model3 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion3 = nn.CrossEntropyLoss()  
        optimizer3 = optim.SGD(model3.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model4 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion4 = nn.CrossEntropyLoss()  
        optimizer4 = optim.SGD(model4.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model5 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion5 = nn.CrossEntropyLoss()  
        optimizer5 = optim.SGD(model5.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model6 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion6 = nn.CrossEntropyLoss()  
        optimizer6 = optim.SGD(model6.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model7 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion7 = nn.CrossEntropyLoss()  
        optimizer7 = optim.SGD(model7.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model8 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion8 = nn.CrossEntropyLoss()  
        optimizer8 = optim.SGD(model8.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model9 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion9 = nn.CrossEntropyLoss()  
        optimizer9 = optim.SGD(model9.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model10 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion10 = nn.CrossEntropyLoss()  
        optimizer10 = optim.SGD(model10.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model11 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion11 = nn.CrossEntropyLoss()  
        optimizer11 = optim.SGD(model11.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model12 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion12 = nn.CrossEntropyLoss()  
        optimizer12 = optim.SGD(model12.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model13 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion13 = nn.CrossEntropyLoss()  
        optimizer13 = optim.SGD(model13.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model14 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion14 = nn.CrossEntropyLoss()  
        optimizer14 = optim.SGD(model14.parameters(), lr=0.01)  # Stochastic Gradient Descent

        model15 = LinearModel(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion15 = nn.CrossEntropyLoss()  
        optimizer15 = optim.SGD(model15.parameters(), lr=0.01)  # Stochastic Gradient Descent

    else:
        model1 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion1 = nn.CrossEntropyLoss()  
        optimizer1 = optim.SGD(model1.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model2 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion2 = nn.CrossEntropyLoss()  
        optimizer2 = optim.SGD(model2.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model3 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion3 = nn.CrossEntropyLoss()  
        optimizer3 = optim.SGD(model3.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model4 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion4 = nn.CrossEntropyLoss()  
        optimizer4 = optim.SGD(model4.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model5 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion5 = nn.CrossEntropyLoss()  
        optimizer5 = optim.SGD(model5.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model6 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion6 = nn.CrossEntropyLoss()  
        optimizer6 = optim.SGD(model6.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model7 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion7 = nn.CrossEntropyLoss()  
        optimizer7 = optim.SGD(model7.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model8 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion8 = nn.CrossEntropyLoss()  
        optimizer8 = optim.SGD(model8.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model9 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion9 = nn.CrossEntropyLoss()  
        optimizer9 = optim.SGD(model9.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model10 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion10 = nn.CrossEntropyLoss()  
        optimizer10 = optim.SGD(model10.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model11 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion11 = nn.CrossEntropyLoss()  
        optimizer11 = optim.SGD(model11.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model12 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion12 = nn.CrossEntropyLoss()  
        optimizer12 = optim.SGD(model12.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model13 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion13 = nn.CrossEntropyLoss()  
        optimizer13 = optim.SGD(model13.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model14 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion14 = nn.CrossEntropyLoss()  
        optimizer14 = optim.SGD(model14.parameters(), lr=0.001)  # Stochastic Gradient Descent

        model15 = NN(input_size, output_size).to(device_in_use)

        # Define loss function and optimizer
        criterion15 = nn.CrossEntropyLoss()  
        optimizer15 = optim.SGD(model15.parameters(), lr=0.001)  # Stochastic Gradient Descent

    test_loss_dic = {}
    train_loss_dic = {}

    for _ in range(epochs):
        model1.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer1.zero_grad()  
            outputs = model1(inputs)  
            loss = criterion1(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer1.step() 
        train_loss_dic[f'm1_{_}'] = loss.item()

        model1.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model1(inputs)
                loss = criterion1(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm1_{_}'] = loss.item()

    for _ in range(epochs):
        model2.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer2.zero_grad()  
            outputs = model2(inputs)  
            loss = criterion2(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer2.step()
        train_loss_dic[f'm2_{_}'] = loss.item()

        model2.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model2(inputs)
                loss = criterion2(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm2_{_}'] = loss.item()

    for _ in range(epochs):
        model3.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer3.zero_grad()  
            outputs = model3(inputs)  
            loss = criterion3(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer3.step()
        train_loss_dic[f'm3_{_}'] = loss.item()

        model3.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model3(inputs)
                loss = criterion3(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm3_{_}'] = loss.item()

    for _ in range(epochs):
        model4.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer4.zero_grad()  
            outputs = model4(inputs)  
            loss = criterion4(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer4.step()
        train_loss_dic[f'm4_{_}'] = loss.item()

        model4.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model4(inputs)
                loss = criterion4(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm4_{_}'] = loss.item()
    
    for _ in range(epochs):
        model5.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer5.zero_grad()  
            outputs = model5(inputs)  
            loss = criterion5(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer5.step() 
        train_loss_dic[f'm5_{_}'] = loss.item()

        model5.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model5(inputs)
                loss = criterion5(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm5_{_}'] = loss.item()

    for _ in range(epochs):
        model6.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer6.zero_grad()  
            outputs = model6(inputs)  
            loss = criterion6(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer6.step()
        train_loss_dic[f'm6_{_}'] = loss.item()

        model6.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model6(inputs)
                loss = criterion6(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm6_{_}'] = loss.item()

    for _ in range(epochs):
        model7.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer7.zero_grad()  
            outputs = model7(inputs)  
            loss = criterion7(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer7.step()
        train_loss_dic[f'm7_{_}'] = loss.item()

        model7.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model7(inputs)
                loss = criterion7(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm7_{_}'] = loss.item()

    for _ in range(epochs):
        model8.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer8.zero_grad()  
            outputs = model8(inputs)  
            loss = criterion8(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer8.step()
        train_loss_dic[f'm8_{_}'] = loss.item()

        model8.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model8(inputs)
                loss = criterion8(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm8_{_}'] = loss.item()

    # Now all base models are trained, now average and do second layer

    for param1, param2, param_avg in zip(model1.parameters(), model2.parameters(), model9.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for param1, param2, param_avg in zip(model3.parameters(), model4.parameters(), model10.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for param1, param2, param_avg in zip(model5.parameters(), model6.parameters(), model11.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for param1, param2, param_avg in zip(model7.parameters(), model8.parameters(), model12.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for _ in range(epochs):
        model9.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer9.zero_grad()  
            outputs = model9(inputs)  
            loss = criterion9(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer9.step()
        train_loss_dic[f'm9_{_}'] = loss.item()

        model9.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model9(inputs)
                loss = criterion9(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm9_{_}'] = loss.item()

    for _ in range(epochs):
        model10.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer10.zero_grad()  
            outputs = model10(inputs)  
            loss = criterion10(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer10.step()
        train_loss_dic[f'm10_{_}'] = loss.item()

        model10.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model10(inputs)
                loss = criterion10(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm10_{_}'] = loss.item()

    for _ in range(epochs):
        model11.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer11.zero_grad()  
            outputs = model11(inputs)  
            loss = criterion11(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer11.step()
        train_loss_dic[f'm11_{_}'] = loss.item()

        model11.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model11(inputs)
                loss = criterion11(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm11_{_}'] = loss.item()

    for _ in range(epochs):
        model12.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer12.zero_grad()  
            outputs = model12(inputs)  
            loss = criterion12(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer12.step()
        train_loss_dic[f'm12_{_}'] = loss.item()

        model12.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model12(inputs)
                loss = criterion12(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm12_{_}'] = loss.item()

    # Now average these 4 to make the third layer of two

    for param1, param2, param_avg in zip(model9.parameters(), model10.parameters(), model13.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)
    
    for param1, param2, param_avg in zip(model11.parameters(), model12.parameters(), model14.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for _ in range(epochs):
        model13.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer13.zero_grad()  
            outputs = model13(inputs)  
            loss = criterion13(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer13.step()
        train_loss_dic[f'm13_{_}'] = loss.item()

        model13.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model13(inputs)
                loss = criterion13(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm13_{_}'] = loss.item()

    for _ in range(epochs):
        model14.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer14.zero_grad()  
            outputs = model14(inputs)  
            loss = criterion14(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer14.step()
        train_loss_dic[f'm14_{_}'] = loss.item()

        model14.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model14(inputs)
                loss = criterion14(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm14_{_}'] = loss.item()

    for param1, param2, param_avg in zip(model13.parameters(), model14.parameters(), model15.parameters()):
        # Average the weights and update the parameters of the averaged_model
        param_avg.data.copy_((param1.data + param2.data) / 2)

    for _ in range(epochs):
        model15.train()  

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
            optimizer15.zero_grad()  
            outputs = model15(inputs)  
            loss = criterion15(outputs.squeeze(0), targets.squeeze(0))  
            loss.backward() 
            optimizer15.step()
        train_loss_dic[f'm15_{_}'] = loss.item()

        model15.eval()  

        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_in_use), targets.to(device_in_use)
                outputs = model15(inputs)
                loss = criterion15(outputs.squeeze(0), targets.squeeze(0))
        test_loss_dic[f'm15_{_}'] = loss.item()
    
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





























