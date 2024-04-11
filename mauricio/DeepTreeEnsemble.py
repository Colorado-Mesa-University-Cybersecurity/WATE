import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets
import random
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor




from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing, load_wine, fetch_covtype

def create_dataloaders_for_dataset(dataset_name, task, test_size=0.2, batch_size=32):
    dataset_load_functions = {
        'breast_cancer': load_breast_cancer,
        'diabetes': load_diabetes,
        'cali_housing': fetch_california_housing,
        'wine': load_wine,
        'covertype': fetch_covtype,
    }

    if dataset_name not in dataset_load_functions:
        raise ValueError(f"Dataset {dataset_name} not recognized. Please choose from {list(dataset_load_functions.keys())}.")

    tasks = [
        'regression',
        'classification',
        ]
    
    if task not in tasks:
        raise ValueError(f"Task {task} not recognized. Please choose from {list(tasks)}.")
    
    dataset = dataset_load_functions[dataset_name]()
    # print(dataset.data.shape)
    X = dataset.data
    # print(X.shape)
    y = dataset.target
    # print(y.shape)

    # makes sure class labels are 0-indexed
    if task == 'classification':
        y = y - y.min()

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Split data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print('xtrain size', len(X_train))
    print('xtest size', len(X_test))
    print('y_train size', len(y_train))
    print('y_test size', len(y_test))
    

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data to tensors
    if task == 'regression':
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    if task == 'classification':
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.int64)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    
    if len(y_train_tensor.shape) == 1:
        y_train_tensor = y_train_tensor.unsqueeze(1)
    if len(y_test_tensor.shape) == 1:
        y_test_tensor = y_test_tensor.unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate input size (number of features)
    input_size = X_train_tensor.shape[1]

    # Calculate output size
    unique_targets = torch.unique(y_train_tensor[:, 0]).numel()
    if unique_targets > 30:  # Assume regression if there are more than 30 unique values, classification otherwise
        output_size = 1  # regression
    else:
        output_size = unique_targets  # Classification task

    return train_dataloader, test_dataloader, input_size, output_size


def create_imageloaders_for_dataset(imageset_name, test_size = 0.1, batch = 32):
    """loads in image datasets into batches for processing
    Will be processing datasets such as: MNIST, Fashion-MNIST, 
    Caltech-101"""

    #TO-DO Wrap into dataloaders inorder 

    # figure out a way to only download the data once the specific name has been called
    # instead of always having the data downloaded. More helpful in long term

    mnist_dataset = datasets.MNIST(root='data', train=True, transform=ToTensor, download=True)
    fashion_mnist_dataset = datasets.FashionMNIST(root='data', train=True, transform=ToTensor, download=True)
    caltech101_dataset = datasets.Caltech101(root='data',transform=ToTensor,download=True)
    
    imageset_load_functions = {
    'MNIST': mnist_dataset,
    'Fashion_MNIST': fashion_mnist_dataset,
    'CalTech101': caltech101_dataset
    }

    imageset = imageset_load_functions[imageset_name]()

    train_size = 1-test_size
    dataset_size = len(imageset)

    train_dataset_size = int(train_size * dataset_size)
    test_dataset_size = dataset_size - train_dataset_size

    train_dataset, test_dataset = random_split(imageset, [train_dataset_size, test_dataset_size])

    first_sample = imageset[0]
    input_size = first_sample[0].size() 
    output_size = len(set(sample[1] for sample in imageset))

    #properly structuring into Pytorch dataset for Dataloader conversion
    trainset = DatasetClass(train_dataset)
    testset = DatasetClass(test_dataset)

    train_loader = DataLoader(trainset, batch_size = batch, shuffle = True)
    test_loader = DataLoader(testset, batch_size = batch, shuffle = False)

    return train_loader, test_loader, input_size, output_size

class DatasetClass(Dataset):
    def init(self, dataset):
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def getitem(self, idx):
        image_tensor, label = self.dataset[idx]

        return image_tensor, label

class DeepTreeEnsemble(object):
    def __init__(self, task_name, model_arch: nn.Module, base_number:int, epochs: int, model_dir:str, train_dataloader, test_dataloader, learning_rate):
        tasks = {
        'regression': nn.MSELoss(),
        'classification': nn.CrossEntropyLoss(),
        }
        if task_name not in tasks:
            raise ValueError(f"Task {task_name} not recognized. Please choose from {list(tasks.keys())}.")
        
        self.criterion=tasks[task_name]
        
        self.model_dir = model_dir
        self.model_arch = model_arch
        self.base_number = base_number
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.task = task_name #criterion should be inferred from this 
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train_DTE(self):
        # ensure that base_number is even
        assert(self.base_number%2 == 0)

        # set up the base number of models 
        print(f'Randomly initializing {self.base_number} models')
        model_list = []
        for i in range(self.base_number):
            model = copy.deepcopy(self.model_arch)
            rand_model = self.initialize_model_with_random_weights(model)
            model_list.append(rand_model)

        # print_model_parameters(model_list[0])
        # print_model_parameters(model_list[1])
        
        for i in range(len(model_list)):
            if i < len(model_list)-1 and self.models_are_equal(model_list[i], model_list[i+1]):
                print("WARNING some models were initialized to the same weights. See model", i)

        iter = 0

        # need lists to record highest metrics from anywhere in DTE
        rmse_list = []
        acc_list = []
        f1_list = []
        # losses too
        min_train_list = []
        min_test_list = []

        if self.task == 'regression':
            for each_model in model_list:
                iter+=1
                primary = self.training_loop(each_model, iter)

                rmse_list.append(primary)
                min_test_list.append(min_test)
                min_train_list.append(min_train)

        if self.task == 'classification':
            for each_model in model_list:
                iter+=1
                primary, secondary, min_test, min_train = self.training_loop(each_model, iter)

                acc_list.append(primary)
                f1_list.append(secondary)
                min_test_list.append(min_test)
                min_train_list.append(min_train)

        models_to_combine = []
        for i in range(self.base_number):
            i+=1
            trained_model = self.model_arch
            trained_model.load_state_dict(torch.load(f'{self.model_dir}model_{i}.pth'))
            models_to_combine.append(trained_model)

        id = self.base_number

        if self.task == 'regression':
            final_model, rmse, min_test, min_train = self.combine_and_retrain(self.model_arch, models_to_combine, self.model_dir, self.epochs, self.criterion, id, rmse_list, min_test_list, min_train_list)
            rmse_list.append(rmse)
            min_test_list.append(min_test)
            min_train_list.append(min_train)

            min_rmse = min(rmse_list)

            print(f' Minimum RMSE Obtained over all training: {min_rmse}')
            return final_model, min_rmse
        
        if self.task == 'classification':
            final_model, acc, f1, min_test, min_train = self.combine_and_retrain(self.model_arch, models_to_combine, self.model_dir, self.epochs, self.criterion, id, acc_list, min_test_list, min_train_list, f1_list)
            acc_list.append(acc)
            f1_list.append(f1)
            min_test_list.append(min_test)
            min_train_list.append(min_train)

            max_acc = max(acc_list)
            max_index = acc_list.index(max(acc_list))
            max_test_f1 = f1_list[max_index]

            print(f' Maximum accuracy: {max_acc}, F1: {max_test_f1}, Test: {min(min_test_list)}, Train: {min(min_train_list)}')

            metrics = {
            'DTE_acc': [max_acc],
            'DTE_f1': [max_test_f1],
            'DTE_test': [min(min_test_list)],
            'DTE_train': [min(min_train_list)]
             }
            
            metrics_df = pd.DataFrame(metrics)
            
            return metrics_df


    def models_are_equal(self, model1, model2):
        # Check if the models have the same number of parameters
        if sum(p.numel() for p in model1.parameters()) != sum(p.numel() for p in model2.parameters()):
            return False
        # Check if the model parameters are equal
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            # print(f'p1: {p1}, p2: {p2}')
            if not torch.equal(p1, p2):
                return False
        return True

    # Recursive function to combine models and retrain
    def combine_and_retrain(self, model, models, model_dir, epochs, criterion, id, primary_metric, min_test_loss, min_train_loss, secondary_metric = []):
        # Base case: If there's only one model, return it
        if len(models) == 1:
            if self.task=='regression':
                rmse = float('inf')
                min_test = float('inf')
                min_train = float('inf')
                return models[0], rmse, min_test, min_train
            if self.task=='classification':
                acc = 0 # temporary values for these metrics that are ultimately ignored
                f1 = 0
                min_test = float('inf')
                min_train = float('inf')
                return models[0], acc, f1, min_test, min_train
        
        # Calculate the midpoint to split the models into two groups
        midpoint = len(models) // 2
        
        # Recursively combine and retrain the left and right halves
        # regression
        if self.task=='regression':
            id += 1
            left_half, rmse, min_test, min_train = self.combine_and_retrain(model, models[:midpoint], model_dir, epochs, criterion, id, primary_metric, min_test_loss, min_train_loss, secondary_metric)
            primary_metric.append(rmse)
            min_test_loss.append(min_test)
            min_train_loss.append(min_train)

            id += 1
            right_half, rmse, min_test, min_train = self.combine_and_retrain(model, models[midpoint:], model_dir, epochs, criterion, id, primary_metric, min_test_loss, min_train_loss, secondary_metric)
            primary_metric.append(rmse)
            min_test_loss.append(min_test)
            min_train_loss.append(min_train)

        #classification
        if self.task=='classification':
            id += 1
            left_half, acc, f1, min_test, min_train = self.combine_and_retrain(model, models[:midpoint], model_dir, epochs, criterion, id, primary_metric, min_test_loss, min_train_loss, secondary_metric)
            primary_metric.append(acc)
            secondary_metric.append(f1)
            min_test_loss.append(min_test)
            min_train_loss.append(min_train)

            id += 1
            right_half, acc, f1, min_test, min_train = self.combine_and_retrain(model, models[midpoint:], model_dir, epochs, criterion, id, primary_metric, min_test_loss, min_train_loss, secondary_metric)
            primary_metric.append(acc)
            secondary_metric.append(f1)
            min_test_loss.append(min_test)
            min_train_loss.append(min_train)

        # Create a new model and average the weights of the left and right halves
        combined_model = model
        for param_left, param_right, param_combined in zip(left_half.parameters(), right_half.parameters(), combined_model.parameters()):
            param_combined.data.copy_(0.5 * param_left.data + 0.5 * param_right.data)
        
        # Retrain the combined model
        if self.task=='regression':
            rmse, min_test, min_train = self.training_loop(combined_model, id)
            return combined_model, rmse, min_test, min_train 
        if self.task=='classification':
            acc, f1, min_test, min_train = self.training_loop(combined_model, id)
            return combined_model, acc, f1, min_test, min_train
        
        

    def initialize_model_with_random_weights(self, model):
        def weight_init(m):
            random_seed = random.randint(1, 10000)  # Consider if you truly need this inside the loop
            torch.manual_seed(random_seed)
            
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):  # Example for convolutional layers, if applicable
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):  # Normalization layers
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                m.running_mean.zero_()
                m.running_var.fill_(1)
            elif isinstance(m, nn.MultiheadAttention):  # Specific to transformer models
                # Custom initialization for multihead attention components if necessary
                pass  # Placeholder for any specific initialization, if needed
            # Add other specific initializations here based on your model's components
            
        model.apply(weight_init)
        return model

    def training_loop(self, model, iter):
        print(f'Training Model with id {iter}')
        # Assume we are running on a CUDA machine
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Keep track of loss and accuracy for plotting
        train_losses = []
        test_losses = []
        avg_train_losses = []
        avg_test_losses = []

        # accuracy metrics for plotting (classification)
        train_accuracies = []
        test_accuracies = []
        avg_train_acc = []
        avg_test_acc = [] 

        # for storing and calculating f-1
        train_f1 = []
        test_f1 = []

        all_labels_test = []
        all_predictions_test = []
        all_labels_train = []
        all_predictions_train = []

        # for rmse (regression)
        train_rmse = []
        test_rmse = []
        mean_train_rmse = []
        mean_test_rmse = []


        # multi class Training loop
        if self.task == 'classification':
            print('in multi')
            for epoch in tqdm(range(self.epochs), desc="Training Process"):
                # Training Phase 
                model.train()
                for (inputs, labels) in self.train_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(torch.long).squeeze().to(device)

                    outputs = model(inputs)

                    outputs = torch.softmax(outputs, dim=1)
                    outputs = outputs.to(torch.float32).to(device)
                    
                    loss = self.criterion(outputs, labels)

                    # Backward and optimizeytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.int64
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track the accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    # print('predicted', predicted)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    train_accuracies.append(correct / total)

                    # track f1
                    all_labels_train.extend(labels.cpu().numpy())
                    all_predictions_train.extend(predicted.cpu().numpy())

                    # Track the loss
                    train_losses.append(loss.item())


                # Testing phase
                model.eval()
                with torch.no_grad():
                    for (inputs, labels) in self.test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(torch.long).squeeze().to(device)

                        outputs = model(inputs)

                        outputs = torch.softmax(outputs, dim=1)
                        outputs = outputs.to(torch.float32).to(device)

                        loss = self.criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)
                        total = labels.size(0)
                        correct = (predicted == labels).sum().item()
                        test_accuracies.append(correct / total)

                        all_labels_test.extend(labels.cpu().numpy())
                        all_predictions_test.extend(predicted.cpu().numpy())

                        test_losses.append(loss.item())
                
                # create a list of average train and test accuacies per epoch
                avg_train_acc.append(np.mean(train_accuracies))
                avg_test_acc.append(np.mean(test_accuracies))

                #create list of average losses
                avg_train_losses.append(np.mean(train_losses))
                avg_test_losses.append(np.mean(test_losses))

                #create list of 'weighted avereage f1
                train_f1.append(f1_score(all_labels_train, all_predictions_train, average="weighted"))
                test_f1.append(f1_score(all_labels_train, all_predictions_train, average="weighted"))

                # metrics per epoch
                # tqdm.write(f'Epoch [{epoch+1}/{self.epochs}], Train F1:{f1_score(all_labels_train, all_predictions_train, average="weighted")}, Train Loss: {np.mean(train_losses):.4f}, Train Acc: {np.mean(train_accuracies):.4f}, Test F1:{f1_score(all_labels_test, all_predictions_test, average="weighted")}, Test Loss: {np.mean(test_losses):.4f}, Test Acc: {np.mean(test_accuracies):.4f}')

        if self.task == 'regression':
            print(' in regress')
            for epoch in tqdm(range(self.epochs), desc="Training Process"):
                model.train()
                for (inputs, labels) in self.train_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(torch.float32).to(device)

                    outputs = model(inputs)
                    outputs.to(device)
                    
                    # print('outputs:', outputs.shape)
                    # print('labels:', labels.shape)
                    loss = self.criterion(outputs, labels)

                    # Backward and optimizeytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.int64
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    rmse_value = rmse(labels, outputs)
                    train_rmse.append(rmse_value)

                    train_losses.append(loss.item())

                # Testing phase
                model.eval()
                with torch.no_grad():
                    for (inputs, labels) in self.test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(torch.float32).to(device)

                        outputs = model(inputs)

                        loss = self.criterion(outputs, labels)

                        rmse_value = rmse(labels, outputs)
                        test_rmse.append(rmse_value)

                        test_losses.append(loss.item())
                        
                # list of mean rmse values per epoch
                mean_train_rmse.append(np.mean(train_rmse))
                mean_test_rmse.append(np.mean(test_rmse))

                #create list of average losses
                avg_train_losses.append(np.mean(train_losses))
                avg_test_losses.append(np.mean(test_losses))

                # tqdm.write(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {np.mean(train_losses):.4f}, Train RMSE:{np.mean(train_rmse)}, Test Loss: {np.mean(test_losses):.4f}, Test RMSE: {np.mean(test_rmse)}')

        
        if self.task == 'classification':
            print(f'Total Epochs: {self.epochs}')
            print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
            print(f'Highest Mean Train Accuracy (over epoch): {max(avg_train_acc)}')
            print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')

            # Record metrics for testing
            max_test_acc = max(avg_test_acc)
            max_index = avg_test_acc.index(max(avg_test_acc))
            max_test_f1 = test_f1[max_index]
            min_train_loss = min(avg_train_losses)
            min_test_loss = min(avg_test_losses)

            print(f'Highest mean Test Accuracy (over epoch): {max_test_acc}')
            print(f' Test F1 score at highest accuracy: {max_test_f1}')


            # # UNCOMENT FOR PLOTS 

            # # Plotting the loss
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, self.epochs+1), avg_train_losses, label='Train Loss')
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, self.epochs+1), avg_test_losses, label='Test Loss')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.title('Training and Test Loss Curve')
            # plt.legend()

            # # Plotting the accuracy
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(avg_train_acc, label='Train Accuracy')
            # plt.subplot(1,2,2)
            # plt.plot(avg_test_acc, label='Test Accuracy')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.title('Training and Test Accuracy Curve')
            # plt.legend()
            # plt.grid()

            # plt.show()


            print(f'{self.model_dir}model_{iter}.pth \n')
            torch.save(model.state_dict(), f'{self.model_dir}model_{iter}.pth')

            return max_test_acc, max_test_f1, min_test_loss, min_train_loss


        if self.task == 'regression':
            print(f'Total Epochs: {self.epochs}')
            print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
            print(f'Lowest Mean Train RMSE (over epoch): {min(mean_train_rmse)}')
            print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')
            print(f'Lowest mean Test RMSE (over epoch): {min(mean_test_rmse)}')

            lowest_mse = min(mean_test_rmse)
            min_train_loss = min(avg_train_losses)
            min_test_loss = min(avg_test_losses)
           
            # # Uncomment for charts
            # # Plotting the loss
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, self.epochs+1), avg_train_losses, label='Train Loss')
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, self.epochs+1), avg_test_losses, label='Test Loss')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.title('Training and Test Loss Curve')
            # plt.legend()

            # # Plotting the accuracy
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(mean_train_rmse, label='Train RMSE')
            # plt.subplot(1,2,2)
            # plt.plot(mean_test_rmse, label='Test RMSE')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.title('Training and Test RMSE Curve')
            # plt.legend()
            # plt.grid()


            # plt.show()

            print(f'{self.model_dir}model_{iter}.pth')
            torch.save(model.state_dict(), f'{self.model_dir}model_{iter}.pth')

            return lowest_mse, min_test_loss, min_train_loss


    def single_model(self):
         # create new model
        modelCopy = copy.deepcopy(self.model_arch)
        model = self.initialize_model_with_random_weights(modelCopy)
        # Assume we are running on a CUDA machine
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Keep track of loss and accuracy for plotting
        train_losses = []
        test_losses = []
        avg_train_losses = []
        avg_test_losses = []

        # accuracy metrics for plotting (classification)
        train_accuracies = []
        test_accuracies = []
        avg_train_acc = []
        avg_test_acc = [] 

        # for storing and calculating f-1
        train_f1 = []
        test_f1 = []

        all_labels_test = []
        all_predictions_test = []
        all_labels_train = []
        all_predictions_train = []

        # for rmse (regression)
        train_rmse = []
        test_rmse = []
        mean_train_rmse = []
        mean_test_rmse = []

        single_epoch_range = number_of_models(self.base_number) * self.epochs

        # classification Training loop
        if self.task == 'classification':
            print('in multi')
            for epoch in tqdm(range(single_epoch_range), desc="Training Process"):
                # Training Phase 
                model.train()
                for (inputs, labels) in self.train_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(torch.long).squeeze().to(device)

                    outputs = model(inputs)

                    outputs = torch.softmax(outputs, dim=1)
                    outputs = outputs.to(torch.float32).to(device)
                    
                    loss = self.criterion(outputs, labels)

                    # Backward and optimizeytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.int64
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track the accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    # print('predicted', predicted)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    train_accuracies.append(correct / total)

                    # track f1
                    all_labels_train.extend(labels.cpu().numpy())
                    all_predictions_train.extend(predicted.cpu().numpy())

                    # Track the loss
                    train_losses.append(loss.item())


                # Testing phase
                model.eval()
                with torch.no_grad():
                    for (inputs, labels) in self.test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(torch.long).squeeze().to(device)

                        outputs = model(inputs)

                        outputs = torch.softmax(outputs, dim=1)
                        outputs = outputs.to(torch.float32).to(device)

                        loss = self.criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)
                        total = labels.size(0)
                        correct = (predicted == labels).sum().item()
                        test_accuracies.append(correct / total)

                        all_labels_test.extend(labels.cpu().numpy())
                        all_predictions_test.extend(predicted.cpu().numpy())

                        test_losses.append(loss.item())
                
                # create a list of average train and test accuacies per epoch
                avg_train_acc.append(np.mean(train_accuracies))
                avg_test_acc.append(np.mean(test_accuracies))

                #create list of average losses
                avg_train_losses.append(np.mean(train_losses))
                avg_test_losses.append(np.mean(test_losses))

                #create list of 'weighted avereage f1
                train_f1.append(f1_score(all_labels_train, all_predictions_train, average="weighted"))
                test_f1.append(f1_score(all_labels_train, all_predictions_train, average="weighted"))

                # metrics per epoch
                # tqdm.write(f'Epoch [{epoch+1}/{self.epochs}], Train F1:{f1_score(all_labels_train, all_predictions_train, average="weighted")}, Train Loss: {np.mean(train_losses):.4f}, Train Acc: {np.mean(train_accuracies):.4f}, Test F1:{f1_score(all_labels_test, all_predictions_test, average="weighted")}, Test Loss: {np.mean(test_losses):.4f}, Test Acc: {np.mean(test_accuracies):.4f}')

        if self.task == 'regression':
            print(' in regress')
            for epoch in tqdm(range(single_epoch_range), desc="Training Process"):
                model.train()
                for (inputs, labels) in self.train_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(torch.float32).to(device)

                    outputs = model(inputs)
                    outputs.to(device)
                    
                    # print('outputs:', outputs.shape)
                    # print('labels:', labels.shape)
                    loss = self.criterion(outputs, labels)

                    # Backward and optimizeytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.int64
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    rmse_value = rmse(labels, outputs)
                    train_rmse.append(rmse_value)

                    train_losses.append(loss.item())

                # Testing phase
                model.eval()
                with torch.no_grad():
                    for (inputs, labels) in self.test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(torch.float32).to(device)

                        outputs = model(inputs)

                        loss = self.criterion(outputs, labels)

                        rmse_value = rmse(labels, outputs)
                        test_rmse.append(rmse_value)

                        test_losses.append(loss.item())
                        
                # list of mean rmse values per epoch
                mean_train_rmse.append(np.mean(train_rmse))
                mean_test_rmse.append(np.mean(test_rmse))

                #create list of average losses
                avg_train_losses.append(np.mean(train_losses))
                avg_test_losses.append(np.mean(test_losses))

                # tqdm.write(f'Epoch [{epoch+1}/{single_epoch_range}], Train Loss: {np.mean(train_losses):.4f}, Train RMSE:{np.mean(train_rmse)}, Test Loss: {np.mean(test_losses):.4f}, Test RMSE: {np.mean(test_rmse)}')

        
        if self.task == 'classification':
            print(f'Total Epochs: {single_epoch_range}')
            print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
            print(f'Highest Mean Train Accuracy (over epoch): {max(avg_train_acc)}')
            print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')

            # Record metrics for testing
            max_test_acc = max(avg_test_acc)
            max_index = avg_test_acc.index(max(avg_test_acc))
            max_test_f1 = test_f1[max_index]
            min_train_loss = min(avg_train_losses)
            min_test_loss = min(avg_test_losses)

            print(f'Highest mean Test Accuracy (over epoch): {max_test_acc}')
            print(f' Test F1 score at highest accuracy: {max_test_f1}')


            # # UNCOMENT FOR PLOTS 

            # # Plotting the loss
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, single_epoch_range+1), avg_train_losses, label='Train Loss')
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, single_epoch_range+1), avg_test_losses, label='Test Loss')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.title('Training and Test Loss Curve')
            # plt.legend()

            # # Plotting the accuracy
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(avg_train_acc, label='Train Accuracy')
            # plt.subplot(1,2,2)
            # plt.plot(avg_test_acc, label='Test Accuracy')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.title('Training and Test Accuracy Curve')
            # plt.legend()
            # plt.grid()

            # plt.show()


            metrics = {
            'single_acc': [max_test_acc],
            'single_f1': [max_test_f1],
            'single_test': [min_test_loss],
            'single_train': [min_train_loss]
             }
            
            metrics_df = pd.DataFrame(metrics)

            return metrics_df


        if self.task == 'regression':
            print(f'Total Epochs: {single_epoch_range}')
            print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
            print(f'Lowest Mean Train RMSE (over epoch): {min(mean_train_rmse)}')
            print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')
            print(f'Lowest mean Test RMSE (over epoch): {min(mean_test_rmse)}')

            lowest_mse = min(mean_test_rmse)
            min_train_loss = min(avg_train_losses)
            min_test_loss = min(avg_test_losses)
           
            # # Uncomment for charts
            # # Plotting the loss
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, single_epoch_range+1), avg_train_losses, label='Train Loss')
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, single_epoch_range+1), avg_test_losses, label='Test Loss')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.title('Training and Test Loss Curve')
            # plt.legend()

            # # Plotting the accuracy
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 2)
            # plt.plot(mean_train_rmse, label='Train RMSE')
            # plt.subplot(1,2,2)
            # plt.plot(mean_test_rmse, label='Test RMSE')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.title('Training and Test RMSE Curve')
            # plt.legend()
            # plt.grid()


            # plt.show()

            metrics = {
            'single_rmse': [lowest_mse],
            'single_test': [min_test_loss],
            'single_train': [min_train_loss]
             }
            
            metrics_df = pd.DataFrame(metrics)

            return metrics_df

    def train_single_model(self):
        # create new model
        modelCopy = copy.deepcopy(self.model_arch)
        model = self.initialize_model_with_random_weights(modelCopy)
        # Assume we are running on a CUDA machine
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Keep track of loss and accuracy for plotting
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        mean_train_acc = []
        mean_test_acc = [] 

        f1scores = []
        all_labels_test = []
        all_predictions_test = []
        all_labels_train = []
        all_predictions_train = []

        single_epoch_range = number_of_models(self.base_number) * self.epochs
        # Training loop
        for epoch in tqdm(range(single_epoch_range), desc="Training Process"):
            # Training Phase 
            model.train()
            for (inputs, labels) in self.train_dataloader:
                inputs = inputs.to(torch.float32).to(device)
                # print(inputs.dtype)
                labels = labels.to(device)
                # print(labels.dtype)

                # print('inputs: ', inputs)
                # print('labels: ', labels.shape)

                # Forward passs
                # print('inputs: ', inputs.shape)
                outputs = model(inputs)
                # print('outputs: ', outputs.shape)
                if self.task == 'classification':
                    labels = labels.to(torch.long)
                    outputs = torch.softmax(outputs, dim=1)
                    outputs = outputs.to(torch.float32)

                loss = self.criterion(outputs, labels.squeeze())

                # Backward and optimizeytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.float32Layer dtype:  torch.float32tensor dytpe:  torch.int64
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                _, predicted = torch.max(outputs.data, 1)
                # print('predicted', predicted)
                total = labels.size(0)
                correct = (predicted == labels.squeeze()).sum().item()
                train_accuracies.append(correct / total)

                # track f1
                all_labels_train.extend(labels.cpu().numpy())
                all_predictions_train.extend(predicted.cpu().numpy())

                # Track the loss
                train_losses.append(loss.item())

            # Testing phase
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(self.test_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)

                    if self.task == 'classification':
                        labels = labels.to(torch.long)
                        outputs = torch.softmax(outputs, dim=1)
                        outputs = outputs.to(torch.float32)

                    loss = self.criterion(outputs, labels.squeeze())

                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels.squeeze()).sum().item()
                    test_accuracies.append(correct / total)

                    all_labels_test.extend(labels.cpu().numpy())
                    all_predictions_test.extend(predicted.cpu().numpy())

                    test_losses.append(loss.item())

            mean_train_acc.append(np.mean(train_accuracies))
            mean_test_acc.append(np.mean(test_accuracies))
            # print(f1_score(all_labels_train, all_predictions_test, average='weighted'))

            tqdm.write(f'Epoch [{epoch+1}/{single_epoch_range}], Train F1:{f1_score(all_labels_train, all_predictions_train, average="weighted")}, Train Loss: {np.mean(train_losses):.4f}, Train Acc: {np.mean(train_accuracies):.4f}, Test F1:{f1_score(all_labels_test, all_predictions_test, average="weighted")}, Test Loss: {np.mean(test_losses):.4f}, Test Acc: {np.mean(test_accuracies):.4f}')

        print(f'Total Epochs: {single_epoch_range}')
        print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
        print(f'Highest Mean Train Accuracy (over epoch): {max(mean_train_acc)}')
        print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')
        print(f'Highest Mean Test Accuracy (over epoch): {max(mean_test_acc)}')
        
        # Calculate average loss per epoch
        avg_train_losses = [np.mean(train_losses[i:i+len(self.train_dataloader)]) for i in range(0, len(train_losses), len(self.train_dataloader))]
        avg_test_losses = [np.mean(test_losses[i:i+len(self.test_dataloader)]) for i in range(0, len(test_losses), len(self.test_dataloader))]

        # Calculate average accuracy per epoch
        avg_train_accuracies = [np.mean(train_accuracies[i:i+len(self.train_dataloader)]) for i in range(0, len(train_accuracies), len(self.train_dataloader))]
        avg_test_accuracies = [np.mean(test_accuracies[i:i+len(self.test_dataloader)]) for i in range(0, len(test_accuracies), len(self.test_dataloader))]

        # Uncomment for plots
        # Plotting the loss
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 2, 2)
        # plt.plot(range(1, single_epoch_range+1), avg_train_losses, label='Train Loss')
        # plt.subplot(1, 2, 2)
        # plt.plot(range(1, single_epoch_range+1), avg_test_losses, label='Test Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training and Test Loss Curve')
        # plt.legend()

        # # Plotting the accuracy
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 2, 2)
        # plt.plot(avg_train_accuracies, label='Train Accuracy')
        # plt.subplot(1,2,2)
        # plt.plot(avg_test_accuracies, label='Test Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Training and Test Accuracy Curve')
        # plt.legend()
        # plt.grid()
        # plt.show()

        metrics = {
            'single_acc': [max_acc],
            'single_f1': [max_test_f1],
            'single_test': [min(avg_test_accuracies)],
            'single_train': [min(avg_train_accuracies)]
             }
            
        metrics_df = pd.DataFrame(metrics)
            
        return metrics_df

def number_of_models(base_number):
    # Calculate the total number of nodes in a binary tree of a given depth
    return 2 * base_number - 1

def rmse(y_true, y_pred):
    # Calculate the squared differences
    squared_diff = (y_true - y_pred)**2

    try:
        # Calculate the mean of the squared differences
        mean_squared_diff = torch.mean(squared_diff)

        # Calculate the square root to obtain RMSE
        rmse = torch.sqrt(mean_squared_diff)

        return rmse.item()  # Convert to a Python float
    except:
        # Calculate the mean of the squared differences
        mean_squared_diff = np.mean(squared_diff)

        # Calculate the square root to obtain RMSE
        rmse = np.sqrt(mean_squared_diff)

        return rmse