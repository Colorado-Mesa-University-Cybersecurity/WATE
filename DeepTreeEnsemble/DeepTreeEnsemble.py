import torch
import torch.nn as nn
import torch.nn.init as init
import random
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


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
                print("WARNING so models were initialized to the same weights. See model", i)

        iter = 0
        for each_model in model_list:
            iter+=1
            self.training_loop(each_model, iter)

        models_to_combine = []
        for i in range(self.base_number):
            i+=1
            trained_model = self.model_arch
            trained_model.load_state_dict(torch.load(f'{self.model_dir}model_{i}.pth'))
            models_to_combine.append(trained_model)

        id = self.base_number
        final_model = self.combine_and_retrain(self.model_arch, models_to_combine, self.model_dir, self.epochs, self.criterion, id)

        return final_model

    def train_DTE_patience(self, model_arch: nn.Module, base_number:int, epochs: int, patience:int, model_dir, criterion):
        # ensure that base_number is even
        assert(base_number%2 == 0)

        # set up the base number of models 
        model_list = []
        for i in range(base_number):
            print(f'Randomly initializing {base_number} models')
            model = copy.deepcopy(model_arch)
            rand_model = initialize_model_with_random_weights(model)
            model_list.append(rand_model)

        # print_model_parameters(model_list[0])
        # print_model_parameters(model_list[1])

        if models_are_equal(model_list[0], model_list[1]):
            print("model1 and model2 are the same.")
        else:
            print("model1 and model2 are different.")

        iter = 0
        for each_model in model_list:
            iter+=1
            training_loop_with_early_stopping(each_model, model_dir, patience, epochs, criterion, iter)

        models_to_combine = []
        for i in range(base_number):
            i+=1
            trained_model = model_arch
            trained_model.load_state_dict(torch.load(f'{model_dir}model_{i}.pth'))
            models_to_combine.append(trained_model)

        id = base_number
        final_model = combine_and_retrain(model_arch, models_to_combine, model_dir, epochs, criterion, id)

        return final_model

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
    def combine_and_retrain(self, model, models, model_dir, epochs, criterion, id):
        # Base case: If there's only one model, return it
        if len(models) == 1:
            return models[0]
        
        # Calculate the midpoint to split the models into two groups
        midpoint = len(models) // 2
        
        # Recursively combine and retrain the left and right halves
        id += 1
        left_half = self.combine_and_retrain(model, models[:midpoint], model_dir, epochs, criterion, id)

        id += 1
        right_half = self.combine_and_retrain(model, models[midpoint:], model_dir, epochs, criterion, id)
        
        # Create a new model and average the weights of the left and right halves
        combined_model = model
        for param_left, param_right, param_combined in zip(left_half.parameters(), right_half.parameters(), combined_model.parameters()):
            param_combined.data.copy_(0.5 * param_left.data + 0.5 * param_right.data)
        
        # Retrain the combined model
        self.training_loop(combined_model, id)
        
        return combined_model

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

        # for calculating f-1
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

                # metrics per epoch
                tqdm.write(f'Epoch [{epoch+1}/{self.epochs}], Train F1:{f1_score(all_labels_train, all_predictions_train, average="weighted")}, Train Loss: {np.mean(train_losses):.4f}, Train Acc: {np.mean(train_accuracies):.4f}, Test F1:{f1_score(all_labels_test, all_predictions_test, average="weighted")}, Test Loss: {np.mean(test_losses):.4f}, Test Acc: {np.mean(test_accuracies):.4f}')

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

                        loss = self.criterion(outputs, labels.squeeze())

                        rmse_value = rmse(labels.unsqueeze(1), outputs)
                        test_rmse.append(rmse_value)

                        test_losses.append(loss.item())
                        
                # list of mean rmse values per epoch
                mean_train_rmse.append(np.mean(train_rmse))
                mean_test_rmse.append(np.mean(test_rmse))

                #create list of average losses
                avg_train_losses.append(np.mean(train_losses))
                avg_test_losses.append(np.mean(test_losses))

                tqdm.write(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {np.mean(train_losses):.4f}, Train RMSE:{np.mean(train_rmse)}, Test Loss: {np.mean(test_losses):.4f}, Test RMSE: {np.mean(test_rmse)}')

        
        if self.task == 'classification':
            print(f'Total Epochs: {self.epochs}')
            print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
            print(f'Highest Mean Train Accuracy (over epoch): {max(avg_train_acc)}')
            print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')
            print(f'Highest mean Test Accuracy (over epoch): {max(avg_test_acc)}')

            # Plotting the loss
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 2)
            plt.plot(range(1, self.epochs+1), avg_train_losses, label='Train Loss')
            plt.subplot(1, 2, 2)
            plt.plot(range(1, self.epochs+1), avg_test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss Curve')
            plt.legend()

            # Plotting the accuracy
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 2)
            plt.plot(avg_train_acc, label='Train Accuracy')
            plt.subplot(1,2,2)
            plt.plot(avg_test_acc, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Test Accuracy Curve')
            plt.legend()
            plt.grid()


            plt.show()
            print(f'{self.model_dir}model_{iter}.pth')
            torch.save(model.state_dict(), f'{self.model_dir}model_{iter}.pth')


        if self.task == 'regression':
            print(f'Total Epochs: {self.epochs}')
            print(f'Train loss start: {train_losses[0]}, Train loss end: {train_losses[-1]}')
            print(f'Lowest Mean Train RMSE (over epoch): {min(mean_train_rmse)}')
            print(f'Test loss start: {test_losses[0]}, Test loss end: {test_losses[-1]}')
            print(f'Lowest mean Test RMSE (over epoch): {min(mean_test_rmse)}')
           
            print(len(mean_train_rmse), len(mean_test_rmse), self.epochs+1)
            # Plotting the loss
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 2)
            plt.plot(range(1, self.epochs+1), avg_train_losses, label='Train Loss')
            plt.subplot(1, 2, 2)
            plt.plot(range(1, self.epochs+1), avg_test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss Curve')
            plt.legend()

            # Plotting the accuracy
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 2)
            plt.plot(mean_train_rmse, label='Train RMSE')
            plt.subplot(1,2,2)
            plt.plot(mean_test_rmse, label='Test RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Test RMSE Curve')
            plt.legend()
            plt.grid()


            plt.show()
            print(f'{self.model_dir}model_{iter}.pth')
            torch.save(model.state_dict(), f'{self.model_dir}model_{iter}.pth')


            

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

        # Plotting the loss
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 2)
        plt.plot(range(1, single_epoch_range+1), avg_train_losses, label='Train Loss')
        plt.subplot(1, 2, 2)
        plt.plot(range(1, single_epoch_range+1), avg_test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curve')
        plt.legend()

        # Plotting the accuracy
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 2)
        plt.plot(avg_train_accuracies, label='Train Accuracy')
        plt.subplot(1,2,2)
        plt.plot(avg_test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy Curve')
        plt.legend()
        plt.grid()


        plt.show()
        

    def training_loop_with_early_stopping(self, model, model_dir, patience, criterion, iter):
        # Assume we are running on a CUDA machine if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Keep track of loss and accuracy for plotting
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        best_test_loss = float('inf')
        epochs_without_improvement = 0

        max_epochs = 100  # You can adjust this based on your preference

        # Training loop
        for epoch in range(max_epochs):
            # Training Phase 
            model.train()
            for (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                train_accuracies.append(correct / total)

                # Track the loss
                train_losses.append(loss.item())

            # Testing phase
            model.eval()
            with torch.no_grad():
                for (inputs, labels) in enumerate(test_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    test_accuracies.append(correct / total)

                    test_losses.append(loss.item())

            avg_test_loss = np.mean(test_losses)

            # Print and log the epoch-wise training and testing metrics
            print(f'Epoch [{epoch + 1}/{max_epochs}], Train Loss: {np.mean(train_losses):.4f}, Train Acc: {np.mean(train_accuracies):.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {np.mean(test_accuracies):.4f}')

            # Check for early stopping
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping! Training stopped.")
                break

        # Calculate average loss per epoch
        avg_train_losses = [np.mean(train_losses[i:i + len(train_dataloader)]) for i in
                            range(0, len(train_losses), len(train_dataloader))]
        avg_test_losses = [np.mean(test_losses[i:i + len(test_dataloader)]) for i in
                        range(0, len(test_losses), len(test_dataloader))]

        # Calculate average accuracy per epoch
        avg_train_accuracies = [np.mean(train_accuracies[i:i + len(train_dataloader)]) for i in
                                range(0, len(train_accuracies), len(train_dataloader))]
        avg_test_accuracies = [np.mean(test_accuracies[i:i + len(test_dataloader)]) for i in
                            range(0, len(test_accuracies), len(test_dataloader))]

        # Plotting the loss
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Train Loss')
        plt.plot(range(1, len(avg_test_losses) + 1), avg_test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curve')
        plt.legend()

        # Plotting the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(avg_train_accuracies) + 1), avg_train_accuracies, label='Train Accuracy')
        plt.plot(range(1, len(avg_test_accuracies) + 1), avg_test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy Curve')
        plt.legend()
        plt.grid()

        plt.show()

        # Save the trained model's weights with a unique identifier based on the iteration
        print(f'Saving model to {model_dir}model_{iter}.pth')
        torch.save(model.state_dict(), f'{model_dir}model_{iter}.pth')

    
        # ensure that base_number is even
        assert(base_number%2 == 0)

        # set up the base number of models 
        model_list = []
        for i in range(base_number):
            model = copy.deepcopy(model_arch)
            print(i)
            rand_model = initialize_model_with_random_weights(model)
            model_list.append(rand_model)

        # print_model_parameters(model_list[0])
        # print_model_parameters(model_list[1])

        if models_are_equal(model_list[0], model_list[1]):
            print("model1 and model2 are the same.")
        else:
            print("model1 and model2 are different.")

        iter = 0
        for each_model in model_list:
            iter+=1
            training_loop_with_early_stopping(each_model, model_dir, patience, epochs, criterion, iter)

        models_to_combine = []
        for i in range(base_number):
            i+=1
            trained_model = model_arch
            trained_model.load_state_dict(torch.load(f'{model_dir}model_{i}.pth'))
            models_to_combine.append(trained_model)

        id = base_number
        final_model = combine_and_retrain(model_arch, models_to_combine, model_dir, epochs, criterion, id)

        return final_model

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