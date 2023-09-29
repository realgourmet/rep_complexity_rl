
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

from cs285.infrastructure import pytorch_util as ptu

import wandb
import random # for demo script

wandb.login()

class Approx(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        # self.task = None
        # self.logger = Logger(self.params['logdir'])



    def prepare_data(self, file_path):

        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # self.task = task
        task = self.params['task']

        # self.models = {}
        # self.optimizer = {}
        # self.criterion = {}
        
        # Step 3: Preprocess the data if needed
        # Depending on your data, you may need to preprocess it (e.g., normalize, reshape, etc.).

        # Split the data into features and labels (assuming the data is a list of tuples)

        #############################################################
        # obs should be array([[*, *, ... ]])
        # acs should be array([[*, *, ... ]])
        # next_obs should be array([*, *, ... ])
        # q_value should be array([[*, *, ... ]])
        # reward should be *
        #############################################################
        obs = [entry['observations'][0] for entry in data]
        acs = [entry['actions'][0] for entry in data]
        next_obs = [entry['next_obs'] for entry in data]
        q_values = [entry['q_value'][0] for entry in data]
        rews = [[entry['reward']] for entry in data]

        # Convert them to NumPy arrays and then to PyTorch tensors
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        acs = torch.tensor(np.array(acs), dtype=torch.float32)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
        rews = torch.tensor(np.array(rews), dtype=torch.float32)
        q_values = torch.tensor(np.array(q_values), dtype=torch.float32)

        # q_var = torch.norm(q_values, p='fro')
        # obs_var = torch.norm(next_obs, p='fro')

        print("sizes: ", obs.size(), acs.size(), next_obs.size(), q_values.size(), rews.size())

        # Step 4: Define a neural network model

        if task == "value":
            total_variance = torch.norm(q_values, p='fro')

            model = ptu.build_mlp(
                        obs.size(dim=1) + acs.size(dim=1),
                        1,
                        n_layers=self.params['n_layers'],
                        size=self.params['size'],
                        activation='relu'
                    )

            # Step 5: Define a loss function and optimizer
            criterion = nn.MSELoss()  # Mean Squared Error for regression
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed

            # Step 6: Split the data into training and testing sets
            # X_train, X_test, y_train, y_test = train_test_split(torch.cat((obs,acs), dim=1), q_values, test_size=0.2, random_state=42)
            X_train = torch.cat((obs,acs), dim=1)
            y_train = q_values
            X_test = torch.cat((obs,acs), dim=1)
            y_test = q_values

        elif task == "model":
            total_variance = torch.norm(next_obs, p='fro')

            model = ptu.build_mlp(
                        obs.size(dim=1) + acs.size(dim=1),
                        next_obs.size(dim=1),
                        n_layers=self.params['n_layers'],
                        size=self.params['size'],
                        activation='relu'
                    )

            # Step 5: Define a loss function and optimizer
            criterion = nn.MSELoss()  # Mean Squared Error for regression
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed

            # Step 6: Split the data into training and testing sets
            # X_train, X_test, y_train, y_test = train_test_split(torch.cat((obs,acs), dim=1), q_values, test_size=0.2, random_state=42)
            X_train = torch.cat((obs,acs), dim=1)
            y_train = next_obs
            X_test = torch.cat((obs,acs), dim=1)
            y_test = next_obs

        else:
            total_variance = torch.norm(rews, p='fro')

            model = ptu.build_mlp(
                        obs.size(dim=1) + acs.size(dim=1),
                        1,
                        n_layers=self.params['n_layers'],
                        size=self.params['size'],
                        activation='relu'
                    )

            # Step 5: Define a loss function and optimizer
            criterion = nn.MSELoss()  # Mean Squared Error for regression
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed

            # Step 6: Split the data into training and testing sets
            # X_train, X_test, y_train, y_test = train_test_split(torch.cat((obs,acs), dim=1), q_values, test_size=0.2, random_state=42)
            X_train = torch.cat((obs,acs), dim=1)
            y_train = rews
            X_test = torch.cat((obs,acs), dim=1)
            y_test = rews

        # Create DataLoader for training and testing
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return model, criterion, optimizer, train_loader, test_loader, total_variance

    def fit_model(self, model, criterion, optimizer, train_loader, test_loader, total_variance):

        run = wandb.init(
                        # Set the project where this run will be logged
                        project="rl-approx-error-" + str(self.params["n_layers"]) + "-" + str(self.params["size"]),
                        # Track hyperparameters and run metadata
                        config={
                            "epochs": self.params['epochs'],
                            "task": self.params['task'],
                            "environment": self.params['rollout_path'],
                            "group": self.params['task'] + '_'+ self.params['rollout_path']
                        })

        epochs = self.params['epochs']

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0  # Initialize total loss for the epoch
            num_batches = len(train_loader)

            # Create a tqdm progress bar for training

            # progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Epoch {epoch + 1}/{epochs}')
            progress_bar = enumerate(train_loader)

            for batch_idx, (inputs, labels) in progress_bar:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # Accumulate the loss for this batch

                # Display the training loss for the current batch
                # progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            # Calculate and print the average training loss for the epoch
            average_loss = total_loss / num_batches

            relative_loss = total_loss / total_variance

            print_period = 10
            if epoch % print_period == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}, Relative Loss: {relative_loss:.4f}")

            if epoch > 1:
                wandb.log({self.params['rollout_path']+'_'+self.params['task']+'_'+"relative_loss": relative_loss, 
                       self.params['rollout_path']+'_'+self.params['task']+'_'+"loss": average_loss})

            # print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}")

        # Rest of the evaluation code remains the same
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        relative_loss = total_loss / total_variance
        print(f"Test Loss: {average_loss:.4f}, Relative Loss: {relative_loss:.4f}")

        wandb.finish

if __name__ == "__main__":
    
    # Step 1: Import the necessary libraries

    # Step 2: Load the data from the .pkl file
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--rollout_path', type=str, default='sac_pendulum')
    parser.add_argument('--task', type=str, default='model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)


    # file_path = 'rollout/rollout_cheetah_3.pkl'  # Replace with your file path
    file_path = 'rollout/'+params['rollout_path'] + '.pkl'

    approx = Approx(params)

    ###################
    ### RUN APPROXIMATION FOR TASK IN REWARD, MODEL, VALUE
    ###################

    print("Run approximation for " + params['task'] + " reward function:")

    model, criterion, optimizer, train_loader, test_loader, total_variance = approx.prepare_data(file_path)
    
    # Step 7: Train the model

    approx.fit_model(model, criterion, optimizer, train_loader, test_loader, total_variance)
    
    # ###################
    # ### RUN APPROXIMATION FOR MODEL
    # ###################

    # print("Run approximation for model function:")

    # model, criterion, optimizer, train_loader, test_loader, total_variance = approx.prepare_data(file_path, "model")
    
    # # Step 7: Train the model

    # approx.fit_model(model, criterion, optimizer, train_loader, test_loader, total_variance)

    # ###################
    # ### RUN APPROXIMATION FOR VALUE
    # ###################

    # print("Run approximation for value function:")

    # model, criterion, optimizer, train_loader, test_loader, total_variance = approx.prepare_data(file_path, "value")
    
    # # Step 7: Train the model

    # approx.fit_model(model, criterion, optimizer, train_loader, test_loader, total_variance)
