import os
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import models as models
import pickle
import time

cwd = os.getcwd()

#Device to be used in training. If GPU is available, it will be automatically used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load Training + Test Data. The data is generated using Latin Hypercube Sampling (LHS)
training_data = scipy.io.loadmat('RL_solenoid_data.mat')
train_test_y = training_data['RL_solenoid_data']
train_test_y = train_test_y.squeeze()
train_test_x = training_data['trainingInputs']

#Select how many samples to be used for training.
Ndata = train_test_y.shape[0]
Ntrain = 800
Ntest = Ndata-Ntrain

#get train/test indices by linspace to avoid disrupting LHS
train_indices = np.linspace(0, Ndata-1, num=Ntrain).round().astype(int)
dumm = np.arange(0, Ndata)
test_idx = np.delete(dumm, train_indices)

training_x = train_test_x[train_indices, :]
test_x = train_test_x[test_idx, :]

training_y = train_test_y[train_indices, :, :]
test_y = train_test_y[test_idx, :, :]

# Get normalization values using Training Data.
meanY = training_y.mean(axis=0)
stdY = training_y.std(axis=0)

MeanY = torch.tensor(meanY).to(device)
StdY = torch.tensor(stdY).to(device)

# Inputs are scaled between [-1, 1].
meanX = train_test_x.min(axis=0)
stdX = train_test_x.max(axis=0) - train_test_x.min(axis=0)

#Normalize Train and Test Data
#Use mean and std of train data to normalize test data to avoid bias
training_x = -1+2*(training_x-meanX)/stdX
training_y = (training_y-meanY)/stdY

test_x = -1+2*(test_x - meanX)/stdX
test_y = (test_y - meanY)/stdY

dimIn = training_x.shape[1]

train_test_y_normalized = (train_test_y-meanY)/stdY
train_test_x_normalized = -1+2*(train_test_x-meanX)/stdX

# Convert everything to tensor
tensor_y = torch.Tensor(training_y).to(device)
tensor_x = torch.Tensor(training_x).to(device)

tensor_test_y = torch.Tensor(test_y).to(device)
tensor_test_x = torch.Tensor(test_x).to(device)

print('Done loading data and pre-processing.')


NMSE_Test = []
modelWeights= []
NMSE_Test_Median = []
#Select model. 2 different STCNN types are provided in "models.py". (Un)comment below to choose.
# model = models.Solenoid_STCNN().to(device)
model = models.Solenoid_STCNN_V2().to(device)

#Select optimizer to be used, define initial "learning rate (lr)", and learning rate reduction ratio (gamma) at milestones.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [300, 500, 1000, 2500, 3500], gamma=0.5)
numParams = sum([p.numel() for p in model.parameters()])

print(f"Model is loaded. Number of learnable parameters in the network {numParams:d}")


#Training Loss Function. This is different than MSE loss, see the paper for details.
def calc_training_loss(recon_y, y):
    err = (torch.abs(y-recon_y)**2).sum(dim=2)/recon_y.shape[2]
    return err.sqrt().mean()

#Calculate Test Accuracy. For each response in the data (L(f), R(f)), calculate NMSE (see paper for details).
def calc_test_NMSE(recon_y, y):
    err = (torch.abs(y-recon_y)**2).sum(dim=2)
    mean_ref = y.mean(dim=2)
    mean_ref = mean_ref.unsqueeze(-1).repeat(1, 1, y.shape[2])
    norm = (torch.abs(y-mean_ref)**2).squeeze().sum(dim=2)
    NMSE = err/norm
    return NMSE

#Closure to be called by optimizer during training.
def closure(data_x, data_y):
    optimizer.zero_grad()
    output = model(data_x)
    loss = calc_training_loss(output, data_y)
    return loss


#Man Training Loop.
#Results are printed at every "test_schedule" epochs.
test_schedule = 5
training_iter = 1000
current_time = time.time()
print(f"Starting training the model. \n")
print(f"""-----------------------------------------------------------------""")

for a in range(training_iter):
    model.train()
    train_data_x = tensor_x
    train_data_y = tensor_y
    loss = closure(train_data_x, train_data_y)

    loss.backward()

    optimizer.step()
    scheduler.step()

    # Set into eval mode for testing.
    if a % test_schedule == 0:
        model.eval()
        with torch.no_grad():
            test_y = tensor_test_y
            test_x = tensor_test_x
            test_output = model(test_x)

            NMSE = calc_test_NMSE(test_output, test_y)
            avNMSE = NMSE.mean()
            medNMSE = NMSE.median()

            print(f"Train Iter {(a+1):d}/{training_iter:d} - Train Loss: {loss.item():.3f} --> "
                  f"Test Av. NMSE: {avNMSE.item():.3f}, Med. NMSE: {medNMSE.item():.3f}")

            #save model weights at each iteration. The best model will be chosen after training is finished.
            modelWeights += [model.state_dict()]
            #save test NMSE (both average and median)
            NMSE_Test += [avNMSE.item()]
            NMSE_Test_Median += [medNMSE.item()]

elapsed = time.time() - current_time
print(f"""\n-----------------------------------------------------------------""")
print(f"""Training is completed in {elapsed/60 :.3f} minutes""")
NMSE_Test = np.asarray(NMSE_Test)
best_idx = np.argmin(NMSE_Test)
final_model_weights = modelWeights[best_idx]

#The weights of the final model will be saved under the name "save_name".
save_name = "STCNN_Solenoid.pth"
print(f"Final Model Performance on Test Set -> Average NMSE: {100*NMSE_Test[best_idx]:.2f}%, Median NMSE: {100*NMSE_Test_Median[best_idx]:.2f}%.")
torch.save(final_model_weights, cwd + "/" + save_name)
print(f"Model weights are saved in \"{cwd}\\{save_name}\"")
print(f"""-----------------------------------------------------------------""")
print("Exiting.")

