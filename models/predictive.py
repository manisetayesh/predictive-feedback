# Import TorchSeq2PC
import TorchSeq2PC as T2PC
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Seed rng
torch.manual_seed(0)

from torchvision.datasets import MNIST

# Construct the absolute path to the data directory
# This makes the path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

# Get training data structure
train_dataset = MNIST(root=data_dir, 
                      train=True, 
                      transform=torchvision.transforms.ToTensor(),  
                      download=False)

# Number of trainin data points
m = len(train_dataset)

# Print the size of the training data set
print('\n\n\n')
print("Number of data points in training set = ",m)
print("Size of training inputs (X)=",train_dataset.data.size())
print("Size of training labels (Y)=",train_dataset.targets.size())

# Define batch size
batch_size = 300      # Batch size to use with training data

# Create data loader. 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)


# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ',device)

# Define the nunmber of epochs, learning rate, 
# and how often to print progress
num_epochs=2
LearningRate=0.0001
PrintEvery=50

# Choose an optimizer
WhichOptimizer=torch.optim.Adam

# Compute size of each batch
steps_per_epoch = len(train_loader) 
total_num_steps  = num_epochs*steps_per_epoch
print("steps per epoch (mini batch size)=",steps_per_epoch)





model=nn.Sequential(
    
    nn.Sequential(nn.Conv2d(1,10,3),
    nn.ReLU(),
    nn.MaxPool2d(2)
    ),

    nn.Sequential(
    nn.Conv2d(10,5,3),
    nn.ReLU(),
    nn.Flatten()
    ),

 nn.Sequential(    
    nn.Linear(5*11*11,50),
    nn.ReLU()
    ),

 nn.Sequential(    
    nn.Linear(50,30),
    nn.ReLU()
    ),


nn.Sequential(
   nn.Linear(30,10)
 )

).to(device)

# Define the loss function
LossFun = nn.CrossEntropyLoss()

# Compute one batch of output and loss to make sure
# things are working
with torch.no_grad():
  TrainingIterator=iter(train_loader)
  X,Y=next(TrainingIterator)  
  X=X.to(device)
  Y=Y.to(device)
  Yhat=model(X).to(device)
  print('output shape = ',Yhat.shape)
  print('loss on initial model = ',LossFun(Yhat,Y).item())


NumParams=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of trainable parameters in model =',NumParams)




ErrType="Strict"
eta=.1
n=20



optimizer = WhichOptimizer(model.parameters(), lr=LearningRate)

# Initialize vector to store losses
LossesToPlot=np.zeros(total_num_steps)


j=0     # Counters
jj=0    
t1=time.time() # Get start time
for k in range(num_epochs):

  # Re-initialize the training iterator (shuffles data for one epoch)
  TrainingIterator=iter(train_loader)
  
  for i in range(steps_per_epoch): # For each batch

    # Get one batch of training data, reshape it
    # and send it to the current device        
    X,Y=next(TrainingIterator)  
    X=X.to(device)
    Y=Y.to(device)

    # Perform inference on this batch
    vhat,Loss,dLdy,v,epsilon=T2PC.PCInfer(model,LossFun,X,Y,ErrType,eta,n)
    
    # Update parameters    
    optimizer.step() 

    # Zero-out gradients     
    optimizer.zero_grad()

    # Print and store loss
    #with torch.no_grad():
    if(i%PrintEvery==0):
      print('epoch =',k,'step =',i,'Loss =',Loss.item())
    LossesToPlot[jj]=Loss.item() 
    jj+=1

# Compute and print time spent training
tTrain=time.time()-t1
print('Training time = ',tTrain,'sec')

# Plot the loss curve
plt.figure()
plt.plot(LossesToPlot)
plt.ylim(bottom=0)  
plt.ylabel('training loss')
plt.xlabel('iteration number')
