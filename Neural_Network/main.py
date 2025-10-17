import torch
import torch.nn as nn
import torch.nn.functional as F 

class Model(nn.Module):
    # we have two layers and output node
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # initiate our nn model
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x 

torch.manual_seed(43)
model = Model() # create an instance of model

# Load dataset
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
iris = load_iris()
# data frame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df)
#Define variables
X = df.drop(['target'], axis=1)
y = df['target']

#Convert to numpy arrays
X = X.values
y = y.values

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

#Convert to Tensors. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters. Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators.
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far off the predictions are from data
criterion = nn.CrossEntropyLoss()

#Choose ADAM optimizer, lr = learning rate
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

#Train our model by choosing the numbe rof epochs (iterations)
epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train) #get predicted results
    # Measure loss
    loss = criterion(y_pred, y_train)
    
    # Keep track of our losses
    losses.append(loss.detach().numpy())
    
    # Print every 10 epochs
    if i % 10 == 0:
        print(f"Epoch: {i} and the loss: {loss}")

    # Do some backpropogation to fine neurons' weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    

# Plot epochs
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")
plt.show()


# Evaluate model
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        if y_test[i] == 0:
            x = 'Setosa'
        elif y_test[i] == 1:
            x = 'Versicolor'
        else:
            x = 'Virginica'
            
        #Will tell us what type of flowe class our network think it is
        print(f"{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}")
        
        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct +=1

print(f"We got {correct} correct")
        
# Do the predictios
new_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])
with torch.no_grad():
    print(model(new_iris))
    
# Save our NN model
torch.save(model.state_dict(), 'iris_easy_NN_model.pt')

new_model = Model()
new_model.load_state_dict(torch.load('iris_easy_NN_model.pt'))

print(new_model.eval())
