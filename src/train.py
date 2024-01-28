# import os
import pickle
# import sys
# import csv
import statistics
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import yaml
from dvclive import Live

class IrisDatatset(Dataset):

    def __init__(self,file_name):
        price_df=pd.read_csv(file_name)

        x=price_df.iloc[:,0:5].values
        y=price_df.iloc[:,5].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

def train(model, train_loader, optimizer, criterion, n_epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        losses = []
        accs = []
        for batch in train_loader:
            X_batch = batch[0]
            y_batch = batch[1]
            optimizer.zero_grad()
            pred = model(X_batch.to(device))
            loss = criterion(pred, y_batch.to(torch.long).to(device))
            acc = accuracy_score(y_batch.to(device),torch.argmax(pred,1))
            losses.append(float(loss))
            accs.append(float(acc))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}: train loss = {round(statistics.mean(losses),3)}, train acc = {round(statistics.mean(accs),3)}')
    return model




def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    opt_name = params["optimizer"]
    learning_rate = params["lr"]
    n_epochs = params["epochs"]


    model = nn.Sequential(
        nn.Linear(5,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32 ,3),
    )

    if opt_name.upper == "ADAM":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    else:
        opt = torch.optim.SGD(model.parameters(),lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    myDs = IrisDatatset('data/prepared/train.csv')
    train_loader = DataLoader(myDs,batch_size=16,shuffle=False)

    model_tr = train(model, train_loader, opt, criterion, n_epochs)

    with open("model.pkl", "wb") as f:
        pickle.dump(model_tr, f)

if __name__ == "__main__":
    main()


