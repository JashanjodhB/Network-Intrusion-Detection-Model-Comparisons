import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Defining a custom Dataset class
class NetworkDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        #Converting data to torch tensors
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

#Defining the Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        #Layers= input -> hidden layers with ReLU and Dropout -> output
        super(NeuralNetwork,self).__init__()
        layers=[]
        prev_dim=input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim,dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim=dim
        layers.append(nn.Linear(prev_dim,1))
        self.model=nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class PyTorchModel:
    def __init__(self, input_dim, **hyperparameters):
        #Initializing model parameters and hyperparameters
        self.input_dim=input_dim
        self.hidden_dims=hyperparameters.get('hidden_dims',[64,32])
        self.dropout=hyperparameters.get('dropout',0.3)
        self.learning_rate=hyperparameters.get('learning_rate',0.001)
        self.batch_size=hyperparameters.get('batch_size',32)
        self.epochs=hyperparameters.get('epochs',20)
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            print("Using GPU for training")
        else:
            self.device=torch.device('cpu')
            print("Using CPU for training")
        self.model=NeuralNetwork(self.input_dim,self.hidden_dims,self.dropout).to(self.device)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark=True
        self.criterion=nn.BCEWithLogitsLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.history={
            'loss':[],
            'val_loss':[],
            'accuracy':[],
            'val_accuracy':[]
        }
        print("Model initialized.")
    #Training the model
    def train(self, X_train, y_train, X_val=None, y_val=None):
        #loading data into DataLoader
        training_data=NetworkDataset(X_train,y_train)
        train_loader=DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        if X_val is not None and y_val is not None:
            val_data=NetworkDataset(X_val,y_val)
            val_loader=DataLoader(val_data,batch_size=self.batch_size,shuffle=False)

        #Training loop
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            #calling training mode
            self.model.train()
            #resetting metrics
            running_loss=0.0
            correct=0
            total=0

            #Iterating over batches
            for inputs, labels in train_loader:
                inputs, labels=inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                #Calculating loss and backpropagating
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()*inputs.size(0)
                probs= torch.sigmoid(outputs)
                predicted=(probs > 0.5).float()
                correct+=(predicted==labels).sum().item()
                total+=labels.size(0)


            #Calculating epoch metrics and storing in history
            epoch_loss=running_loss/total
            epoch_acc=correct/total
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc)

            #Validation step
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_running_loss=0.0
                val_correct=0
                val_total=0
                
                with torch.no_grad():
                    #Iterating over validation batches
                    for val_inputs, val_labels in val_loader:
                        val_inputs, val_labels=val_inputs.to(self.device), val_labels.to(self.device).float().unsqueeze(1)
                        val_outputs=self.model(val_inputs)
                        val_loss=self.criterion(val_outputs,val_labels)
                        val_running_loss+=val_loss.item()*val_inputs.size(0)
                        val_predicted=(val_outputs>0.5).float()
                        val_correct+=(val_predicted==val_labels).sum().item()
                        val_total+=val_labels.size(0)
                #Calculating validation metrics and storing in history
                val_epoch_loss=val_running_loss/val_total
                val_epoch_acc=val_correct/val_total
                self.history['val_loss'].append(val_epoch_loss)
                self.history['val_accuracy'].append(val_epoch_acc)
        
        return self.history
    

    #Generating predictions
    def predict(self, X_test):
        #Setting model to evaluation mode
        self.model.eval()
        #Loading test data
        test_data=NetworkDataset(X_test,torch.zeros(len(X_test)))  # Dummy labels
        test_loader=DataLoader(test_data,batch_size=self.batch_size,shuffle=False)
        all_probs=[]

        with torch.no_grad():
            #Iterating over test batches
            for inputs, _ in test_loader:
                inputs=inputs.to(self.device)
                outputs=self.model(inputs)
                all_probs.extend(outputs.cpu().numpy())

        #Converting probabilities to predictions
        all_probs=torch.tensor(all_probs).squeeze()
        predictions=(all_probs>0.5).int().numpy()
        print(f"Generated predictions for {len(X_test)} records")
        return predictions, all_probs.numpy()
    

    #Evaluating the model
    def evaluate(self, X_test, y_test):
        #Setting model to evaluation mode
        self.model.eval()
        test_data=NetworkDataset(X_test,y_test)
        test_loader=DataLoader(test_data,batch_size=self.batch_size,shuffle=False)
        running_loss=0.0
        correct=0
        total=0
        #Iterating over test batches
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels=inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)
                outputs=self.model(inputs)
                loss=self.criterion(outputs,labels)
                running_loss+=loss.item()*inputs.size(0)
                predicted=(outputs>0.5).float()
                correct+=(predicted==labels).sum().item()
                total+=labels.size(0)

        #Calculating overall test metrics
        test_loss=running_loss/total
        test_acc=correct/total
        print(f"Model initialized on {self.device}")
        return{'loss': test_loss, 'accuracy': test_acc}
    

    #saving and loading the model
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {filepath}")
