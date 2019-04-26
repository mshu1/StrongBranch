import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable


num_of_graphs = 10000
num_of_nodes = 20
#graphs = generate_graphs(num_of_nodes, num_of_graphs)
graphs = np.load("graphs_updated.npy")
labels = np.load("labels.npy")

def compress_graphs(graphs):
    """Convert 3d matrix of graphs, dims = (graphs,verts,verts), 
    to 2d matrix dims = (n_graphs, (verts choose 2) - verts). 
    We can omit half of the data as the matrix of each graph is 
    symmetric as we are working with undirected graphs. We can
    also not take any entries on the main diagonal as they will 
    always be zero as we do not allow self loops.
    
    
    Args: 
        graphs : (n_graphs,verts,verts) hypermatrix of graphs.
        
    Returns:
        graphs_comp : (n_graphs, (vert choose 2) - verts) matrix
            of compressed adjancey matrices.
        
        
    """
    
    verts = graphs.shape[1]
    graphs_comp =  graphs.T[np.triu_indices(verts, 1)].T
    
    return graphs_comp	

# compress 2d adjacency matrix to 1d array, prepare input
compressed_graph = np.zeros((graphs.shape[0], 190))
for i in range(0,graphs.shape[0]):
    compressed_graph[i] = compress_graphs(graphs[i])

graphs_train, graphs_test, train_labels, val_labels =\
	train_test_split(compressed_graph, labels, test_size=0.20, random_state=42)

def batch(batch_size, training=True):
    """Create a batch of examples.
  
    This creates a batch of input graphs and a batch of corresponding
    ground-truth labels. We assume CUDA is available (with a GPU).
  
    Args:
        batch_size: An integer.
        training: A boolean. If True, grab examples from the training
        set; otherwise, grab them from the validation set.
    Returns:
        A tuple,
        input_batch: A Variable of floats with shape
        [batch_size, 1, height, width]
        label_batch: A Variable of ints with shape
        [batch_size].
    """
    if training:
        random_ind = np.random.choice(graphs_train.shape[0], size=batch_size, replace=False)
        input_batch = graphs_train[random_ind]
        label_batch = train_labels[random_ind]
    else:
        input_batch = graphs_test[:batch_size]
        label_batch = val_labels[:batch_size]
 
  
    volatile = not training
    if volatile:
        with torch.no_grad():
            if torch.cuda.is_available():
                input_batch = Variable(torch.from_numpy(input_batch).cuda())
                label_batch = Variable(torch.from_numpy(label_batch).cuda())
            else:
                input_batch = Variable(torch.from_numpy(input_batch))
                label_batch = Variable(torch.from_numpy(label_batch))
    else:
        if torch.cuda.is_available():
            input_batch = Variable(torch.from_numpy(input_batch).cuda())
            label_batch = Variable(torch.from_numpy(label_batch).cuda())
        else:
            input_batch = Variable(torch.from_numpy(input_batch))
            label_batch = Variable(torch.from_numpy(label_batch))
        

    return input_batch, label_batch

def train_step(batch_size=50):
 
    model.train()
    correct_count, total_loss, total_acc = 0., 0., 0.
    
    input_batch, label_batch = batch(batch_size, training=True)

    label_batch = label_batch.long()
    input_batch = input_batch.float()

    output_batch = model(input_batch)
    print(output_batch[0].shape)
    loss = F.cross_entropy(output_batch[0], label_batch[:,0]) \
        + F.cross_entropy(output_batch[1], label_batch[:,1])
    
    pred1 = output_batch[0].data.max(1)[1]
    pred2 = output_batch[1].data.max(1)[1]
    print(pred1.shape)
    print(pred1)
    print(output_batch[0])
    matches = (label_batch[:,0] == pred1) & (label_batch[:,1] == pred2)
    accuracy = matches.float().mean()
    correct_count += matches.sum()
    
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
    #print(pred1)
    return loss.data.item(), accuracy

def val():
    
    model.eval()
    input_batch, label_batch = batch(graphs_test.shape[0], training=False)
    
    label_batch = label_batch.long()
    input_batch = input_batch.float()
    
    output_batch = model(input_batch)
    
    
    loss = F.cross_entropy(output_batch[0], label_batch[:,0]) \
        + F.cross_entropy(output_batch[1], label_batch[:,1])
    
    pred1 = output_batch[0].data.max(1)[1]
    pred2 = output_batch[1].data.max(1)[1]
    
    matches = (label_batch[:,0] == pred1) & (label_batch[:,1] == pred2)
    accuracy = matches.float().mean()
    
    return loss.data.item(), accuracy    

class CompressMatrixNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(190, 570)
        self.fc2 = nn.Linear(570, 380)
        self.fc3 = nn.Linear(380, 85)
        self.fc4 = nn.Linear(85, 40)
        self.fc5 = nn.Linear(85, 40)
        self.fc6 = nn.Linear(40, 20)
        self.fc7 = nn.Linear(40, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x1 = F.relu(self.fc4(x))
        x1 = self.fc6(x1)
        x2 = F.relu(self.fc5(x))
        x2 = self.fc7(x2)
        return F.softmax(x1, dim=1), F.softmax(x2, dim=1)


model = CompressMatrixNetwork()
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer.zero_grad()

for module in model.children():
    module.reset_parameters()

info = []
fig, ax = plt.subplots(2, 1, sharex=True)
num_steps = 10000
num_steps_per_val = 50
best_val_acc = 0.0
best_train_acc = 0.0
for step in range(num_steps):
    train_loss, train_acc = train_step()
    if step % num_steps_per_val == 0:
        val_loss, val_acc = val()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print('Step {:5d}: Obtained a best validation acc of {:.3f}.'.format(step, best_val_acc))
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        print('Step {:5d}: Obtained a best training acc of {:.3f}.'.format(step, best_train_acc))
    info.append([step, train_acc, val_loss, train_acc, val_acc])
    x, y11, y12, y21, y22 = zip(*info)
    ax[0].plot(x, y11, x, y12)
    ax[0].legend(['Train loss', 'Val loss'])
    ax[1].plot(x, y21, x, y22)
    ax[1].legend(['Train acc', 'Val acc'])
    ax[1].set_ylim([0.0, 0.25])       