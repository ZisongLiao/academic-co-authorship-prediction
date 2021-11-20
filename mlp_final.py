import torch
import torch.nn as nn
import torch.nn.functional as F

# multilayer perceptron model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sig(out)

        return out


import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors
from node2vec.edges import HadamardEmbedder


filepath = 'train.txt'

# read the dataset text file and return a dictionary for all unique edges
def read_text(filepath):
    author_edge = defaultdict(list)
    #two_way = defaultdict(list)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for ln in lines:
            authors = ln.split()
            i = 0
            while i < len(authors):
                j = i + 1
                while j < len(authors):
                    if authors[i] != authors[j]:
                        if authors[j] not in author_edge[authors[i]]:
                            author_edge[authors[i]].append(authors[j])
                        '''
                        if authors[j] not in two_way[authors[i]]:
                            two_way[authors[i]].append(authors[j])
                        if authors[i] not in two_way[authors[j]]:
                            two_way[authors[j]].append(authors[i])
                        '''
                    j += 1
                i += 1

    return author_edge

# turn edge dictionary into a dataframe and two author lists
def dict_to_list(author_dict):
    author_df = pd.DataFrame()
    author_1 = []
    author_2 = []

    for key, item in author_dict.items():
        for i in item:
            author_1.append(int(key))
            author_2.append(int(i))

    author_df['a1'] = author_1
    author_df['a2'] = author_2

    return author_df, author_1, author_2

# get weighted edge in the form (author1, author2, weight)
def get_weighted_edge(filepath):
    a1 = []
    a2 = []

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for ln in lines:
            authors = ln.split()
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    a1.append(authors[i])
                    a2.append(authors[j])
    wedge = list(zip(a1, a2))
    wedge = Counter(frozenset(edge) for edge in wedge)
    weight_edge = []
    for i in wedge.keys():
        ls = list(i)
        ls.append(wedge[i])
        weight_edge.append(ls)
    weight_edge = [w for w in weight_edge if len(w) == 3]

    return weight_edge

# unconnected edges between authors
def get_unconnected_author(adj_matrix, author_list):
    unconnected = []
    i = 0
    while i < adj_matrix.shape[0]:
        j = 0
        while j < i:
            if adj_matrix.item((i, j)) == 0:
                unconnected.append((author_list[i], author_list[j]))

            j += 1
        i += 1
    return unconnected


author_dict = read_text(filepath)
author_df, author1, author2 = dict_to_list(author_dict)
author_df['link'] = 1
print('number of unique edges', author_df.shape[0])

author_list = author1 + author2
# all unique authors
author_list = list(dict.fromkeys(author_list))
author_list.sort()
# missing authors from the dataset
missing_node = list(set(range(author_list[-1]+1))-set(author_list))

author_graph = nx.convert_matrix.from_pandas_edgelist(author_df, 'a1', 'a2')
adj_matrix = nx.to_numpy_matrix(author_graph, nodelist=author_list)
# all unconnected authors edges
unconnected_list = get_unconnected_author(adj_matrix, author_list)

a1_uc = [i[0] for i in unconnected_list]
a2_uc = [i[1] for i in unconnected_list]
uc_df = pd.DataFrame({'a1': a1_uc, 'a2': a2_uc})
uc_df['link'] = 0
print('number of negative edges', uc_df.shape[0])
# weighted graph
author_graph_weighted = nx.Graph()
weighted_edge = get_weighted_edge(filepath)
author_graph_weighted.add_weighted_edges_from(weighted_edge)
author_graph_weighted.add_nodes_from(missing_node)

# node2vec model
n2v = Node2Vec(author_graph_weighted, dimensions=128, walk_length=100, num_walks=200, workers=multiprocessing.cpu_count()-1)
n2v_model = n2v.fit(window=10, min_count=1, batch_words=4)
#n2v_model.save('n2v.model')
#n2v_model = Word2Vec.load('n2v.model')
# hadamard embedding
edges_embs = HadamardEmbedder(keyed_vectors=n2v_model.wv)

# iteration for sampling unconnected authors
uc_remain = uc_df
uc_selections = []
iter_num = 1  # can be between 1 - 400
for i in range(iter_num):
    uc_remain, uc_select = train_test_split(uc_remain, test_size=author_df.shape[0])
    uc_selections.append(uc_select)

# fit them n times and get n results
test_file = 'test-public.csv'
test = pd.read_csv(test_file)
results = pd.DataFrame()
for iteration in tqdm(range(iter_num)):
    data = pd.concat([author_df, uc_selections[iteration]])
    #x = [(n2v_model.wv[str(i)] + n2v_model.wv[str(j)]) for i, j in zip(data['a1'], data['a2'])]
    x = [edges_embs[(str(i), str(j))] for i, j in zip(data['a1'], data['a2'])]
    y = data['link'].to_list()
    #xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), np.array(y), test_size=0.3, stratify=data['link'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train_torch = torch.Tensor(x).cuda()
    #x_test_torch = torch.Tensor(xtest).cuda()
    y_train_torch = torch.Tensor(y).cuda()
    #y_test_torch = torch.Tensor(ytest).cuda()
    batch_size = 200
    train_dataset = TensorDataset(x_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataset = TensorDataset(x_test_torch, y_test_torch)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # MLP model
    model = None
    model = MLP(128, 100)
    model = model.cuda()

    print(device)

    # Loss and optimizer
    criterion = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_step = len(train_loader)
    epochs = 50
    for epoch in range(epochs):
        for i, (fts, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(fts)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # progress update
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, num_step, loss.item()))

    # predict test dataset
    model.eval()
    t_x = [edges_embs[(str(i), str(j))] for i, j in
           zip(test['Source'], test['Sink'])]
    tx_torch = torch.Tensor(t_x)
    with torch.no_grad():
        pred = model(tx_torch.to(device))
    pred = pred.detach().cpu().numpy()
    pred = np.concatenate(pred)
    results[str(iteration)] = pred

    '''
    model.eval()
    with torch.no_grad():
        pred = model(x_test_torch)
    pred = pred.detach().cpu().numpy()
    pred = np.concatenate(pred)
    print(pred)
    print(roc_auc_score(ytest, pred))
    '''

# take the average of n iterations
print(results)
results['average'] = results.mean(axis=1)
pred = results['average'].to_list()
print(pred)

# write to csv
index = list(range(1, len(pred) + 1))
pred_data = pd.DataFrame()
pred_data['id'] = index
pred_data['Predicted'] = pred
pred_data.to_csv('mlp.csv', index=False)
