import csv
import networkx as nx
import numpy as np
from random import randint
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gensim.models.doc2vec import Doc2Vec

# Create a graph
G = nx.read_edgelist('data/initial_data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

# Read the abstract of each paper
abstracts = dict()
with open('data/initial_data/abstracts.txt', 'r') as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = abstract

# Read the authors to each paper
authors = dict()
with open('data/processed_data/authors_ids.txt', 'r') as f:
    for line in f:
        node, node_authors = line.rstrip('\n').split('|--|')
        authors[int(node)] = node_authors.split(',')

# Read the Doc2vec model
doc2vec_model = Doc2Vec.load('data/models/doc2vec_dm_64.model')

# Map text to set of terms
for node in abstracts:
    abstracts[node] = set(abstracts[node].split())

# Read test data. Each sample is a pair of nodes
node_pairs = list()
with open('data/processed_data/non_edges.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))

# Create the training matrix. Each row corresponds to a pair of nodes and
# its class label is 1 if it corresponds to an edge and 0, otherwise.
# Use the following 5 features for each pair of nodes:
# (1) sum of degrees of two nodes
# (2) absolute value of difference of degrees of two nodes
# (1) sum of number of unique terms of the two nodes' abstracts
# (2) absolute value of difference of number of unique terms of the two nodes' abstracts
# (3) number of common terms between the abstracts of the two nodes

X_train = np.zeros((2*m, 7))
y_train = np.zeros(2*m)
for i,edge in enumerate(G.edges()):
    # an edge
    X_train[2*i,0] = G.degree(edge[0]) + G.degree(edge[1])
    X_train[2*i,1] = abs(G.degree(edge[0]) - G.degree(edge[1]))
    X_train[2*i,2] = len(abstracts[edge[0]]) + len(abstracts[edge[1]])
    X_train[2*i,3] = abs(len(abstracts[edge[0]]) - len(abstracts[edge[1]]))
    X_train[2*i,4] = len(abstracts[edge[0]].intersection(abstracts[edge[1]]))
    X_train[2*i,5] = doc2vec_model.docvecs.similarity(edge[0], edge[1])
    X_train[2*i,6] = len(set(authors[edge[0]]) & set(authors[edge[1]]))
    y_train[2*i] = 1

    # a randomly generated pair of nodes
    n1 = node_pairs[i][0]
    n2 = node_pairs[i][1]
    X_train[2*i+1,0] = G.degree(n1) + G.degree(n2)
    X_train[2*i+1,1] = abs(G.degree(n1) - G.degree(n2))
    X_train[2*i+1,2] = len(abstracts[n1]) + len(abstracts[n2])
    X_train[2*i+1,3] = abs(len(abstracts[n1]) - len(abstracts[n2]))
    X_train[2*i+1,4] = len(abstracts[n1].intersection(abstracts[n2]))
    X_train[2*i+1,5] = doc2vec_model.docvecs.similarity(n1, n2)
    X_train[2*i+1,6] = len(set(authors[n1]) & set(authors[n2]))
    y_train[2*i+1] = 0

print('Size of training matrix:', X_train.shape)

# Read test data. Each sample is a pair of nodes
node_pairs = list()
with open('data/initial_data/test.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))

# Create the test matrix. Use the same 2 features as above
X_test = np.zeros((len(node_pairs), 7))
for i,node_pair in enumerate(node_pairs):
    X_test[i,0] = G.degree(node_pair[0]) + G.degree(node_pair[1])
    X_test[i,1] = abs(G.degree(node_pair[0]) - G.degree(node_pair[1]))
    X_test[i,2] = len(abstracts[node_pair[0]]) + len(abstracts[node_pair[1]])
    X_test[i,3] = abs(len(abstracts[node_pair[0]]) - len(abstracts[node_pair[1]]))
    X_test[i,4] = len(abstracts[node_pair[0]].intersection(abstracts[node_pair[1]]))
    X_test[i,5] = doc2vec_model.docvecs.similarity(node_pair[0], node_pair[1])
    X_test[i,6] = len(set(authors[node_pair[0]]) & set(authors[node_pair[1]]))

print('Size of test matrix:', X_test.shape)

# Use logistic regression to predict if two nodes are linked by an edge
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_pred = y_pred[:,1]

# Write predictions to a file
predictions = zip(range(len(y_pred)), y_pred)
with open("submission.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row) 

