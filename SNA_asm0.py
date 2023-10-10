# -*- coding: utf-8 -*-
import pandas as pd
from pandas import read_csv
import networkx as nx
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt

#Import Data
names = ['userA','userB'] #custom column name
medium = pd.read_csv(r'E:\SNA\medium.tsv', sep='\t', header=None, names=names)
large = pd.read_csv(r'E:\SNA\large.tsv', sep='\t', header=None, names=names)

#Q2.1
#Calculate the number of the directed links for the network in medium.tsv
G1 = nx.DiGraph()
edges1 = medium[['userA','userB']].values.tolist()
G1.add_edges_from(edges1)
num_edges1 = G1.number_of_edges()

#Calculate the number of the directed links for the network in large.tsv
G2 = nx.DiGraph()
edges2 = large[['userA','userB']].values.tolist()
G2.add_edges_from(edges2)
num_edges2 = G2.number_of_edges()
print(num_edges1,num_edges2)

#Q2.2
#Calculate the number of the nodes for the network in medium.tsv and large.tsv
num_nodes1 = G1.number_of_nodes()
num_nodes2 = G2.number_of_nodes()
print(num_nodes1,num_nodes2)

#Q2.3
#Draw the indegree and outdegree distributions
Indegree1 = G1.in_degree()
Indegree2 = G2.in_degree()
Outdegree1 = G1.out_degree()
Outdegree2 = G2.out_degree()

in_degrees1 = dict(Indegree1)
indegree_values1 = list(in_degrees1.values())
in_degrees2 = dict(Indegree2)
indegree_values2 = list(in_degrees2.values())
out_degrees1 = dict(Outdegree1)
outdegree_values1 = list(out_degrees1.values())
out_degrees2 = dict(Outdegree2)
outdegree_values2 = list(out_degrees2.values())

rcParams['axes.titlepad'] = 20 
fig = plt.figure(figsize=(12,6), dpi=150)
plt.hist(indegree_values1, bins=50, facecolor="steelblue", edgecolor="black", 
         range=(0,50),alpha=0.6)
plt.xlabel("Indegree",fontsize=26)
plt.ylabel("Frequency",fontsize=26)
plt.tick_params(axis='x', labelsize=26) 
plt.tick_params(axis='y', labelsize=26)
plt.title("Indegree distribution for the network in medium.tsv",fontsize=28)
plt.show()

rcParams['axes.titlepad'] = 20 
fig = plt.figure(figsize=(12,6), dpi=150)
plt.hist(indegree_values2, bins=50, facecolor="steelblue", edgecolor="black", 
         range=(0,50),alpha=0.6)
plt.xlabel("Indegree",fontsize=26)
plt.ylabel("Frequency",fontsize=26)
plt.tick_params(axis='x', labelsize=26) 
plt.tick_params(axis='y', labelsize=26)
plt.title("Indegree distribution for the network in large.tsv",fontsize=28)
plt.show()

rcParams['axes.titlepad'] = 20 
fig = plt.figure(figsize=(12,6), dpi=150)
plt.hist(outdegree_values1, bins=50, facecolor="steelblue", edgecolor="black", 
         range=(0,50),alpha=0.6)
plt.xlabel("Outdegree",fontsize=26)
plt.ylabel("Frequency",fontsize=26)
plt.tick_params(axis='x', labelsize=26) 
plt.tick_params(axis='y', labelsize=26)
plt.title("Outdegree distribution for the network in medium.tsv",fontsize=28)
plt.show()

rcParams['axes.titlepad'] = 20 
fig = plt.figure(figsize=(12,6), dpi=150)
plt.hist(outdegree_values2, bins=50, facecolor="steelblue", edgecolor="black", 
         range=(0,50),alpha=0.6)
plt.xlabel("Outdegree",fontsize=26)
plt.ylabel("Frequency",fontsize=26)
plt.tick_params(axis='x', labelsize=26) 
plt.tick_params(axis='y', labelsize=26)
plt.title("Outdegree distribution for the network in large.tsv",fontsize=28)
plt.show()

#Q2.4 
#Compute weakly connected components and strongly connected components
weakly_connected_components1 = nx.number_weakly_connected_components(G1)
print(weakly_connected_components1)
weakly_connected_components2 = nx.number_weakly_connected_components(G2)
print(weakly_connected_components2)
strongly_connected_components1 = nx.number_strongly_connected_components(G1)
print(strongly_connected_components1)
strongly_connected_components2 = nx.number_strongly_connected_components(G2)
print(strongly_connected_components2)

# Calculate the number of nodes and links of the largest strongly and largest weakly connected component
largest_weakly_connected_component1 = max(nx.weakly_connected_components(G1), key=len)
largest_weakly_connected_component2 = max(nx.weakly_connected_components(G2), key=len)
largest_strongly_connected_component1 = max(nx.strongly_connected_components(G1), key=len)
largest_strongly_connected_component2 = max(nx.strongly_connected_components(G2), key=len)
num_nodes_in_largest_weakly_connected1 = len(largest_weakly_connected_component1)
num_edges_in_largest_weakly_connected1 = G1.subgraph(largest_weakly_connected_component1).number_of_edges()
num_nodes_in_largest_strongly_connected1 = len(largest_strongly_connected_component1)
num_edges_in_largest_strongly_connected1 = G1.subgraph(largest_strongly_connected_component1).number_of_edges()

num_nodes_in_largest_weakly_connected2 = len(largest_weakly_connected_component2)
num_edges_in_largest_weakly_connected2 = G2.subgraph(largest_weakly_connected_component2).number_of_edges()
num_nodes_in_largest_strongly_connected2 = len(largest_strongly_connected_component2)
num_edges_in_largest_strongly_connected2 = G2.subgraph(largest_strongly_connected_component2).number_of_edges()
print(num_nodes_in_largest_weakly_connected1,num_edges_in_largest_weakly_connected1,
      num_nodes_in_largest_strongly_connected1,num_edges_in_largest_strongly_connected1)
print(num_nodes_in_largest_weakly_connected2,num_edges_in_largest_weakly_connected2,
      num_nodes_in_largest_strongly_connected2,num_edges_in_largest_strongly_connected2)


#Q2.5
#Compute the average clustering coefficient
Coefficient1 = nx.average_clustering(G1)
Coefficient2 = nx.average_clustering(G2)
print(Coefficient1,Coefficient2)

#Q2.6
#Compute The distance distribution of the maximum weakly connected component
largest_wcc = max(nx.weakly_connected_components(G2), key=len)

distance_distribution = {}
for source_node in largest_wcc:
    shortest_paths = nx.single_source_shortest_path_length(G2, source_node)
    for target_node, distance in shortest_paths.items():
        if target_node != source_node:
            if distance in distance_distribution:
                distance_distribution[distance] += 1
            else:
                distance_distribution[distance] = 1

distances = list(distance_distribution.keys())
frequencies = [distance_distribution[distance] for distance in distances]

rcParams['axes.titlepad'] = 20 
fig = plt.figure(figsize=(12,6), dpi=150)
plt.bar(distances*0.91, frequencies*0.93)
plt.xlabel("Distance",fontsize=26)
plt.ylabel("Frequency",fontsize=26)
plt.tick_params(axis='x', labelsize=26) 
plt.tick_params(axis='y', labelsize=26)
plt.title('The distance distribution of the maximum weakly \n connected component for the network in large.tsv',fontsize=28)
plt.show()

largest_wcc = max(nx.weakly_connected_components(G1), key=len)
lwcc_subgraph = G1.subgraph(largest_wcc)

# 计算所有节点对的最短路径长度并求平均值
total_distance = 0
pair_count = 0

for node1 in lwcc_subgraph.nodes():
    for node2 in lwcc_subgraph.nodes():
        if node1 != node2:
            shortest_path_length = nx.shortest_path_length(lwcc_subgraph, source=node1, target=node2)
            total_distance += shortest_path_length
            pair_count += 1

average_distance = total_distance / pair_count

print("最大弱连通分量中任意两个节点对的平均距离：", average_distance)