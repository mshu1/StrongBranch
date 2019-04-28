import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_of_graphs = 50000
num_of_nodes = 20


def connected_graph(verts): 
    """ Create_connected graph with specified number
    of vertices. Random number of edges.
    
    Args:
        verts: number of vertices.
    Returns:
        graphs: connected networkx graph on verts vertices.
    """
    edge_l = verts - 1
    edge_h = (verts*(verts-1))//2 # verts choose 2
    
    connected = False
    while connected == False:
        edges = np.random.randint(edge_l, edge_h + 1)
        graph = nx.gnm_random_graph(verts,edges)
        connected = nx.is_connected(graph)
       
    return graph


def generate_graphs(verts, n_graphs, edges = None) :
    """Generate connected graphs on n vertices. 
    Sample without replacement. 
    
    Note: this may never finish if user asks for more graphs
    then number of possible connected graphs on n vertices.
    See: http://garsia.math.yorku.ca/~zabrocki/math3260w03/nall.html
    for idea of the magnitude of this number.
    
    Args:
        verts: number of vertices in graph
        n_graphs: number of graphs to make
        edges: number of edges in each graph.
            must be in range (n-1) to (n choose 2).
            If `none` then will sample from all possible.
    
    Returns:
        graphs: [graphs, verts, verts] hypermatrix where
            graphs[i] is ith adjacency matrix.
        OR
        graphs_obj - an array of graph obejcts
            graphs_obj[i] is the ith graph object.
        
    """
    
    graphs_done = 0
    G = nx.Graph() # dummy variable, just need type(G)
    graphs = np.zeros((n_graphs,verts,verts), dtype = int)
    graphs_obj = np.empty(n_graphs, dtype=type(G))
    
    while (graphs_done < n_graphs):
        new_graph = False
        
        while new_graph == False:
            graph = connected_graph(verts)
            graph_mat = np.asarray(nx.adjacency_matrix(graph).todense())
            
            #does our new graph equal anything we've seen before
            new_graph = ~((graph_mat == graphs).all((1,2)).any())
        
        graphs[graphs_done] = graph_mat
        graphs_obj[graphs_done] = graph
        graphs_done += 1

    return graphs #can also return graphs if need adjacency matrix



def shrink(graph, node_base, node):
    graph = nx.contracted_nodes(graph, node_base, node, self_loops=False)
    return graph

def find_non_connected_pairs(G):
    """
    Find all non-connected pair of nodes in graph G.
    
    Args:
        G: a single input graph in adjacency matrix form.
    
    Returns:
        pairs_of_nodes: a list of pairs of non-connected nodes, (x, y), where x and y 
        are indices of two nodes. For example, if (x,y) is in the list, then there is no edge between
        xth and yth node. Note, to avoid duplication, only pairs with y > x will be added to the list.
        i.e. if there is not edge between node 5 and 7, only (5,7) will show up in the list, (7,5) will
        be ignored.
    
    """
    pairs_of_nodes = []
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i][j] == 0 and i < j:
                pairs_of_nodes.append((i,j))
    return pairs_of_nodes 


def find_ground_truth(G, pairs_of_nodes):
    """
    Find the pair of nodes that, after shrink and add, gives the maximum lower bound. 
    
    Args: 
        G: input graph
        pairs_of_nodes: a list of pairs of non-connected nodes associated with G
      
    Returns:
        A pair(tuple) of nodes (x, y) which gives the maximum lower bound.
    """
    
    g = nx.from_numpy_matrix(G) # convert adjacency matrix to graph object
    lower_bound = {} # create a dictionary to hold pairs and assosiated lower bound
    for i in pairs_of_nodes:
        c = g.copy() # add_edge modifies graph inplace, thus I make a copy of the original graph.
        graph1 = shrink(g, i[0], i[1]) # shrink two nodes
        c.add_edge(i[0], i[1]) # add an edge
        lower_bound[i] = min(nx.graph_clique_number(graph1), nx.graph_clique_number(c)) # find the
        # smaller clique number between shrunk graph and added graph, which is the lower bound.
        
    return max(lower_bound, key = lower_bound.get) # return the pair of nodes with maximum lower bound.

graphs = generate_graphs(num_of_nodes, num_of_graphs) # generate graphs

labels = np.zeros(2*num_of_graphs).reshape((num_of_graphs, 2)).astype('int') # initialize labels

count = graphs.shape[0] - 1

for i in range(graphs.shape[0]):
    if i > count:
          break
    pairs_of_nodes = find_non_connected_pairs(graphs[i]) # find non-connected pairs of nodes
    if not pairs_of_nodes: # if all nodes are connected, i.e. a complete graph, remove the graph
        graphs = np.delete(graphs, i, axis=0)
        count -= 1
        continue
    labels[i] = find_ground_truth(graphs[i], pairs_of_nodes)

labels = labels[:count+1]

np.save("graph_dataset", graphs)
np.save("labels_dataset", labels)
