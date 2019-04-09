"""
Sample usage :


import graph_ops as go

## generate 100 unique graphs on 10 vertices
graphs = go.generate_graphs(10, 100)
// Some code on graph adjancies matrix

## compress graphs
graphs_comp = go.compress_graphs(graphs)

// train nueral nets with graphs_comp as X


"""


import numpy as np
import networkx as nx

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
        
    """
    
    graphs_done = 0
    graphs = np.zeros((n_graphs,verts,verts), dtype = int)
    
    while (graphs_done < n_graphs):
        new_graph = False
        
        while new_graph == False:
            graph = connected_graph(verts)
            graph_mat = np.asarray(nx.adjacency_matrix(graph).todense())
            
            #does our new graph equal anything we've seen before
            new_graph = ~((graph_mat == graphs).all((1,2)).any())
        
        graphs[graphs_done] = graph_mat
        graphs_done += 1

    return graphs


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