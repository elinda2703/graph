import osmnx as ox
from math import *
from networkx import Graph, MultiGraph


def load_data (place_name):
    
    mulitidigraph = ox.graph_from_place(place_name, network_type='drive')
    unoriented_graph = Graph(mulitidigraph)
    return mulitidigraph,unoriented_graph

def visualize_skeleton(subgraph,whole_multi_graph):

    # Plot the entire graph
    fig, ax = ox.plot_graph(
        whole_multi_graph, 
        node_size=5, 
        edge_linewidth=1,
        show=False, 
        close=False  # Keep the plot open for overlay
    )
    xlim = ax.get_xlim()  # (xmin, xmax)
    ylim = ax.get_ylim()  # (ymin, ymax)
    
    # Plot the path on top of the graph
    ox.plot_graph(
        MultiGraph(subgraph), 
        ax=ax,  # Use the same axis as the main graph
        node_color="red", 
        edge_color="red", 
        node_size=10, 
        edge_linewidth=3,
        edge_alpha=0.5,
        node_alpha=0.5,
        bbox= (xlim[0],ylim[0],xlim[1],ylim[1]),
        show=True,  # Display the combined plot
        close=True
        )
def graph_to_edges_and_vertices(graph_from_osm):
    E = []  # List of edges: [node1, node2, weight]
    V = list(graph_from_osm.nodes())  # List of vertices (nodes)

    for node, neighbors in graph_from_osm.adjacency():
        for neighbor, edge in neighbors.items():
            length = edge["length"]  # Get the edge length
            E.append([node, neighbor, length])  # Add edge to list

    return V, E

class DisjointSet:
    def __init__(self, nodes):
        # Initialize Disjoint set
        self.parent = {v: v for v in nodes} # Every node is its own parent
        self.rank = {v: 0 for v in nodes} # Every node has rank 0

    def find(self, vertex):
        # Find parent of a node
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])  # Path compression
        return self.parent[vertex]

    def union(self, root1, root2):
        
        # Connect subtree with lower rank to subtree with higher rank
        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        elif self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1


def kruskal(edges,nodes):
    
    # Sort edges by weight
    edges.sort(key=lambda edge: (edge[2], edge[0], edge[1]))  # Sort edges by weight, then by vertices for tie-breaking

    # Initialize disjoint sets for cycle detection
    disjoint_set = DisjointSet(nodes)
    total_weight = 0
    
    # Iterate through edges
    mst = []
    for u, v, weight in edges:
        root_u = disjoint_set.find(u)
        root_v = disjoint_set.find(v)

        # If u and v are not in the same set, include this edge in the MST
        if root_u != root_v:
            mst.append((u, v, weight))
            disjoint_set.union(root_u, root_v)
            total_weight+=weight

    return mst,total_weight

place_name = "Lousa, Coimbra, Portugal"

multidi_coimbra,graph_coimbra=load_data(place_name)

vertices,edges=graph_to_edges_and_vertices(graph_coimbra)

skeleton,weight=kruskal(edges,vertices)


edges_without_weights = [(u, v) for u, v, _ in skeleton]
subgraph_skeleton=graph_coimbra.edge_subgraph(edges_without_weights)

visualize_skeleton(subgraph_skeleton,multidi_coimbra)