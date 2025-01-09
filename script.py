import osmnx as ox
import heapq
from math import *
import numpy as np
from networkx import Graph, MultiGraph
from shapely.geometry import Point
import pandas as pd


# Dictionary with maximum speeds for different road types
DEFAULT_MAXSPEEDS = {
        'trunk': 120,
        'trunk_link': 120,
        'tertiary_link': 120,    
        'unclassified': 50,   
        'residential': 50,        
        'living_street': 50,     
        'tertiary': 90, 
        'secondary_link': 90,
        'secondary': 90,          
        'primary_link': 90,
        'primary': 90,           
    }



def load_data (place_name):
    # Loads MultiDiGraph from OSM
    mulitidigraph = ox.graph_from_place(place_name, network_type='drive')
    
    # Conversion to shapefiles
    #gdf_nodes,gdf_edges=ox.graph_to_gdfs(mulitidigraph)
    #gdf_nodes.to_file(f"nodes.shp")
    #gdf_edges.to_file(f"edges.shp")
    
    # Conversion to unoriented graph
    unoriented_graph = Graph(mulitidigraph)
    return mulitidigraph,unoriented_graph


def get_length(edge):
    
    # Calculation and return of edge length from attribute
    length=edge['length']
    return length

def get_time(edge):
    
    # Calculation and return of travel time trough an edge
    length_attr=edge["length"] # Edge length from the attribute
            
    if "maxspeed" in edge and isinstance(edge["maxspeed"], str):
        # If maximum speed is defined
        max_speed = int(edge["maxspeed"]) / 3.6
    else:
        
        road_type = edge["highway"]
        if road_type in DEFAULT_MAXSPEEDS.keys():
            # Calculate maximum speed from road type if it is in the dictionary
            max_speed = DEFAULT_MAXSPEEDS[road_type] / 3.6
        else:
            # Set maximum speed to 18.89 m/s (50 km/h)
            max_speed=50/3.6
    if "geometry" in edge:
        shapely_length=edge['geometry'].length  # Using shapely length to calculate sinusoidality
        coords = edge['geometry'].coords
        x1, y1 = coords[0]  # First point
        x2, y2 = coords[-1]  # Last point
        start_point = Point(x1, y1)
        end_point = Point(x2, y2)
        euclidean_distance = start_point.distance(end_point) # Distance between first and last point of the Linestring
        sinusoidality = shapely_length / euclidean_distance if euclidean_distance != 0 else 1
        enhanced_distance = sinusoidality * length_attr
        time = enhanced_distance / max_speed  # Time in seconds
    else:
        time=length_attr/max_speed
    return time



def graph_dict(graph_from_osm,weight_type):
    # Creation of a simple dictionary for further processing
    
    if weight_type=="length":
        get_weight=get_length
    elif weight_type=="time":
        get_weight=get_time
    else:
        raise ValueError(f"Invalid weight_type: {weight_type}. Must be 'length' or 'time'.")
    G={}
    for node, neighbors in graph_from_osm.adjacency():
        
        # Initialization of subdicionary for each node
        G[node] = {}
        for neighbor, edge in neighbors.items():
            
            # Assign weight to neighbour
            weight=get_weight(edge)
            G[node][neighbor] = weight
    return G

def dijkstra(graph, start):
    """
    Use Dijkstra algorithm to find shortest paths from a start node to all nodes in the graph.
    
    Parameters:
        graph (dict): Dictionary with all nodes as keys and dictionary with values neigbour:weight as values.
        start (int): Id of the starting node
    
    Returns:
        distances (dict): Dictionary with node ids as keys and distance along shortest path as values.
        predecessors (dict): Dictionary with node ids as keys and their predecessors along shortest path as values.
    """
    # Priority queue for the Dijkstra algorithm
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    
    # Initialization distances and predecessors
    distances = {node: inf for node in graph} # Infinity as distance to all points
    distances[start] = 0 # Start set to zero
    predecessors = {node: None for node in graph} # Predecessors set to None
    
    # Initialization set for visited nodes
    visited = set()
    
    while priority_queue:
        # Start processing current node
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip if the node is already visited
        if current_node in visited:
            continue
        
        # Mark the current node as visited
        visited.add(current_node)
        
        # Relaxation step for all neighbors
        for neighboring_node, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # If a shorter path is found, update the distance and predecessor
            if distance < distances[neighboring_node]:
                distances[neighboring_node] = distance
                predecessors[neighboring_node] = current_node
                heapq.heappush(priority_queue, (distance, neighboring_node))
    
    return distances, predecessors

def get_path(predecessors,node_id):
    
    # Returns list of nodes that the path went trough from predecessor dictionary
    node_list=[node_id]
    
    # Check and add predecessors until it reaches the start node
    while predecessors[node_id] is not None:
        node_list.append(predecessors[node_id])
        node_id=predecessors[node_id]
    
    # Reverse the node list from start to end    
    node_list.reverse()    
    return node_list

def get_distance(calculated_distances,end_node_id):
    
    # Return distance to an end node from distances dictionary
    start_end_distance = calculated_distances[end_node_id]
    return start_end_distance
    
def all_distances(graph):
    # Places distances between all pairs of nodes in a matrix and exports it to XLS file
    
    node_list=list(graph.keys()) # List of all node ids
    number_of_nodes=len(graph)

    # Initialize the matrix with infinity
    dist_matrix = np.full((number_of_nodes,number_of_nodes), np.inf)
        
    for node_start in node_list:
        
        # Run Dijkstra algorithm for all starting nodes
        dists,_=dijkstra(graph,node_start)
        
        # Get index of the start node from node list
        start_index=node_list.index(node_start)
        
        for node_end,distance in dists.items():
            
            # Get index of the end node from node list
            end_index=node_list.index(node_end)
            
            # Put distance in the matrix
            dist_matrix[start_index,end_index]=distance
    df = pd.DataFrame(dist_matrix, index=node_list, columns=node_list)

    # Export to Excel
    output_file = "distance_matrix.xlsx"
    df.to_excel(output_file)
    return dist_matrix
                        

def visualize_path(subgraph,whole_multi_graph):

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
    
    # Plot the path (subgraph) on top of the graph
    ox.plot_graph(
        MultiGraph(subgraph), 
        ax=ax,  # Use the same axis as the main graph
        node_color="red", 
        edge_color="red", 
        node_size=30, 
        edge_linewidth=3,
        edge_alpha=0.5,
        node_alpha=0.5,
        bbox= (xlim[0],ylim[0],xlim[1],ylim[1]),
        show=True,  # Display the combined plot
        close=True
        )


place_name = "Lousa, Coimbra, Portugal"

multidi_coimbra,graph_coimbra=load_data(place_name)

start_node=252865248
end_node=1601656647

weights="length"

G=graph_dict(graph_coimbra,weights)


distances,predecessors=dijkstra(G,start_node)

path_weight=get_distance(distances,end_node)


if weights=="time":
    print(f"Total time of the shortest path is {path_weight} seconds.")
if weights=="length":
    print(f"Total length of the shortest path is {path_weight} meters.")

path=get_path(predecessors,end_node)

# Convert path to subgraph for visualization
path_subgraph=graph_coimbra.subgraph(path)

all_dists=all_distances(G)
visualize_path(path_subgraph,multidi_coimbra)    