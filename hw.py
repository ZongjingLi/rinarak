import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

def visualize_graph_and_mst_hexagonal(edges, mst_edges):
    """
    Visualize the original graph and its minimum spanning tree with a hexagonal layout
    
    Args:
        edges: List of tuples (u, v, weight) representing edges of the original graph
        mst_edges: List of tuples (u, v, weight) representing edges of the MST
    """
    # Create the original graph
    G = nx.Graph()
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
    
    # Create the MST graph
    MST = nx.Graph()
    for u, v, weight in mst_edges:
        MST.add_edge(u, v, weight=weight)
    
    # Set up the visualization
    plt.figure(figsize=(15, 7))
    
    # Define a hexagonal layout
    # Position nodes at the vertices of a regular hexagon
    pos = {
        'a': (0, 0),          # bottom left
        'b': (1, 0),          # bottom right
        'c': (1.5, 0.9),      # right
        'd': (1, 1.8),        # top right
        'e': (0, 1.8),        # top left
        'f': (-0.5, 0.9)      # left
    }
    
    # Plot the original graph
    plt.subplot(1, 2, 1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')
    
    # Draw specific edges with different colors
    edge_colors = []
    edge_list = []
    
    for u, v, _ in edges:
        edge_list.append((u, v))
        # Check if this edge is on the outer hexagon
        if ((u == 'a' and v == 'b') or (u == 'b' and v == 'a') or
            (u == 'b' and v == 'c') or (u == 'c' and v == 'b') or
            (u == 'c' and v == 'd') or (u == 'd' and v == 'c') or
            (u == 'd' and v == 'e') or (u == 'e' and v == 'd') or
            (u == 'e' and v == 'f') or (u == 'f' and v == 'e') or
            (u == 'f' and v == 'a') or (u == 'a' and v == 'f')):
            edge_colors.append('navy')
        else:
            edge_colors.append('gray')
    
    # Draw edges with appropriate colors
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=2.0, edge_color=edge_colors)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', 
                          edgecolors='black', linewidths=2)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold')
    
    # Add edge labels (weights)
    edge_labels = {(u, v): w for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='blue')
    
    plt.title("Original Graph", fontsize=16)
    plt.axis('off')
    
    # Plot the MST
    plt.subplot(1, 2, 2)
    
    # Draw edges
    nx.draw_networkx_edges(MST, pos, width=2.5, edge_color='red')
    
    # Draw nodes
    nx.draw_networkx_nodes(MST, pos, node_size=700, node_color='lightgreen', 
                          edgecolors='black', linewidths=2)
    
    # Add node labels
    nx.draw_networkx_labels(MST, pos, font_size=15, font_weight='bold')
    
    # Add edge labels (weights) for MST
    mst_edge_labels = {(u, v): w for u, v, w in mst_edges}
    nx.draw_networkx_edge_labels(MST, pos, edge_labels=mst_edge_labels, font_size=12, font_color='blue')
    
    plt.title("Minimum Spanning Tree", fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("graph_with_mst_hexagonal.png", dpi=300, bbox_inches='tight')
    plt.show()

def kruskal_algorithm(edges):
    """
    Implementation of Kruskal's algorithm to find minimum spanning tree
    
    Args:
        edges: List of tuples (u, v, weight) representing edges
        
    Returns:
        List of tuples (u, v, weight) forming the MST
    """
    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    # Initialize disjoint set for each vertex
    vertices = set()
    for u, v, _ in edges:
        vertices.add(u)
        vertices.add(v)
    
    parent = {vertex: vertex for vertex in vertices}
    
    # Find operation with path compression
    def find(vertex):
        if parent[vertex] != vertex:
            parent[vertex] = find(parent[vertex])
        return parent[vertex]
    
    # Union operation
    def union(vertex1, vertex2):
        root1 = find(vertex1)
        root2 = find(vertex2)
        parent[root1] = root2
    
    # Initialize MST
    mst = []
    
    # Process edges in sorted order
    for u, v, weight in sorted_edges:
        # Check if adding this edge creates a cycle
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, weight))
            
            # MST has (n-1) edges where n is the number of vertices
            if len(mst) == len(vertices) - 1:
                break
    
    return mst

# Define edges based on the hexagonal image provided
edges = [
    # Outer hexagon edges
    ('a', 'b', 3), ('b', 'c', 1), ('c', 'd', 1), 
    ('d', 'e', 2), ('e', 'f', 1), ('f', 'a', 5),
    # Inner edges
    ('a', 'c', 2), ('a', 'e', 4), ('b', 'd', 3),
    ('b', 'e', 3), ('c', 'f', 4),
    ('d', 'f', 3), 
]

# Find the MST using Kruskal's algorithm
mst_edges = kruskal_algorithm(edges)

# Print the results
print("Edges in the Minimum Spanning Tree:")
for u, v, weight in sorted(mst_edges, key=lambda x: x[2]):
    print(f"({u}, {v}): {weight}")

# Calculate total weight
total_weight = sum(weight for _, _, weight in mst_edges)
print(f"\nTotal weight of MST: {total_weight}")

# Check if MST is unique
edge_weights = {}
for _, _, weight in sorted(edges, key=lambda x: x[2]):
    if weight not in edge_weights:
        edge_weights[weight] = 0
    edge_weights[weight] += 1

print("\nEdge weight distribution:")
for weight, count in sorted(edge_weights.items()):
    print(f"Weight {weight}: {count} edges")

# Visualize the graphs
visualize_graph_and_mst_hexagonal(edges, mst_edges)