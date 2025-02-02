"""
This module contains the definition of the ControlledWaxmanGraph class,
which is used to generate and manipulate a two-layer Waxman graph with controlled parameters.
"""

import networkx as nx
import numpy as np
import random
import math

from networkx.utils import py_random_state
from itertools import combinations

class ControlledWaxmanGraph():
    def __init__(self, graph, k_param, radius):
                
        self.graph = graph
        self.generate_graph(k_param, radius)
        self.nodes = list(self.graph.nodes())
        
        self.original_edges = [random.choice([(u, v), (v, u)]) 
                               for (u, v) in self.graph.edges() 
                               if u < v]
        
        random.shuffle(self.original_edges)
        
    def add_coords(self, G: nx.Graph, radius: int = 1) -> nx.Graph:
        """Add x and y coordinates to each node in the graph.
        
        Args:
            G (nx.Graph): A graph.
            radius (int, optional): Radius of the circle. Defaults to 1.
        
        Returns:
            nx.Graph: Returns a graph with x and y coordinates.
        """
        angle_increment = 2 * np.pi / G.number_of_nodes()
    
        # Assign x and y coordinates to each node
        for i, node in enumerate(G.nodes()):
            angle = i * angle_increment
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            G.nodes[node]['x'] = x
            G.nodes[node]['y'] = y
            G.nodes[node]['coords'] = (x, y)
        
        return G
            
    def calculate_distance(self, n:int, k:int, radius:float) -> float:
        """Calculate the distance between two nodes.
        
        Args:
            n (int): Number of nodes
            k (int): Parameter k of linking edges.
            radius (float): Radius of the circle.
        
        Returns:
            float: Returns the distance between two nodes.
        """
        a = 1
        b = 1
        angle = math.pi * k / n
        
        return math.sqrt(a**2 + b**2 - 2*a*b*math.cos(angle))*radius
    
    def calc_sum(self, N, alpha):
        sum_result = 0
        for i in range(N//2, 0, -1): # going from N/2 to 1 (inclusive)
            d = math.sqrt(2*(1-math.cos(math.pi*i*2/N)))
            val = math.exp(-(d/2)/alpha)
            sum_result += 2 * val
            if i == N/2 :
                sum_result -= val
        return sum_result

    def est_alpha(self, N, k, eps): # eps = precision, supposing N is even
        lo_a, hi_a = 0, math.inf # E[d_v] = 0 (degenerate) for alpha = 0, E[d_v] --> N/2 for alpha --> oo
        while hi_a - lo_a > eps:
            mid_a = lo_a + (hi_a - lo_a) / 2 if hi_a != math.inf else (2 * lo_a if lo_a > 0 else 0.0001)
            sum_result = self.calc_sum(N, mid_a)
            if sum_result < k:
                lo_a, hi_a = mid_a, hi_a
            else:
                lo_a, hi_a = lo_a, mid_a
        # print("N =", N, "/ k =", k, "==> alpha = ", lo_a + (hi_a - lo_a) / 2)
        # print(str([lo_a, hi_a]))
        # print(list(map(lambda a:self.calc_sum(N,a),[lo_a, hi_a])))
        return lo_a + (hi_a - lo_a) / 2
        
    @py_random_state(3)
    def generate_graph(self, k: int, radius: int, seed=None) -> nx.Graph:
        """Generate a Waxman graph.
        
        Args:
            k(int): Parameter k of linking edges.
            radius(int): Radius of the circle.
            seed (optional): Random seed. Defaults to None            
        
        Returns:
            nx.Graph: Returns a Waxman graph.
            
        Raises:
            nx.NetworkXError: If k <= 0, a NetworkXError is raised.
            
        """
        if k <= 0:
            raise nx.NetworkXError("k must be positive")
        
        n = self.graph.number_of_nodes()            
        
        beta_waxman = 1
        l_waxman = radius*2
        alpha_waxman = self.est_alpha(n, k, 1e-9)
        
        def dist(u, v):
            if 'coords' not in self.graph.nodes[u] or 'coords' not in self.graph.nodes[v]:            
                x1 = self.graph.nodes[u]['x']
                y1 = self.graph.nodes[u]['y']
                x2 = self.graph.nodes[v]['x']
                y2 = self.graph.nodes[v]['y']
            else:
                x1, y1 = self.graph.nodes[u]['coords']
                x2, y2 = self.graph.nodes[v]['coords']
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # `pair` is the pair of nodes to decide whether to join.b
        def should_join(pair):
            u, v = pair
            prob = beta_waxman * math.exp(-dist(u, v) / (alpha_waxman * l_waxman))
            return seed.random() < prob
        
        self.graph.add_edges_from(filter(should_join, combinations(self.graph, 2)))
        
        return self.graph
    
    @py_random_state(2)
    def randomize_edges(self, e, seed=None) -> None:
        """Randomize e edges of the graph.

        Args:
            e: Number of edges to be randomized.
            seed (optional): Random seed. Defaults to None.

        Raises:
            nx.NetworkXError: number of edges to be randomized is greater than the number of original edges.
        """
        if e > len(self.original_edges):
            raise nx.NetworkXError("e > original_edges, choose smaller e")
        
        for _ in range(e):
            (u,v) = self.original_edges.pop()
            
            while True:
                w = seed.choice(self.nodes)
                
                if (u, w) in self.graph.edges() \
                    or u == w:
                    continue
                
                self.graph.remove_edge(u, v)
                self.graph.add_edge(u, w)
                break
        