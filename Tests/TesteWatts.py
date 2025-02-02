
import seaborn as sns; sns.set_theme()
import Models.ControlledWattsGraph as cwg
import matplotlib.pyplot as plt
import AcessoryMethods
import networkx as nx
import pandas as pd
import numpy as np
import time
import math
import os

from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from scipy.optimize import linear_sum_assignment

am = AcessoryMethods.AcessoryMethods()

# Definindo número de vértices do grafo
NUMBER_OF_VERTICES = 50000

ks = [10, 20, 40, 80, 160] # Parâmetros k para o modelo Watts-Strogatz de duas camadas
porcentage_edges_to_randomize = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 16] # Porcentagens de arestas a serem randomizadas
results_df = pd.DataFrame(columns=porcentage_edges_to_randomize, index=ks)

for k in ks:
    print(f"Gerando o grafo controlado com k={k}...")
    controle_grafo = cwg.ControlledWattsGraph(NUMBER_OF_VERTICES, k)
    grafo_geo = controle_grafo.graph
    
    radius = 1
    angle_increment = 2 * np.pi / NUMBER_OF_VERTICES

    for i, node in enumerate(grafo_geo.nodes()):
        angle = i * angle_increment
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        grafo_geo.nodes[node]['x'] = x
        grafo_geo.nodes[node]['y'] = y
        grafo_geo.nodes[node]['coords'] = (x, y)
        
    number_of_edges = grafo_geo.number_of_edges()
    
    a = 1
    b = 1
    angulo = (360 * (k+1) / 2) / NUMBER_OF_VERTICES

    c = math.sqrt(a**2 + b**2 - 2*a*b*math.cos(math.radians(angulo)))*radius
    
    
    for index, porcentage in enumerate(porcentage_edges_to_randomize):
        print(f"Randomizando {porcentage}% das arestas...")
        if index == 0:
            edges = int(round((number_of_edges * porcentage / 100), 0))
        else:
            edges = int(round((number_of_edges * (porcentage - porcentage_edges_to_randomize[index-1]) / 100), 0))
        
        controle_grafo.randomize_edges(edges)
        grafo_social = controle_grafo.graph
        nx.write_edgelist(grafo_social, f"../watts_grafo_{k}_porcentagem_{str(porcentage)}.edgelist")
        
        
        t0 = time.time()
        comunidade_social_watts = nx.community.greedy_modularity_communities(grafo_social)
        comunidade_geo = nx.community.greedy_modularity_communities(grafo_geo)
        time_difference = (time.time() - t0) / 60
        print(f"Tempo de execução: {time_difference:.2f} minutos")        

        node_community_map_social = am.map_community_nodes(comunidade_social_watts)
        node_community_map_geo = am.map_community_nodes(comunidade_geo)

        for i, node in enumerate(grafo_social.nodes()):
            grafo_social.nodes[node]['social_community'] = node_community_map_social[node]
            grafo_social.nodes[node]['geo_community'] = node_community_map_geo[node]
        
        am.plot_colored_communities(grafo_social, grafo_geo, comunidade_social_watts, comunidade_geo, 
                                 with_labels=False, latitude='y', longitude='x',
                                 output_file_name=f'watts_comunidades_{k}_porcentagem_{str(porcentage)}.png')
        
        jaccard_matrix_watts = np.zeros((len(comunidade_social_watts), len(comunidade_geo)))

        for i, com_social in enumerate(comunidade_social_watts):
            for j, com_geo in enumerate(comunidade_geo):
                jaccard_matrix_watts[i][j] = am.jaccard_similarity(set(com_social), set(com_geo))
        
        jaccard_values_watts = jaccard_matrix_watts[jaccard_matrix_watts !=0]
        
        # Usando o algoritmo Hungarian para encontrar a melhor correspondência para a diagonal
        cost_matrix = -jaccard_matrix_watts  # Inverter sinais para transformar em problema de maximização
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reordenando as linhas e colunas para maximizar a diagonal
        permuted_matrix = jaccard_matrix_watts[row_ind][:, col_ind]
        
        missing_rows = [i for i in range(jaccard_matrix_watts.shape[0]) if i not in row_ind]
        if missing_rows:
            additional_rows = jaccard_matrix_watts[missing_rows, :]
            permuted_matrix = np.vstack([permuted_matrix, additional_rows])

        plt.figure(figsize=(24, 12))

        plt.subplot(1, 2, 1)
        sns.heatmap(jaccard_matrix_watts, cmap='Oranges')
        plt.xlabel('Comunidades geográficas')
        plt.ylabel('Comunidades sociais')
        plt.title('Similaridade de Jaccard entre comunidades sociais e geográficas')

        plt.subplot(1, 2, 2)
        sns.heatmap(permuted_matrix, cmap='Oranges')
        plt.xlabel('Comunidades geográficas')
        plt.ylabel('Comunidades sociais')
        plt.title('Similaridade de Jaccard entre comunidades sociais e geográficas (permutada)')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"..\watts_heatmap_{k}_porcentagem_{str(porcentage)}.png", dpi=300, bbox_inches='tight')
        plt.close()

       # AMI
        ami_jaccard = ami(
            am.assign_labels(comunidade_social_watts, grafo_social.nodes()),
            am.assign_labels(comunidade_geo, grafo_geo.nodes())
        )
        print(f"AMI para k={k}, {porcentage}%: {ami_jaccard}")

        # Salvando resultados
        if not os.path.exists(f"../watts_ami_{k}.txt"):            
            with open(f"../watts_ami_{k}.txt", 'w', encoding='utf-8') as f:
                f.write(f"Porcentagem {porcentage} - AMI {ami_jaccard}\n")
            print(f"arquivo ami_{k}.txt salvo")
        else:
            with open(f"../watts_ami_{k}.txt", 'a', encoding='utf-8') as f:
                f.write(f"Porcentagem {porcentage} - AMI {ami_jaccard}\n")
    




