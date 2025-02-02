
import seaborn as sns; sns.set_theme()
import Models.ControlledWaxmanGraph as cwg
import matplotlib.pyplot as plt
import AcessoryMethods
import networkx as nx
import pandas as pd
import numpy as np
import os
import time

from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from scipy.optimize import linear_sum_assignment

am = AcessoryMethods.AcessoryMethods()

# Definindo número de vértices do grafo
NUMBER_OF_VERTICES = 50000

ks = [10, 20, 40, 80, 160] # Parâmetros k para o modelo Waxman de duas camadas
porcentage_edges_to_randomize = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 16] # Porcentagens de arestas a serem randomizadas
results_df = pd.DataFrame(columns=porcentage_edges_to_randomize, index=ks)

grafo: nx.Graph = nx.Graph()
grafo.add_nodes_from(range(NUMBER_OF_VERTICES))
radius = 1
angle_increment = 2 * np.pi / NUMBER_OF_VERTICES

for i, node in enumerate(grafo.nodes()):
    angle = i * angle_increment
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    grafo.nodes[node]['x'] = x
    grafo.nodes[node]['y'] = y

for k in ks:        
    print(f"Gerando o grafo Waxman controlado com k={k}...")
    
    t0 = time.time()
    controle_grafo = cwg.ControlledWaxmanGraph(grafo, k, radius)
    print("Grafo gerado com sucesso em {0:.2f} minutos".format((time.time() - t0)/60))
    grafo_original = controle_grafo.graph.copy()
    comunidade_geo = nx.community.louvain_communities(grafo_original)
     
    nx.write_edgelist(grafo_original, f"../grafo_geo_{k}.edgelist")
    nx.write_graphml(grafo_original, f"../grafo_geo_{k}.graphml")
    print("Grafo original salvo com sucesso")
        
    results_df.index.name = f"edges randomized for k={k}"
    
    number_of_edges = grafo.number_of_edges()

    
    for index, porcentage in enumerate(porcentage_edges_to_randomize):
        print(f"Randomizando {porcentage}% das arestas...")
        if index == 0:
            edges = int(round((number_of_edges * porcentage / 100), 0))
        else:
            edges = int(round((number_of_edges * (porcentage - porcentage_edges_to_randomize[index-1]) / 100), 0))
        
        controle_grafo.randomize_edges(edges)
        grafo = controle_grafo.graph
        
        nx.write_edgelist(grafo, f"../waxman_grafo_{k}_porcentagem_{str(porcentage)}.edgelist")

        # %%
        print("Gerando as comunidades...")
        t0 = time.time()
        comunidade_social_waxman = nx.community.louvain_communities(grafo)
        print(f"Tempo para gerar as comunidades: {(time.time() - t0)/60} minutos")

        nx.write_graphml(grafo, f"../waxman_grafo_social_{k}_porcentagem_{str(porcentage)}.graphml")
        
        node_community_map_social = am.map_community_nodes(comunidade_social_waxman)
        node_community_map_geo = am.map_community_nodes(comunidade_geo)
        
        am.plot_colored_communities(grafo, grafo_original, comunidade_social_waxman, comunidade_geo,
                                 with_labels=False, latitude='y', longitude='x',
                                 output_file_name=f'waxman_comunidades_{k}_porcentagem_{str(porcentage)}.png')
        
        jaccard_matrix = np.zeros((len(comunidade_social_waxman), len(comunidade_geo)))

        for i, com_social in enumerate(comunidade_social_waxman):
            for j, com_geo in enumerate(comunidade_geo):
                jaccard_matrix[i][j] = am.jaccard_similarity(set(com_social), set(com_geo))

        jaccard_values = jaccard_matrix[jaccard_matrix != 0]
        upper_limit = np.max(jaccard_values)*1.2
        division = np.max(jaccard_values)/5
        
        # Usando o algoritmo Hungarian para encontrar a melhor correspondência para a diagonal
        cost_matrix = -jaccard_matrix  # Inverter sinais para transformar em problema de maximização
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reordenando as linhas e colunas para maximizar a diagonal
        permuted_matrix = jaccard_matrix[row_ind][:, col_ind]
        
        if jaccard_matrix.shape[0] > jaccard_matrix.shape[1]:
            missing_rows = [i for i in range(jaccard_matrix.shape[0]) if i not in row_ind]
            if missing_rows:
                additional_rows = jaccard_matrix[missing_rows, :]
                permuted_matrix = np.vstack([permuted_matrix, additional_rows])
        elif jaccard_matrix.shape[0] < jaccard_matrix.shape[1]:
            missing_columns = [i for i in range(jaccard_matrix.shape[1]) if i not in col_ind]
            if missing_columns:
                additional_columns = jaccard_matrix[:, missing_columns]
                permuted_matrix = np.hstack([permuted_matrix, additional_columns])
                
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(permuted_matrix, cmap='Oranges')
        plt.title(f'Similaridade de Jaccard entre comunidades para k={k} e {porcentage}% de arestas randomizadas')
        plt.xlabel('Comunidades Geográficas')
        plt.ylabel('Comunidades Sociais')
        plt.savefig(f"../waxman_heatmap_{k}_{porcentage}.png", dpi=300)
        plt.close()

        # AMI
        ami_jaccard = ami(
            am.assign_labels(comunidade_social_waxman, grafo.nodes()),
            am.assign_labels(comunidade_geo, grafo_original.nodes())
        )
        print(f"AMI para k={k}, {porcentage}%: {ami_jaccard}")

        # Salvando resultados
        if not os.path.exists(f"../waxman_ami_{k}.txt"):            
            with open(f"../waxman_ami_{k}.txt", 'w', encoding='utf-8') as f:
                f.write(f"Porcentagem {porcentage} - AMI {ami_jaccard}\n")
            print(f"arquivo ami_{k}.txt salvo")
        else:
            with open(f"../waxman_ami_{k}.txt", 'a', encoding='utf-8') as f:
                f.write(f"Porcentagem {porcentage} - AMI {ami_jaccard}\n")
    


