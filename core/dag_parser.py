import json
import networkx as nx

def load_dag_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()
    for task in data["tasks"]:
        G.add_node(task["id"], **task)

    for edge in data["dependencies"]:
        G.add_edge(edge["from"], edge["to"])

    return G
