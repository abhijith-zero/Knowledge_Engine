import json
import networkx as nx
from pyvis.network import Network

def build_knowledge_graph(input_file, output_html):
    # Load triplets (subject, relation, object)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes and edges
    for item in data:
        subject = item.get("subject") or item.get("topic1") or "Unknown"
        relation = item.get("relation") or "related_to"
        obj = item.get("object") or item.get("topic2") or "Unknown"

        G.add_node(subject, title=subject)
        G.add_node(obj, title=obj)
        G.add_edge(subject, obj, title=relation)

    # Create interactive network
    net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.write_html(output_html)

    print(f"✅ Knowledge Graph saved to {output_html}")

if __name__ == "__main__":
    build_knowledge_graph("triplets_output.json", "outputs/knowledge_graph.html")
