import networkx as nx
from utils import clean_dict_datatype, create_tuple_list, remove_tree_brackets, edges_attr_list
from utils import graph_to_string, string_to_graph, build_list_dict_attribute

class SimulationGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()  # Initialize the parent nx.Graph
    
    def __repr__(self):
        return f"Site Graph with {len(self.nodes)} nodes and {len(self.edges)} edges"
    
    def add_nodes(self, node_indices: dict, node_attributes: dict, labels: list):
        node_indices = remove_tree_brackets(node_indices)
        node_attributes = clean_dict_datatype(node_attributes)
        node_attr_tuple = create_tuple_list(node_indices, node_attributes, labels)
        
        super().add_nodes_from(node_attr_tuple)
    
    def add_edges(self, edge_pair_indices: dict, edge_attributes: dict, labels: list):
        edge_attributes = clean_dict_datatype(edge_attributes)
        cleaned_attributes = clean_dict_datatype(edge_attributes)
    
        pairs = list(edge_pair_indices.values())
        paired_values = list(cleaned_attributes.values())

        list_of_dict_from_values = build_list_dict_attribute(paired_values, labels)
        tuples_node_attr = edges_attr_list(pairs, list_of_dict_from_values)

        super().add_edges_from(tuples_node_attr)

    def get_graph(self):
        return graph_to_string(self)