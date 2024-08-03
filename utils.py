import networkx as nx
import json
from networkx.readwrite import json_graph


def clean_dict_datatype(a_dict: dict) -> dict:
    """Hops only accepts one datatype per input.
    We force the str datatype at input (convert if necessary),
    then convert back to either integer, float or string
    
    Note that bools have deliberately not been integrated as they create conflicts when reading
    data from the graph (for example when removing edges from a graph)"""

    for key_index in a_dict.keys():
        for elem in range(len(a_dict[key_index])):
            current_elem = a_dict[key_index][elem]

            if str(current_elem).isdigit():
                a_dict[key_index][elem] = int(current_elem)
            elif current_elem.replace('.', '', 1).isdigit() and current_elem.count('.') < 2:
                a_dict[key_index][elem] = float(current_elem)
            else:
                a_dict[key_index][elem] = str(current_elem)
    return a_dict

def remove_tree_brackets(a_dict: dict) -> dict:
    """remove the {} of the paths (that are the keys of the dictionary)"""
    new_dict = {}
    for key in a_dict.keys():
        try:
            new_dict[int(str(key)[1:-1])] = a_dict[key]  # try get integer key... but is it even good?
        except TypeError:
            new_dict[str(key)[1:-1]] = a_dict[key]
    return new_dict

def create_tuple_list(node_indices: dict, node_attributes: dict, labels: list) -> list:
    node_atr_tuple_list = []
    cleaned_keys_dict = dict(node_indices)

    for key_idx in range(len(list(node_indices.keys()))):
        dict_to_add = {}

        for label_idx in range(len(labels)):
            dict_to_add[labels[label_idx]] = node_attributes[list(node_attributes.keys())[key_idx]][label_idx]
        
        node_atr_tuple_list.append((list(cleaned_keys_dict.keys())[key_idx], dict_to_add))
    
    return node_atr_tuple_list

def build_list_dict_attribute(list_of_attributes: list, labels: list) -> list:
    """
    build a list of dictionaries from a list of lists of attributes
    the length of the list of attributes must be the same as the length of the labels
    """

    # raise error if the length of the list of attributes is not the same as the length of the labels
    if len(list_of_attributes[0]) != len(labels):
        raise ValueError("The length of the list of attributes must be the same as the length of the labels")
    
    dict_list = []
    for paired_values in list_of_attributes:
        current_dict = {}
        
        for label_idx in range(len(labels)):
            current_dict[labels[label_idx]] = paired_values[label_idx]
            
        dict_list.append(current_dict)

    return dict_list

def edges_attr_list(node_list: list, attributes: list):
    """
    create a list of tuples (start, end, attributes)
    """

    tuples_node_attr = []
    
    for index in range(len(node_list)):
        tuples_node_attr.append((node_list[index][0], node_list[index][1], attributes[index]))
        
    return tuples_node_attr

def string_to_graph(a_string_graph):
    a_graph = json.loads(a_string_graph)
    return json_graph.node_link_graph(a_graph, directed=True, multigraph=False,
                                      attrs={"link": "edges", "source": "from", "target": "to"})


def graph_to_string(a_graph):
    jsongraph = json_graph.node_link_data(a_graph, attrs={"link": "edges", "source": "from", "target": "to"})
    return json.dumps(jsongraph, indent=4)