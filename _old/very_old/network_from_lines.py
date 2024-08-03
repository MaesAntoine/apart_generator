import random

from flask import Flask
import ghhops_server as hs
import networkx as nx
from networkx.readwrite import json_graph
import json

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


def clean_dict_datatype(a_dict: dict) -> dict:
    """Hops only accepts one datatype per input.
    We force the str datatype at input (convert if necessary),
    then convert back to either integer, float, bool or string"""

    for key_index in a_dict.keys():
        for elem in range(len(a_dict[key_index])):
            current_elem = a_dict[key_index][elem]

            if str(current_elem).isdigit():
                a_dict[key_index][elem] = int(current_elem)
            elif current_elem.replace('.', '', 1).isdigit() and current_elem.count('.') < 2:
                a_dict[key_index][elem] = float(current_elem)
            elif current_elem == "True":
                a_dict[key_index][elem] = True
            elif current_elem == "False":
                a_dict[key_index][elem] = False
            else:
                a_dict[key_index][elem] = str(current_elem)
    return a_dict


def remove_tree_brackets(a_dict: dict) -> dict:
    """remove the {} of the paths (that are the keys of the dictionary)"""
    new_dict = {}
    for key in a_dict.keys():
        try:
            new_dict[int(str(key)[1:-1])] = a_dict[key]  # try get integer key... but is it even good?
        except:
            new_dict[str(key)[1:-1]] = a_dict[key]
    return new_dict


def list_key_path_old(a_dict: dict) -> list:
    """create a list with sub lists for each paths"""
    temp_list = []
    for key in a_dict.keys():
        temp_list.append(key.split(";"))

    int_key_path_lists = [[int(i) for i in j] for j in temp_list]  # str to int
    return int_key_path_lists


def list_key_path(a_dict: dict) -> list:
    temp_list = []
    new_dict = {}

    # remove the brackets
    for key in a_dict.keys():
        new_dict[str(key)[1:-1]] = a_dict[key]

    # split and append
    for key in new_dict.keys():
        temp_list.append(key.split(";"))

    # str to int
    int_key_path_lists = [[int(i) for i in j] for j in temp_list]

    return int_key_path_lists


def shift_paths(a_dict: dict) -> dict:
    """shift paths with offset = -1 behaviour from Grasshopper, in python"""

    paths_as_list = list_key_path(a_dict)
    checked_list = []
    new_dict = {}

    for index in range(len(paths_as_list)):
        current_value = paths_as_list[index][0]
        current_key = list(a_dict.keys())[index]

        if current_value not in checked_list:
            new_dict[len(checked_list)] = a_dict[current_key]
            checked_list.append(current_value)

        else:
            current_list = new_dict[list(new_dict.keys())[len(checked_list) - 1]]
            current_list.append(a_dict[list(a_dict.keys())[index]][0])

    return new_dict


def add_to_networkx_1(graph, adjacency: dict, circulation: dict):
    for key in range(len(adjacency.keys())):
        counter = 0
        for val in adjacency[list(adjacency.keys())[key]]:
            circulation_value = circulation[list(circulation.keys())[key]][counter]
            graph.add_edge(key, val, is_horizontal=circulation_value)
            counter += 1

    return graph


def create_tuple_list(node_indexes: dict, node_attributes: dict, labels: list) -> list:
    node_atr_tuple_list = []

    cleaned_keys_dict = remove_tree_brackets(node_indexes)

    for key_idx in range(len(list(node_indexes.keys()))):
        dict_to_add = {}

        for label_idx in range(len(labels)):
            dict_to_add[labels[label_idx]] = node_attributes[list(node_attributes.keys())[key_idx]][label_idx]

        node_atr_tuple_list.append((list(cleaned_keys_dict.keys())[key_idx], dict_to_add))

    return node_atr_tuple_list


def read_nodes_attr_from_graph(a_graph, some_attributes: list, some_indexes: list) -> (list, int):
    # read specified attributes from selected nodes. If no nodes then read specified attributes from all nodes !

    if some_indexes[0] < 0:
        some_indexes = list(range(len(a_graph.nodes)))
    value_list = []

    for att in some_attributes:
        for node in range(len(some_indexes)):
            value_list.append(a_graph.nodes[node][att])

    return value_list, len(some_indexes)


def get_outer_branch_path(some_dict: dict) -> list:
    outer_branch = []

    bracketless_dict = remove_tree_brackets(some_dict)
    key_list = list(bracketless_dict.keys())

    for key in key_list:
        outer_branch.append(int(key.split(";")[0]))

    return outer_branch


def get_values(some_dict: dict) -> list:
    value_list = []
    list_of_values = list(some_dict.values())

    for value in list_of_values:
        value_list.append(value[0])

    return value_list


def build_list_dict_attribute(list_of_attributes: list, labels: list) -> list:
    dict_list = []

    for paired_values in list_of_attributes:
        current_dict = {}
        for label_idx in range(len(labels)):
            current_dict[labels[label_idx]] = paired_values[label_idx]
        dict_list.append(current_dict)

    return dict_list


def edges_attr_to_graph(my_graph, node_list: list, attributes: list):
    tuples_node_attr = []

    for index in range(len(node_list)):
        tuples_node_attr.append((node_list[index][0], node_list[index][1], attributes[index]))

    my_graph.add_edges_from(tuples_node_attr)

    return my_graph


def remove_edges_constrains(my_graph, edge_attributes, edge_values):
    """Remove all the edges from a graph where each value is matched for each attributes"""

    remove_list = []

    for attribute_index in range(len(edge_attributes)):
        attribute = edge_attributes[attribute_index]
        value = edge_values[attribute_index]

        remove_list = [edge for edge in my_graph.edges() if my_graph[edge[0]][edge[1]][attribute] == value]

    remove_list = list(set(remove_list))
    my_graph.remove_edges_from(remove_list)

    return my_graph


def get_all_possible_targets(my_graph, start_index: int, apart_length: int) -> list:
    """all reachable indexes are all the descendants at distance apart_length x - 2"""
    possible_descendant_distances = [x for x in range((apart_length - 1) % 2, apart_length, 2) if x > 0]
    possible_target = []

    for intermediate_distance in possible_descendant_distances:

        descendant = list(nx.descendants_at_distance(my_graph, start_index, intermediate_distance))
        if len(descendant) > 0:
            possible_target.append(descendant)
    return possible_target


def get_all_possible_targets_constrained(my_graph, start_index: int, apart_length: int, restrictions: list) -> list:
    """all reachable indexes are all the descendants at distance apart_length - 2 recursively
    then remove the ones that exist in restriction"""
    possible_descendant_distance = [x for x in range(apart_length % 2, apart_length + 2, 2) if x != 0]
    possible_target = []

    for intermediate_value in possible_descendant_distance:
        descendant = list(nx.descendants_at_distance(my_graph, start_index, intermediate_value))
        if len(descendant) > 0:
            possible_target.append(descendant)

    return [[x for x in sublist if x not in restrictions] for sublist in possible_target]


def get_neighbors(my_graph, circulation_indexes):
    """returns all the actual starting apart indexes from a graph and a list of circulation indexes"""
    neighbors = []
    for index in circulation_indexes:
        neighbors.append([x for x in my_graph.neighbors(index)])

    return [[x for x in sublist if x not in circulation_indexes] for sublist in neighbors]  # actual neighbors


def get_neighbors_constrained(my_graph, circulation_indexes, attribute, value):
    """returns a list of indexes that connect with circulation_indexes, where the edge connecting the two meets the
    attribute / value combo"""
    neighbors_list = []
    for index in circulation_indexes:
        index_neighbors = list(my_graph.neighbors(index))
        for neighbor in index_neighbors:
            if my_graph[index][neighbor][attribute] == str(value):
                neighbors_list.append(neighbor)

    return [x for x in neighbors_list if x not in circulation_indexes]  # actual neighbors


def remove_nodes(my_graph, circulation_indexes):
    """remove nodes from a graph (and therefore, the edges to those nodes)"""
    for index in circulation_indexes:
        my_graph.remove_node(index)
    return my_graph


def remove_nodes_inplace(my_graph, circulation_indexes):
    for index in circulation_indexes:
        my_graph.remove_node(index)


def filter_absent_indexes(my_graph, selected_sublist_index):
    """filter the values in selected_sublist_index that are not in the nodes"""
    return [x for x in selected_sublist_index if x in my_graph.nodes()]


def all_possible_targets_multiple_starts(a_graph, start_indexes, length):
    every_possible_targets = []
    for start in start_indexes:
        every_possible_targets.append(get_all_possible_targets(a_graph, start, length))

    return every_possible_targets


def recursive_remove_empty_nested_list(my_list):
    for index, value in enumerate(reversed(my_list)):
        if isinstance(value, list) and value != []:
            recursive_remove_empty_nested_list(value)
        elif isinstance(value, list) and len(value) == 0:
            my_list.remove(value)


def generate_apartments(a_graph, source_positions, compacity_factor, apart_lengths: list):
    min_length = apart_lengths[0]
    max_length = apart_lengths[-1]
    apartment_list = []
    possible_targets = all_possible_targets_multiple_starts(a_graph, source_positions, max_length)

    while len(source_positions) > 0:

        source = source_positions[0]
        targets_for_source = possible_targets[0]

        if len(targets_for_source) < 1 or source not in a_graph.nodes():
            source_positions.remove(source)
            possible_targets.remove(possible_targets[0])
            continue

        compacity = round((1 - compacity_factor) * (len(targets_for_source) - 1))
        target_compacity = targets_for_source[compacity]

        if len(target_compacity) < 1:
            possible_targets[0].remove(target_compacity)
            continue

        target = random.choice(target_compacity)

        if target not in a_graph.nodes():
            possible_targets[0][compacity].remove(target)
            continue

        possible_apartment = list(nx.all_simple_paths(a_graph, source, target, cutoff=max_length))

        if len(possible_apartment) < 1:
            possible_targets[0][compacity].remove(target)
            if len(possible_targets[0][compacity]) < 1:
                if possible_targets[0][compacity] in possible_targets:
                    possible_targets.remove(possible_targets[0][compacity])
            continue

        filtered_apartment = [x for x in possible_apartment if len(x) > min_length - 1]
        if len(filtered_apartment) < 1:
            possible_targets[0][compacity].remove(target)
            continue

        apartment = random.choice(filtered_apartment)
        apartment_list.append(apartment)
        a_graph = remove_nodes(a_graph, apartment)
        source_positions.remove(source)
        possible_targets.remove(possible_targets[0])

    return apartment_list


def generate_apartments_v4(a_graph, source_positions, compacity_factor, apart_lengths: list):
    copied_graph = a_graph.copy()
    min_length = apart_lengths[0]
    max_length = apart_lengths[-1]
    apartment_list = []
    possible_targets = all_possible_targets_multiple_starts(copied_graph, source_positions, max_length)

    while len(source_positions) > 0:

        source = source_positions[0]
        targets_for_source = possible_targets[0]

        if len(targets_for_source) < 1 or source not in copied_graph.nodes():
            source_positions.remove(source)
            possible_targets.remove(possible_targets[0])
            continue

        compacity = round((1 - compacity_factor) * (len(targets_for_source) - 1))
        target_compacity = targets_for_source[compacity]
        if len(target_compacity) < 1:
            possible_targets[0].remove(target_compacity)
            continue

        target = random.choice(target_compacity)

        if target not in copied_graph.nodes():
            possible_targets[0][compacity].remove(target)
            continue

        possible_apartment = list(nx.all_simple_paths(copied_graph, source, target, cutoff=max_length))

        if len(possible_apartment) < 1:
            possible_targets[0][compacity].remove(target)
            if len(possible_targets[0][compacity]) < 1 and possible_targets[0][compacity] in possible_targets:
                possible_targets.remove(possible_targets[0][compacity])
            continue

        filtered_apartment = [x for x in possible_apartment if len(x) > min_length - 1]
        if len(filtered_apartment) < 1:
            possible_targets[0][compacity].remove(target)
            continue

        filtered_apartment.sort(key=len)
        # print([len(x) for x in filtered_apartment])
        apartment = filtered_apartment[-1]

        apartment_list.append(apartment)
        a_graph = remove_nodes(copied_graph, apartment)
        source_positions.remove(source)
        possible_targets.remove(possible_targets[0])

    return copied_graph, apartment_list


def get_attributes(a_graph):
    node_attributes, edge_attributes = [], []

    nodex = list(a_graph.nodes(data=True))[0]
    for n in (nodex[1]):
        node_attributes.append(n)

    edgex = list(a_graph.edges(data=True))[-1]
    attr_dictionary = edgex[-1]
    for n in attr_dictionary.keys():
        edge_attributes.append(n)

    return node_attributes, edge_attributes


def apart_node_data(a_graph, apartments, attribute):
    results = []

    for aparts in apartments.values():
        current_apart = []

        for room in aparts:
            attr_dict = list(a_graph.nodes(data=True))[room][1]
            current_apart.append(attr_dict[attribute])

        results.append(current_apart)

    return results


def apart_edge_data(a_graph, apartments, attribute):
    results = []

    all_connexions = [[tuples[0], tuples[1]] for tuples in list(a_graph.edges(data=True))]

    for aparts in apartments.values():
        current_apart = []
        connexions = [[aparts[x], aparts[x + 1]] for x in range(len(aparts) - 1)]

        for connexion in connexions:
            connexion.sort()

            if connexion in all_connexions:
                connexion_index = all_connexions.index(connexion)
                selected_dict = list(a_graph.edges(data=True))[connexion_index]
                current_apart.append(selected_dict[-1][attribute])
            else:
                continue

        results.append(current_apart)
    return results


def flatten(nested_list) -> list:
    """flatten a 2 level list"""
    return [x for sublist in nested_list for x in sublist]


# ======================================================================================================================


@hops.component(
    "/ajd_to_net",
    name="build a net from an adjacency tree",
    nickname="adjacency to networkx",
    description="Creates a networkx form an adjacency datatree",
    inputs=[
        hs.HopsInteger("Adjacency", "A", "Adjacency as a tree", hs.HopsParamAccess.TREE),
        hs.HopsBoolean("Type", "T", "Type of connection: True=Horizontal, False=Vertical", hs.HopsParamAccess.TREE),
        hs.HopsBoolean("Write gml file", "W", "Write a gml file for external use")
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def ajd_to_net(adj_tree, circulation_type, write=False):
    empty_graph = nx.Graph()

    first_test = add_to_networkx_1(empty_graph, adj_tree, circulation_type)

    if write:
        nx.write_gml(first_test, "first_test.gml")

    return json_graph.node_link_data(
        first_test, {"link": "edges", "source": "from", "target": "to"}
    )


@hops.component(
    "/path_index",
    name="path indexes",
    nickname="path from indexes",
    description="Finds a path (if exists) between INDEXES A and B",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Source", "A", "Source index", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Target", "B", "Target index", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Max paths", "M", "Maximum number of paths (!! can be massively huge). Set -1 for all")
    ],
    outputs=[
        hs.HopsInteger("Single shortest path's indexes", "S", "Path as a list of indexes"),
        hs.HopsInteger("All path's indexes", "A", "All the paths in a single mega list"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def ajd_to_net(my_string_graph, source_index, target_index, maximum_paths=10):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    # nx.all_shortest_paths(my_graph, source_index, target_index) # is a generator
    single_path = nx.shortest_path(my_graph, source_index, target_index)

    my_dictionary = {}
    index = 0

    if maximum_paths == -1:
        for key in nx.all_shortest_paths(my_graph, source_index, target_index):
            my_dictionary[index] = key
            index += 1
    else:
        for key in nx.all_shortest_paths(my_graph, source_index, target_index):
            if index < maximum_paths:
                my_dictionary[index] = key
                index += 1

    for key in nx.all_shortest_paths(my_graph, source_index, target_index):
        if index < maximum_paths:
            my_dictionary[index] = key
            index += 1
        break

    all_path = []
    for key in my_dictionary.keys():
        for j in my_dictionary[key]:
            all_path.append(j)

    return single_path, all_path, len(single_path)


@hops.component(
    "/node_to_graph",
    name="node graph",
    nickname="node to graph",
    description="Creates a graph from nodes only, no edges",
    inputs=[
        hs.HopsInteger("Node indexes", "Ni", "All node indexes", hs.HopsParamAccess.TREE),
        hs.HopsString("Attributes", "Att", "All the attributes for a corresponding index", hs.HopsParamAccess.TREE),
        hs.HopsString("Labels", "L", "Labels for each attributes", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def node_to_graph(node_index_tree: dict, all_attributes: dict, labels: list):
    node_graph = nx.Graph()

    cleaned_attributes = clean_dict_datatype(all_attributes)

    node_att_tuples = create_tuple_list(node_index_tree, cleaned_attributes, labels)
    print(node_att_tuples)

    node_graph.add_nodes_from(node_att_tuples)

    return json_graph.node_link_data(
        node_graph, {"link": "edges", "source": "from", "target": "to"}
    )


@hops.component(
    "/add_edge_graph",
    name="add edge graph",
    nickname="edge graph",
    description="Add edges with attributes to an existing graph",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to add edges with attributes to", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Node indexes", "Ni", "All node indexes", hs.HopsParamAccess.TREE),
        hs.HopsString("Attributes", "Att", "All the attributes for a corresponding index", hs.HopsParamAccess.TREE),
        hs.HopsString("Labels", "L", "Labels for each attributes", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def add_edge_graph(my_string_graph, node_index_tree: dict, all_attributes: dict, labels: list):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    first_index_list = get_outer_branch_path(node_index_tree)
    second_index_list = get_values(node_index_tree)

    pairs = [[first_index_list[i], second_index_list[i]] for i in range(len(first_index_list))]
    paired_values = list(all_attributes.values())

    super_dict = build_list_dict_attribute(paired_values, labels)
    new_graph = edges_attr_to_graph(my_graph, pairs, super_dict)

    return json_graph.node_link_data(
        new_graph, {"link": "edges", "source": "from", "target": "to"}
    )


@hops.component(
    "/read_node",
    name="read node",
    nickname="read node",
    description="Read a node's info",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsString("Attributes", "A", "Attributes you want to retrieve", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Index", "I", "Indexes to read data from", hs.HopsParamAccess.LIST, default=-1)
    ],
    outputs=[
        hs.HopsString("Values", "V", "Values for each node, for each attributes"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def read_node(my_string_graph, attributes, node_index):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    values_from_attributes, list_partitioner = read_nodes_attr_from_graph(my_graph, attributes, node_index)

    return values_from_attributes, list_partitioner


@hops.component(
    "/remove_edges",
    name="remove edges",
    nickname="remove edges",
    description="Remove edges based on attribute == value pairs. \nNote: EA and EV need to be of same length",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsString("Edge attribute", "EA", "Edge attribute constrain", hs.HopsParamAccess.LIST),
        hs.HopsString("Edge value", "EV", "Edge value of the attribute to REMOVE from graph", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def read_node(my_string_graph, edge_attributes, edge_values):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    new_graph = remove_edges_constrains(my_graph, edge_attributes, edge_values)

    return json_graph.node_link_data(
        new_graph, {"link": "edges", "source": "from", "target": "to"}
    )


@hops.component(
    "/one_apartment",
    name="One apartment",
    nickname="One apartment",
    description="Create one apartment and retrieve values for certain attributes",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Index", "I", "Indexes to read data from", hs.HopsParamAccess.ITEM, default=0),
        hs.HopsInteger("Length", "L", "Length of the apartment", hs.HopsParamAccess.ITEM, default=5),
        hs.HopsString("Attributes", "A", "Attributes you want to retrieve", hs.HopsParamAccess.LIST, default=[])
    ],
    outputs=[
        hs.HopsInteger("Furthest reachable indexes", "F", "Indexes of all the apartment blocks"),
        hs.HopsInteger("All reachable indexes", "A", "All reachable indexes from start index and length"),
        hs.HopsString("Values", "V", "Values for each node, for each attributes")
    ]
)
def one_apart(my_string_graph, start_index, apart_length, attributes):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    possible_targets = list(nx.descendants_at_distance(my_graph, start_index, apart_length))

    return possible_targets, [], []


@hops.component(
    "/random_apartment_AB",
    name="random apartment",
    nickname="random apart",
    description="Create a random apart from one index to another with a minimum size",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Start", "A", "Index to start from", hs.HopsParamAccess.ITEM, default=0),
        hs.HopsInteger("Target", "B", "Index to aim for", hs.HopsParamAccess.ITEM, default=4),
        hs.HopsInteger("Length", "L", "Length of the apartment", hs.HopsParamAccess.ITEM, default=5),
        hs.HopsBoolean("Filter", "F", "Filter the apartments that are smaller than Length", hs.HopsParamAccess.ITEM,
                       default=True)
    ],
    outputs=[
        hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def one_apart(my_string_graph, start_index, target_index, apart_length, filter_lengths):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    actual_length = apart_length - 1
    if apart_length > 10:
        return [], []

    possible_targets = list(nx.all_simple_paths(my_graph, start_index, target_index, actual_length))

    if filter_lengths:
        possible_targets = [x for x in possible_targets if len(x) > actual_length - 1]

    flatten = [x for sublist in possible_targets for x in sublist]

    return flatten, [len(x) for x in possible_targets]


@hops.component(
    "/random_apartment",
    name="random apartment",
    nickname="random apart",
    description="Create a random apart from one index to another with a minimum size",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Start", "A", "Index to start from", hs.HopsParamAccess.ITEM, default=0),
        hs.HopsInteger("Length", "L", "Length of the apartment", hs.HopsParamAccess.ITEM, default=5),
        hs.HopsNumber("Compacity", "C", "Simple compacity estimation before creation of the apartment",
                      hs.HopsParamAccess.ITEM, default=0.0),
        hs.HopsNumber("Targets", "T", "Scroll through all the possible targets for your compacity and length value",
                      hs.HopsParamAccess.ITEM, default=1.0),
        hs.HopsBoolean("Filter", "F", "Filter the apartments that are smaller than Length",
                       hs.HopsParamAccess.ITEM, default=True),
    ],
    outputs=[
        hs.HopsString("Message", "M", "Possible error messages"),
        hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def one_apart(my_string_graph, start_index, apart_length, compacity_factor, target_factor, filter_lengths):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    if apart_length > 12:
        message = "Length value is more likely to high for this component"
        return message, [], []
    else:
        message = "Everything's fine"

    targets = get_all_possible_targets(my_graph, start_index, apart_length)
    compacity = round((1 - compacity_factor) * (len(targets) - 1))
    target_index = targets[compacity][round(target_factor * (len(targets[compacity]) - 1))]
    possible_targets = list(nx.all_simple_paths(my_graph, start_index, target_index, apart_length))

    if filter_lengths:
        possible_targets = [x for x in possible_targets if len(x) > apart_length - 1]

    flatten = [x for sublist in possible_targets for x in sublist]
    return message, flatten, [len(x) for x in possible_targets]


@hops.component(
    "/many_apartments",
    name="many apartments",
    nickname="many apart",
    description="Create many random apartments",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Circulation indexes", "I", "Indexes of circulation blocks", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Length", "L", "Length of the apartment", hs.HopsParamAccess.ITEM, default=5),
        hs.HopsNumber("Compacity", "C", "Simple compacity estimation before creation of the apartment",
                      hs.HopsParamAccess.ITEM, default=0.0),
        hs.HopsInteger("Seed", "S", "Seed to shuffle the generation", hs.HopsParamAccess.ITEM, default=2),
        hs.HopsBoolean("Filter", "F", "Filter the apartments that are smaller than Length",
                       hs.HopsParamAccess.ITEM, default=True)
    ],
    outputs=[
        hs.HopsString("Message", "M", "Possible error messages"),
        hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def many_aparts(my_string_graph, circulation_indexes, apart_length, compacity_factor, my_seed, filter_lengths):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    actual_starting_indexes = get_neighbors(my_graph, circulation_indexes)
    circulation_less_graph = remove_nodes(my_graph, circulation_indexes)
    random.seed(my_seed)
    all_apartments = []

    updating_graph = circulation_less_graph
    count = 1
    remaining_attempts = 10

    print("--------------------------------------------------------------------------------------------------------")
    print("BEFORE GENERATION")
    print("actual possible starts", actual_starting_indexes)
    print("Maximum number of apartment possible (but don't dream too much): ", len(actual_starting_indexes), "\n")
    print("STARTING APARTMENT GENERATION\n")

    while len(actual_starting_indexes) > 0:
        if remaining_attempts < 1:
            print("It seems that I cannot find a way to generate apartment", count)
            print("!!! BREAKING THE LOOP, can't find a viable path for the previously named apartment !!!\n")
            print("--------------------")
            break

        print("--------------------")
        print("Initiating generation of apartment number ", count)
        print("Possible starting indexes are ", actual_starting_indexes)

        selected_sublist_index = actual_starting_indexes[0]
        possible_sublist_index = filter_absent_indexes(my_graph, selected_sublist_index)
        if len(possible_sublist_index) < 1:  # exit while loop if no more start positions
            print("!!! BREAKING THE LOOP, ALL START INDEXES ARE ALREADY USED !!!\n")
            print("--------------------")
            break

        print("... Checking if numbers should be removed from the sublist of start indexes...")
        if len(selected_sublist_index) != len(possible_sublist_index):
            print("Items removed. New sublist is ", possible_sublist_index)
        else:
            print("No item removed.")

        current_start_index = random.choice(possible_sublist_index)
        print("First sublist selected. Picking a starting index in ", possible_sublist_index)
        print("Selected starting index is ", current_start_index, "\n")

        possible_targets = get_all_possible_targets(my_graph, current_start_index, apart_length - 1)
        if len(possible_targets) < 1:
            print("!!! BREAKING THE LOOP, THERE IS NO PATH POSSIBLE BETWEEN THE TWO !!!\n")
            print("--------------------")
            break

        print("The possible targets for ", current_start_index, " and length", apart_length, " are :")
        print(possible_targets)

        compacity = round((1 - compacity_factor) * (len(possible_targets) - 1))
        print("Compacity factor = ", compacity)
        reduced_possible_targets = possible_targets[compacity]
        print("Selected group of targets is ", reduced_possible_targets, "... Selecting a target randomly")
        selected_target = random.choice(reduced_possible_targets)
        print("Selected target is ", selected_target, "\n")

        possible_apartments = list(nx.all_simple_paths(my_graph, current_start_index, selected_target, apart_length))
        print("Possible apartments between ", current_start_index, " and ", selected_target, " are:")

        if filter_lengths:
            possible_apartments = [x for x in possible_apartments if len(x) > apart_length - 1]
            if len(possible_apartments) < 1:
                print("!!! SKIPPING THE CURRENT LOOP, NO APARTMENT !!!\n")
                reduced_possible_targets.remove(selected_target)
                remaining_attempts -= 1
                print("--------------------")
                continue

            print(possible_apartments, "\n")

        selected_apart = random.choice(possible_apartments)
        print("Randomly selected apart is ", selected_apart)
        all_apartments.append(selected_apart)

        print("Current list of apartments :")
        print(all_apartments)
        print("--------------------")
        updating_graph = remove_nodes(updating_graph, selected_apart)
        count += 1

        actual_starting_indexes.pop(0)

    print("END OF GENERATION. TOTAL APARTMENT GENERATED: ", len(all_apartments))
    print("--------------------------------------------------------------------------------------------------------")
    flat_apart_list = [x for sublist in all_apartments for x in sublist]

    return json_graph.node_link_data(
        updating_graph, {"link": "edges", "source": "from", "target": "to"}
    ), flat_apart_list, [len(x) for x in all_apartments]


@hops.component(
    "/apartment_builder",
    name="apartment builder",
    nickname="apart builder",
    description="Build apartment randomly with compacity and length constrains",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Circulation indexes", "I", "Indexes of circulation blocks", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Max Length", "M", "Max length of the apartment", hs.HopsParamAccess.ITEM, default=5),
        hs.HopsInteger("Min Length", "m", "min length of the apartment", hs.HopsParamAccess.ITEM, default=-1),
        hs.HopsNumber("Compacity", "C", "Simple compacity estimation before creation of the apartment",
                      hs.HopsParamAccess.ITEM, default=0),
        hs.HopsInteger("Seed", "S", "Seed to shuffle the generation", hs.HopsParamAccess.ITEM, default=2)
    ],
    outputs=[
        hs.HopsString("New graph", "G", "Newly created graph"),
        hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def apartment_builder(my_string_graph, circulation_indexes, max_length, min_length, compacity_factor, my_seed):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    if min_length < 0:
        min_length = max_length
    apart_lengths = [min_length, max_length]

    random.seed(my_seed)
    starting_positions = get_neighbors_constrained(my_graph, circulation_indexes, "is_horizontal", True)
    circulation_less_graph = remove_nodes(my_graph, circulation_indexes)

    unused_graph, all_apartments = generate_apartments_v4(circulation_less_graph, starting_positions, compacity_factor,
                                                          apart_lengths)
    new_graph = my_graph.copy()
    new_graph.remove_nodes_from(n for n in unused_graph if n in circulation_less_graph)

    new_string_graph = json_graph.node_link_data(
               new_graph, {"link": "edges", "source": "from", "target": "to"}
           )

    return new_string_graph, [x for sublist in all_apartments for x in sublist], [len(x) for x in all_apartments]


@hops.component(
    "/graph_attributes",
    name="evaluate apartments",
    nickname="eval apart",
    description="Evaluate the apartments based on one of their attributes",
    inputs=[
        hs.HopsString("Graph", "G", "Graph on which the apartments have been generated from", hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsString("Node attribute list", "N", "List of all the available NODE attributes"),
        hs.HopsString("Edge attribute list", "E", "List of all the available NODE attributes")
    ]
)
def graph_attributes(my_string_graph):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    return get_attributes(my_graph)


@hops.component(
    "/evaluate_apartments",
    name="evaluate apartments",
    nickname="eval apart",
    description="Evaluate the apartments based on one of their attributes",
    inputs=[
        hs.HopsString("Graph", "G", "Graph on which the apartments have been generated from", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Apartments indexes", "A", "All the apartments indexes as datatree", hs.HopsParamAccess.TREE,
                       default=-1),
        hs.HopsString("Attribute", "Att", "Attribute to evaluate the apartments with", hs.HopsParamAccess.ITEM,
                      default="uh")
    ],
    outputs=[
        hs.HopsString("Values", "Nv", "Values per blocks, for the attribute entered"),
        hs.HopsInteger("Value list partitioner", "Np", "Numbers to build Datatree from flat list")
    ]
)
def evaluate_apartments(my_string_graph, generated_apartments: dict, selected_attribute):
    my_json_graph = json.loads(my_string_graph)
    my_graph = json_graph.node_link_graph(my_json_graph, directed=False, multigraph=False,
                                          attrs={"link": "edges", "source": "from", "target": "to"})

    apart_values = []
    partitioner = []

    node_attributes, edge_attributes = get_attributes(my_graph)

    if selected_attribute in node_attributes:
        apart_values = apart_node_data(my_graph, generated_apartments, selected_attribute)

    else:
        apart_values = apart_edge_data(my_graph, generated_apartments, selected_attribute)  # <--- here !!!

    partitioner = [len(x) for x in apart_values]
    apart_values = flatten(apart_values)

    return apart_values, partitioner


@hops.component(
    "/test",
    name="test",
    nickname="test",
    description="test stuff",
    inputs=[
        hs.HopsString("Graph", "G", "Graph on which the apartments have been generated from", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Apartments indexes", "A", "All the apartments indexes as datatree", hs.HopsParamAccess.TREE,
                       default=-1),
        hs.HopsString("Attribute", "Att", "Attribute to evaluate the apartments with", hs.HopsParamAccess.ITEM,
                      default="uh")
    ],
    outputs=[
        hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
        hs.HopsString("Block value", "Bv", "Values per blocks, for the attribute entered"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from a flat list")
    ]
)
def test(my_string_graph, generated_apartments: dict, selected_attribute):
    pass


if __name__ == "__main__":
    app.run(debug=False)
