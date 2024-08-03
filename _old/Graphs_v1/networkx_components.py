import random

import networkx as nx
from flask import Flask
import ghhops_server as hs

from networkx_utils import *

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


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

    return graph_to_string(first_test)


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
    my_graph = string_to_graph(my_string_graph)

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

    node_graph.add_nodes_from(node_att_tuples)

    return graph_to_string(node_graph)


@hops.component(
    "/node_to_directed_graph",
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
    node_graph = DiGraph_GH()

    cleaned_attributes = clean_dict_datatype(all_attributes)

    node_att_tuples = create_tuple_list(node_index_tree, cleaned_attributes, labels)

    node_graph.add_nodes_from(node_att_tuples)

    return node_graph


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
def add_edge_graph(my_string_graph, connexion_indexes: dict, all_attributes: dict, labels: list):
    my_graph = string_to_graph(my_string_graph)
    my_graph = my_graph.to_directed()
    cleaned_attributes = clean_dict_datatype(all_attributes)

    pairs = list(connexion_indexes.values())
    paired_values = list(cleaned_attributes.values())

    super_dict = build_list_dict_attribute(paired_values, labels)
    new_graph = edges_attr_to_graph(my_graph, pairs, super_dict)

    return graph_to_string(new_graph)


@hops.component(
    "/read_nodes",
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
    my_graph = string_to_graph(my_string_graph)

    values_from_attributes, list_partitioner = read_nodes_attr_from_graph(my_graph, attributes, node_index)

    return values_from_attributes, list_partitioner


@hops.component(
    "/read_edges",
    name="read edges",
    nickname="read edges",
    description="Read all edge's attributes",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
        hs.HopsString("Attributes", "A", "Attributes you want to retrieve", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Edges", "E", "A collection of paired integers that describe an edge", hs.HopsParamAccess.TREE,
                       default=-1)
    ],
    outputs=[
        hs.HopsString("Values", "V", "Values for each node, for each attributes"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def read_node(my_string_graph, attributes, edge_indexes: dict):
    my_graph = string_to_graph(my_string_graph)
    nested_edge_indexes = dict_val_to_nested(edge_indexes)

    values_from_attributes = read_edges_attr_from_graph(my_graph, attributes, nested_edge_indexes)

    return values_from_attributes, len(attributes)


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
    my_graph = string_to_graph(my_string_graph)

    new_graph = remove_edges_constrains(my_graph, edge_attributes, edge_values)

    return graph_to_string(new_graph)


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
    my_graph = string_to_graph(my_string_graph)

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
    my_graph = string_to_graph(my_string_graph)

    actual_length = apart_length - 1
    if apart_length > 10:
        return [], []

    possible_targets = list(nx.all_simple_paths(my_graph, start_index, target_index, actual_length))

    if filter_lengths:
        possible_targets = [x for x in possible_targets if len(x) > actual_length - 1]

    flatten_list = [x for sublist in possible_targets for x in sublist]

    return flatten_list, [len(x) for x in possible_targets]


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
    my_graph = string_to_graph(my_string_graph)

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

    flatten_list = [x for sublist in possible_targets for x in sublist]
    return message, flatten_list, [len(x) for x in possible_targets]


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
    my_graph = string_to_graph(my_string_graph)

    if min_length < 0:
        min_length = max_length
    apart_lengths = [min_length, max_length]

    starting_positions = get_neighbors_constrained(my_graph, circulation_indexes, "is_horizontal", True)
    windows = get_nodes_with_attribute(my_graph, "window_count", val=1, more=False)
    priority_start = list(set.intersection(set(starting_positions), set(windows)))

    ordered_starting_positions = reorder_indexes(starting_positions, priority_start, front=True)
    circulation_less_graph = remove_nodes(my_graph, circulation_indexes)

    unused_graph, all_apartments = generate_apartments_v4(circulation_less_graph, ordered_starting_positions,
                                                          compacity_factor, apart_lengths, my_seed)
    new_graph = my_graph.copy()
    new_graph.remove_nodes_from(n for n in unused_graph if n in circulation_less_graph)

    new_string_graph = graph_to_string(new_graph)

    return new_string_graph, [x for sublist in all_apartments for x in sublist], [len(x) for x in all_apartments]


@hops.component(
    "/apartments_directedGraph",
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
    my_graph = string_to_graph(my_string_graph)

    if min_length < 0:
        min_length = max_length
    apart_lengths = [min_length, max_length]

    starting_positions = get_neighbors_constrained(my_graph, circulation_indexes, "is_horizontal", True)
    windows = get_nodes_with_attribute(my_graph, "window_count", val=1, more=False)
    priority_start = list(set.intersection(set(starting_positions), set(windows)))
    ordered_starting_positions = reorder_indexes(starting_positions, priority_start, front=True)

    circulation_less_graph = remove_nodes(my_graph, circulation_indexes)
    unused_graph, all_apartments = generate_apartments_v4(circulation_less_graph, ordered_starting_positions,
                                                          compacity_factor, apart_lengths, my_seed)

    return "", [x for sublist in all_apartments for x in sublist], [len(x) for x in all_apartments]


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
    my_graph = string_to_graph(my_string_graph)

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

    return graph_to_string(updating_graph), flat_apart_list, [len(x) for x in all_apartments]


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
    my_graph = string_to_graph(my_string_graph)

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


if __name__ == "__main__":
    app.run(debug=False)
