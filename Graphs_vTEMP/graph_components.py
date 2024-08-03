
from flask import Flask
import ghhops_server as hs
import json
from graph_generate import generate_apartments
import networkx as nx

from graph_utils import remove_tree_brackets, clean_dict_datatype, create_tuple_list, graph_to_string, string_to_graph, get_type
from graph_utils import remove_vertical_at_circulation, get_circulation_distance, merge_two_dict,build_list_dict_attribute,edges_attr_list
from graph_utils import read_nodes_attr_from_graph, remove_edges_constrains, dict_val_to_nested, read_edges_attr_from_graph, get_all_possible_targets
from graph_utils import get_score_circulation_distance_aparts, check_defaults_generation_settings, remove_nodes_with_attribute, compute_circulation_minmax_precise, circulation_estimation
from graph_utils import get_best_apartment_dict, get_apartments_proportions, graph_copy_from_indices, compute_post_generation_rankings, flatten_gen

from graph_data import CIRCULATION_DISTANCE, BLOCK_TYPE, min_compacity, max_compacities_dict



# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


"""

██████╗░██╗░░░██╗██╗██╗░░░░░██████╗░   ░░░░██╗   ███╗░░░███╗░█████╗░██████╗░██╗███████╗██╗░░░██╗   ░██████╗░██████╗░░█████╗░██████╗░██╗░░██╗
██╔══██╗██║░░░██║██║██║░░░░░██╔══██╗   ░░░██╔╝   ████╗░████║██╔══██╗██╔══██╗██║██╔════╝╚██╗░██╔╝   ██╔════╝░██╔══██╗██╔══██╗██╔══██╗██║░░██║
██████╦╝██║░░░██║██║██║░░░░░██║░░██║   ░░██╔╝░   ██╔████╔██║██║░░██║██║░░██║██║█████╗░░░╚████╔╝░   ██║░░██╗░██████╔╝███████║██████╔╝███████║
██╔══██╗██║░░░██║██║██║░░░░░██║░░██║   ░██╔╝░░   ██║╚██╔╝██║██║░░██║██║░░██║██║██╔══╝░░░░╚██╔╝░░   ██║░░╚██╗██╔══██╗██╔══██║██╔═══╝░██╔══██║
██████╦╝╚██████╔╝██║███████╗██████╔╝   ██╔╝░░░   ██║░╚═╝░██║╚█████╔╝██████╔╝██║██║░░░░░░░░██║░░░   ╚██████╔╝██║░░██║██║░░██║██║░░░░░██║░░██║
╚═════╝░░╚═════╝░╚═╝╚══════╝╚═════╝░   ╚═╝░░░░   ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚═╝╚═╝░░░░░░░░╚═╝░░░   ░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝

"""

@hops.component(
    "/node_to_graph",
    name="node graph",
    nickname="node to graph",
    description="Creates a graph from nodes only, no edges",
    inputs=[
        hs.HopsInteger("Node indices", "Ni", "All node indices", hs.HopsParamAccess.TREE),
        hs.HopsString("Attributes", "Att", "All the attributes for a corresponding index", hs.HopsParamAccess.TREE),
        hs.HopsString("Labels", "L", "Labels for each attributes", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def node_to_graph(node_index_tree: dict, all_attributes: dict, labels: list):
    
    node_graph = nx.Graph()
    
    node_indices = remove_tree_brackets(node_index_tree)
    cleaned_attributes = clean_dict_datatype(all_attributes)
    node_att_tuples = create_tuple_list(node_indices, cleaned_attributes, labels)

    node_graph.add_nodes_from(node_att_tuples)

    return graph_to_string(node_graph)


@hops.component(
    "/graph_set_circulation",
    name="set circulation blocks",
    nickname="block circulations",
    description="Sets the circulation of the graph",
    inputs=[
        hs.HopsString("Graph", "G", "Graph as a string"),
        hs.HopsInteger("Indices", "i", "Indexes of the nodes that are for circulation", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def graph_set_circulation(graph_string: str, circulation_indices):
    
    node_graph = string_to_graph(graph_string)
    
    type_dictionary = get_type(node_graph, circulation_indices)
    new_graph = remove_vertical_at_circulation(node_graph, circulation_indices)
    distance_dictionary = get_circulation_distance(new_graph, circulation_indices)
    merge_dict = merge_two_dict(type_dictionary, distance_dictionary)
    
    node_att_tuples = create_tuple_list(new_graph.nodes, merge_dict, [BLOCK_TYPE, CIRCULATION_DISTANCE])
    node_graph.add_nodes_from(node_att_tuples)
    
    return graph_to_string(node_graph)
    

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
def add_edge_graph(graph_string, connexion_indexes: dict, all_attributes: dict, labels: list):
    
    node_graph = string_to_graph(graph_string)
    new_graph = node_graph.to_directed()
    cleaned_attributes = clean_dict_datatype(all_attributes)
    
    pairs = list(connexion_indexes.values())
    paired_values = list(cleaned_attributes.values())
    
    list_of_dict_from_values = build_list_dict_attribute(paired_values, labels)
    tuples_node_attr = edges_attr_list(pairs, list_of_dict_from_values)

    new_graph.add_edges_from(tuples_node_attr)
    
    return graph_to_string(new_graph)


@hops.component(
    "/remove_edges",
    name="remove edges",
    nickname="remove edges",
    description="Remove edges based on attribute key, value pairs. \nNote: EA and EV need to be of same length",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to remove edges from", hs.HopsParamAccess.ITEM),
        hs.HopsString("Edge attribute", "EK", "Edge attribute Key", hs.HopsParamAccess.LIST),
        hs.HopsString("Edge value", "EV", "Edge attribute Value to REMOVE from graph", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsString("Graph as json", "G", "Network as a json string")
    ]
)
def read_node(my_string_graph, edge_attributes, edge_values):
    my_graph = string_to_graph(my_string_graph)
    
    new_graph = remove_edges_constrains(my_graph, edge_attributes, edge_values)
    
    return graph_to_string(new_graph)



"""
███████╗██╗░░██╗████████╗██████╗░░█████╗░░█████╗░████████╗   ███████╗██████╗░░█████╗░███╗░░░███╗   ░██████╗░██████╗░░█████╗░██████╗░██╗░░██╗
██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝   ██╔════╝██╔══██╗██╔══██╗████╗░████║   ██╔════╝░██╔══██╗██╔══██╗██╔══██╗██║░░██║
█████╗░░░╚███╔╝░░░░██║░░░██████╔╝███████║██║░░╚═╝░░░██║░░░   █████╗░░██████╔╝██║░░██║██╔████╔██║   ██║░░██╗░██████╔╝███████║██████╔╝███████║
██╔══╝░░░██╔██╗░░░░██║░░░██╔══██╗██╔══██║██║░░██╗░░░██║░░░   ██╔══╝░░██╔══██╗██║░░██║██║╚██╔╝██║   ██║░░╚██╗██╔══██╗██╔══██║██╔═══╝░██╔══██║
███████╗██╔╝╚██╗░░░██║░░░██║░░██║██║░░██║╚█████╔╝░░░██║░░░   ██║░░░░░██║░░██║╚█████╔╝██║░╚═╝░██║   ╚██████╔╝██║░░██║██║░░██║██║░░░░░██║░░██║
╚══════╝╚═╝░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝░╚════╝░░░░╚═╝░░░   ╚═╝░░░░░╚═╝░░╚═╝░╚════╝░╚═╝░░░░░╚═╝   ░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝
"""

@hops.component(
	"/read_nodes_attr_value",
	name="read nodes",
	nickname="read nodes",
	description="Read all the attribute's values from every desired nodes and keys",
	inputs=[
        hs.HopsString("Graph", "G", "Graph to read node values from", hs.HopsParamAccess.ITEM),
        hs.HopsString("Attributes", "A", "Attributes keys", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Index", "i", "Indices of the nodes", hs.HopsParamAccess.LIST, default=-1)
    ],
    outputs=[
        hs.HopsString("Values", "V", "Values for each node, for each attributes"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def read_nodes_attr_value(my_string_graph, attributes, node_index):
    my_graph = string_to_graph(my_string_graph)
    
    values_from_attributes, list_partitioner = read_nodes_attr_from_graph(my_graph, attributes, node_index)
    
    return values_from_attributes, list_partitioner


@hops.component(
    "/read_edges_attr_values",
    name="read edges",
    nickname="read edges",
    description="Read all the attribute's values from every desired edges and keys",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to read edge values from", hs.HopsParamAccess.ITEM),
        hs.HopsString("Attributes", "A", "Attributes keys", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Edges", "E", "A collection of paired integers that describe an edge", hs.HopsParamAccess.TREE,
                       default=-1)
    ],
    outputs=[
        hs.HopsString("Values", "V", "Values for each node, for each attributes"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list")
    ]
)
def read_edges_attr_values(my_string_graph, attributes, edge_indexes: dict):
    my_graph = string_to_graph(my_string_graph)
    nested_edge_indices = dict_val_to_nested(edge_indexes)

    values_from_attributes = read_edges_attr_from_graph(my_graph, attributes, nested_edge_indices)

    return values_from_attributes, len(attributes)


@hops.component(
    "/explode_graph",
    name="explode graph",
    nickname="explode graph",
    description="Get all nodes and edges from graph (no data)",
    inputs=[
        hs.HopsString("Graph", "G", "Graph to read edge values from", hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsInteger("Nodes", "N", "Graph's nodes"),
        hs.HopsInteger("Edges", "E", "Graph's edges"),
        hs.HopsInteger("List partitioner", "P", "Number to partition Edges")
    ]
)
def read_edges_attr_values(my_string_graph):
    my_graph = string_to_graph(my_string_graph)
    
    nodes = list(my_graph.nodes)
    edges = [x for sublist in list(my_graph.edges) for x in sublist] # flatten list of tuples
    
    return nodes, edges, 2


@hops.component(
    "/test_bloc_distance",
    name="test_distance",
    nickname="test_distance",
    description="Test the distance from circulation constant",
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
        hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
        hs.HopsInteger("Distances from circulations", "D", "Distances from circulations"),
        hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list"),
        hs.HopsInteger("Distance sums", "S", "Sum of the distances from circulations"),
        hs.HopsInteger("Target", "T", "Target selected"),
        hs.HopsString("MinMax", "M", "Min and Max distance values from aparts") 
    ]
)
def test_bloc_distance(my_string_graph, start_index, apart_length, compacity_factor, target_factor, filter_lengths):
    my_graph = string_to_graph(my_string_graph)
      
    if apart_length > 30:
       raise Exception("ERROR: apart_length is too high")

    targets = get_all_possible_targets(my_graph, start_index, apart_length)
    compacity = round((1 - compacity_factor) * (len(targets) - 1))
    target_index = targets[compacity][round(target_factor * (len(targets[compacity]) - 1))]
    possible_aparts = list(nx.all_simple_paths(my_graph, start_index, target_index, apart_length))
    
    proximity, proximity_sum = get_score_circulation_distance_aparts(my_graph, possible_aparts)
    
    flatten_proximity = [x for sublist in proximity for x in sublist]
    flatten_list = [x for sublist in possible_aparts for x in sublist]
    
    return flatten_list, flatten_proximity, [len(x) for x in possible_aparts], proximity_sum, target_index, [min(flatten_proximity), max(flatten_proximity)]
"""

░██████╗███████╗████████╗████████╗██╗███╗░░██╗░██████╗░░██████╗
██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝██║████╗░██║██╔════╝░██╔════╝
╚█████╗░█████╗░░░░░██║░░░░░░██║░░░██║██╔██╗██║██║░░██╗░╚█████╗░
░╚═══██╗██╔══╝░░░░░██║░░░░░░██║░░░██║██║╚████║██║░░╚██╗░╚═══██╗
██████╔╝███████╗░░░██║░░░░░░██║░░░██║██║░╚███║╚██████╔╝██████╔╝
╚═════╝░╚══════╝░░░╚═╝░░░░░░╚═╝░░░╚═╝╚═╝░░╚══╝░╚═════╝░╚═════╝░
"""

@hops.component(
    "/ranking_settings",
    name="ranking settings",
    nickname="ranking settings",
    description="Set the settings for the rankings of the generated apartments",
    inputs=[
        hs.HopsNumber("Compacity", "compacity", "Compacity of the generated apartments\nValue between 0 and 1", hs.HopsParamAccess.ITEM, default=-1),
        hs.HopsNumber("Circulation", "circulation", "Circulation of the generated apartments\nValue between 0 and 1", hs.HopsParamAccess.ITEM, default=-1),
        hs.HopsNumber("Windows", "windows", "Windows count of the generated apartments\nValue between 0 and 1", hs.HopsParamAccess.ITEM, default=-1)
    ],
    outputs=[
    hs.HopsString("Ranking settings", "RS", "Ranking settings")
    ]
)
def generation_settings(compacity, circulation, windows):

    ranking_settings = {}
    
    # add settings to the dictionary if they are not -1
    if compacity != -1:
        ranking_settings["_compacity"] = compacity
    if circulation != -1:
        ranking_settings["_circulation"] = circulation
    if windows != -1:
        ranking_settings["_windows"] = windows


    return json.dumps(ranking_settings, indent=4)


@hops.component(
    "/generation_settings",
    name="generation settings",
    nickname="generation settings",
    description="Set the settings for the generation of the apartments",
    inputs=[
        hs.HopsInteger("Max targets", "max_targets", "Max targets to compute for each starting nodes", hs.HopsParamAccess.ITEM, default=5),
        hs.HopsInteger("Max paths", "max_paths", "Max paths to compute between each starting nodes and each targets", hs.HopsParamAccess.ITEM, default=10),
        hs.HopsBoolean("Fill", "fill", "Activate second pass after a generation, to try to fill gaps. Note that the proportions it will still try to satisfy the desired proportions",
                        hs.HopsParamAccess.ITEM, default=True),
        hs.HopsBoolean("Fast circulation", "fast_circulation", "Activate fast circulation approximation (min and max value to evaluate the apartments",
                        hs.HopsParamAccess.ITEM, default=True)
    ],
    outputs=[
    hs.HopsString("Generation settings", "GS", "Generation settings")
    ]
)
def generation_settings(max_targets, max_paths, fill, fast_circulation):
    
        generation_settings = {"max_targets": max_targets, "max_paths": max_paths, "fill": fill, "fast_circulation": fast_circulation}
    
        return json.dumps(generation_settings, indent=4)




"""
██████╗░██╗░░░██╗██╗██╗░░░░░██████╗░   ░█████╗░██████╗░░█████╗░██████╗░████████╗░██████╗
██╔══██╗██║░░░██║██║██║░░░░░██╔══██╗   ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
██████╦╝██║░░░██║██║██║░░░░░██║░░██║   ███████║██████╔╝███████║██████╔╝░░░██║░░░╚█████╗░
██╔══██╗██║░░░██║██║██║░░░░░██║░░██║   ██╔══██║██╔═══╝░██╔══██║██╔══██╗░░░██║░░░░╚═══██╗
██████╦╝╚██████╔╝██║███████╗██████╔╝   ██║░░██║██║░░░░░██║░░██║██║░░██║░░░██║░░░██████╔╝
╚═════╝░░╚═════╝░╚═╝╚══════╝╚═════╝░   ╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░╚═════╝░
"""
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


    possible_targets = list(nx.all_simple_paths(my_graph, start_index, target_index, actual_length))

    if filter_lengths:
        possible_targets = [x for x in possible_targets if len(x) > actual_length - 1]

    flatten_list = [x for sublist in possible_targets for x in sublist]

    return flatten_list, [len(x) for x in possible_targets]


@hops.component(
"/generate_v3",
name="generate apartment v3",
nickname="generate apart v3",
description="Generate apartments with a slower but better algorithm",
inputs=[
hs.HopsString("Graph", "G", "Graph to perform a path finding", hs.HopsParamAccess.ITEM),
hs.HopsString("Apartment proportions", "Ap", "Desired apartment proportions and sizes", hs.HopsParamAccess.ITEM),
hs.HopsString("Generation settings", "Gs", "Settings to tweak the generative algorithm", hs.HopsParamAccess.ITEM),
hs.HopsString("Ranking settings", "Rs", "Settings to tweak the ranking algorithm", hs.HopsParamAccess.ITEM),
hs.HopsInteger("Floor count", "F", "Floor count limit. Note: No apartment are allowed go down"),
hs.HopsInteger("Iterations", "I", "Iteration count\nEvery iteration should produce different results, the best one will be kept")
    ],
outputs=[
hs.HopsString("New graph", "G", "Newly created graph"),
hs.HopsString("All rankings", "R", "All rankings"),
hs.HopsInteger("Apart indexes", "A", "Indexes of all the apartment blocks"),
hs.HopsInteger("List partitioner", "P", "Number to build the datatree from the mega list"),
hs.HopsString("Proportions", "P", "Proportions of the generated apartments")
    ]
)
def apartment_builder(my_string_graph, desired_proportions='default', generation_settings='default', ranking_settings='default', floor_count = 2, iterations=1):
    # override or set default values
    apartment_proportions = json.loads(desired_proportions) if desired_proportions != "default" else {"5":1.0}
    ranking_settings = json.loads(ranking_settings) if ranking_settings != "default" else {}
    generation_settings = check_defaults_generation_settings(json.loads(generation_settings)) # !! need to add a check if no settings at all

    graph = string_to_graph(my_string_graph)
    # remove nodes with certain attributes CIRCULATION_DISTANCE == 0
    graph = remove_nodes_with_attribute(graph, CIRCULATION_DISTANCE, 0)
    
    # precompute the circulation bounds
    if generation_settings["fast_circulation"] == False:
        circulation_bounds = compute_circulation_minmax_precise(graph, generation_settings, list(apartment_proportions.keys()), floor_count)
    else:
        circulation_bounds = circulation_estimation(graph, list(apartment_proportions.keys()))

    # print(f"circulation_bounds: {circulation_bounds}")
    iteration_apartment_dict = {}
    for i in range(iterations):
        # copy graph to prevent changes on the original graph, needs to be for each iterations
        graph_copy_1 = graph.copy()
        # generate apartments and store them in a dictionary
        all_apartments = generate_apartments(graph_copy_1, apartment_proportions, generation_settings, ranking_settings, circulation_bounds, floor_count, generation_settings["fill"], i)
        iteration_apartment_dict[i] = all_apartments
    
    # make a json of iteration_apartment_dict
    json_iteration_apartment = json.dumps(iteration_apartment_dict, indent=4)
    
    # select the iteration that has the most apartments
    best_apartments = get_best_apartment_dict(iteration_apartment_dict)

    all_apartments_list = [x for sublist in best_apartments.values() for x in sublist]
    
    apartments_proportions = get_apartments_proportions(best_apartments)
    print(f"apartments_proportions: {apartments_proportions}")
    
    # build a graph from the newly created apartments and the circulation nodes
    generated_graph = graph_copy_from_indices(graph, all_apartments_list)
    compacities = (min_compacity, max_compacities_dict)
    post_generation_rankings = compute_post_generation_rankings(generated_graph, all_apartments_list, circulation_bounds, floor_count, compacities)

    json_graph = graph_to_string(generated_graph)
    json_rankings = json.dumps(post_generation_rankings, indent=4)
    json_proportions = json.dumps(apartments_proportions, indent=4)
    
    # print_recap(best_apartments, desired_proportions)
    
    return json_graph, json_rankings, list(flatten_gen(all_apartments_list)), [len(x) for x in all_apartments_list], "json_proportions"


"""
███████╗██╗░░░██╗░█████╗░██╗░░░░░██╗░░░██╗░█████╗░████████╗███████╗   ░█████╗░██████╗░░█████╗░██████╗░████████╗░██████╗
██╔════╝██║░░░██║██╔══██╗██║░░░░░██║░░░██║██╔══██╗╚══██╔══╝██╔════╝   ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
█████╗░░╚██╗░██╔╝███████║██║░░░░░██║░░░██║███████║░░░██║░░░█████╗░░   ███████║██████╔╝███████║██████╔╝░░░██║░░░╚█████╗░
██╔══╝░░░╚████╔╝░██╔══██║██║░░░░░██║░░░██║██╔══██║░░░██║░░░██╔══╝░░   ██╔══██║██╔═══╝░██╔══██║██╔══██╗░░░██║░░░░╚═══██╗
███████╗░░╚██╔╝░░██║░░██║███████╗╚██████╔╝██║░░██║░░░██║░░░███████╗   ██║░░██║██║░░░░░██║░░██║██║░░██║░░░██║░░░██████╔╝
╚══════╝░░░╚═╝░░░╚═╝░░╚═╝╚══════╝░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚══════╝   ╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░╚═════╝░
"""


    
    
"""
███╗░░░███╗░█████╗░██╗███╗░░██╗
████╗░████║██╔══██╗██║████╗░██║
██╔████╔██║███████║██║██╔██╗██║
██║╚██╔╝██║██╔══██║██║██║╚████║
██║░╚═╝░██║██║░░██║██║██║░╚███║
╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝╚═╝░░╚══╝
"""

if __name__ == "__main__":
	app.run(debug=False)
	