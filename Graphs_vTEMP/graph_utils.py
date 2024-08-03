import networkx as nx
import random
import itertools
import json
import time

from networkx.readwrite import json_graph
from collections import defaultdict

from graph_rankings import rank_compacity, rank_circulation
from graph_data import CIRCULATION_DISTANCE, CIRCULATION_TYPE, APARTMENT_TYPE, BLOCK_TYPE



# create a function decorator to time a function
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r  %2.5f sec' % \
              (method.__name__, te-ts))
        return result

    return timed


def float_range(start, stop=None, step=None, decimals=2):
    # if set start=0.0 and step = 1.0 if not specified
    start = float(start)
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0

    # print("start = ", start, "stop = ", stop, "step = ", step)

    count = 0
    while True:
        temp = float(start + count * step)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield round(temp, decimals)
        count += 1


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


def get_type(node_indices, circulation_indices) -> dict:
    type_dictionary = {}
    for i in node_indices:
        if i in circulation_indices:
            type_dictionary[i] = CIRCULATION_TYPE
        else:
            type_dictionary[i] = APARTMENT_TYPE
    
    return type_dictionary


def remove_vertical_at_circulation(graph, circulation_indices):
    """remove vertical edges at circulation nodes"""
    
    new_graph = graph.copy()
    
    for circulation_index in circulation_indices:
        for edge in graph.edges(circulation_index):
            if graph.edges[edge]["is_horizontal"] == "False":
                # check if the edge is in the graph
                if edge in new_graph.edges:
                    new_graph.remove_edge(edge[1], edge[0])
                    new_graph.remove_edge(edge[0], edge[1])
    return new_graph


def get_circulation_distance(node_graph, circulation_indices) -> dict:
    """ 
    Write the distance of each node to the nearest circulation node ino a dictionary
    """
    distance_dictionary = {} 
    
    circulation_nodes = [x for x in node_graph.nodes if x in circulation_indices]
    
    for node in node_graph.nodes:
        distance_dictionary[node] = smallest_distance_source_targets(node_graph, node, circulation_nodes)
    
    return distance_dictionary


def smallest_distance_source_targets(a_graph, origin, targets) -> int:
    """
    compute the smallest distance of a source node to many target nodes
    """

    distance = float("inf")
    
    for target in targets:
        current_distance = nx.shortest_path_length(a_graph, origin, target)
        distance = min(distance, current_distance)
        
    return distance


def merge_two_dict(numero_uno: dict, numero_duo: dict) -> dict:
    """
    merge two dictionaries with the same keys
    https://stackoverflow.com/a/5946322/10235237
    """

    dd = defaultdict(list)
    
    for d in (numero_uno, numero_duo):
        for key, value in d.items():
            dd[key].append(value)
    
    return dd


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


def read_nodes_attr_from_graph(a_graph, some_attributes: list, some_indices: list):
    """
    read specified attributes from selected nodes. 
    If node value is negative, then read specified attributes from all nodes !
    """

    # if node value is negative, update some_indices to all nodes
    if some_indices[0] < 0:
        some_indices = list(range(len(a_graph.nodes)))

    value_list = []

    for att in some_attributes:
        for node in some_indices:
            value_list.append(a_graph.nodes[node][att])

    return value_list, len(some_indices)


def read_edges_attr_from_graph(a_graph, attributes_in: list, indexes_in: list) -> list:
    """

    Parameters
    ----------
    a_graph : networkx.classes.graph.DiGraph
        DESCRIPTION.
    attributes_in : list
        A user selected list of attribute keys.
    indexes_in : list
        A user selected list of node indices to read attribute values from.
    
    Raises
    ------
    Exception
        1. If the edge index is in an incorrect format.
        2. If the edge index is not in the graph.
        3. If the attribute key is not in the graph.
    
    Returns
    -------
    list
        A list of attribute values per selected edge.

    """

    value_list_out = []
    all_indexes = [list(x) for x in a_graph.edges]
    
    if indexes_in == [[-1]]:
        indexes_in = all_indexes
    
    size_ok = all([len(x) == 2 for x in indexes_in])
    if not size_ok:
        raise Exception("Your edge indexes should be a pair of node index per branch. Not more, not less")

    edges_included = all([x in a_graph.edges for x in indexes_in])
    if not edges_included:
        raise Exception("At least one of your provided edges is not part of your graph")
    
    # check if all the attributes_in are in the graph
    edges = a_graph.edges()
    first_edge = list(edges)[0]
    all_attributes = a_graph.edges[first_edge].keys()
    attr_included = all([x in all_attributes for x in attributes_in])
    if not attr_included:
        raise Exception("At least one of your provided attributes doesn't seem to be in the graph")
    
    # if the user wants to read all the edges
    if indexes_in[0] == -1:
        indexes_in = all_indexes

    for edge in indexes_in:
        for att in attributes_in:
            value_list_out.append(a_graph.edges[edge][att])
    
    return value_list_out


def dict_val_to_nested(a_dict):
    """
    convert a dictionary of values to a nested list
    """
    nested_out = []
    for i in a_dict.keys():
        nested_out.append(a_dict[i])
    return nested_out


def remove_edges_constrains(my_graph, edge_attributes, edge_values):
    """
    This is used in Build/Modify Graph in graph_component.py
    Remove all the edges from a graph where each value is matched for each attributes
    """
    
    remove_list = []
    
    for attribute_index in range(len(edge_attributes)):
        attribute = edge_attributes[attribute_index]
        value = edge_values[attribute_index]
        
        for edge in my_graph.edges():
            if my_graph[edge[0]][edge[1]][attribute] == value:
                remove_list.append(edge)

    # remove_list = list(set(remove_list))
    my_graph.remove_edges_from(remove_list)

    return my_graph


def get_all_possible_targets(G, start_node, apart_length, max_targets, flat=False):
    """
    
    Parameters
    ----------
    G : networkx.classes.graph.DiGraph
        DESCRIPTION.
    start_node : int
        DESCRIPTION.
    apart_length : int
        DESCRIPTION.
    max_targets : int
        DESCRIPTION.
    flat : bool, optional
        This allows to return either nested lists or one flattened list. The default is False (nested list).
    
    Performs
    -------
    1. Compute the intermediate distances between the start node and the target nodes for the given apart length
    2. Compute all the possible targets for each intermediate distance and store them in a dictionary (key: distance, value: list of targets)
    3. One distance after the other, pop one target and store it in another dictionary
    4. Exit the loop when the total number of targets is equal to the max_targets value

    Returns
    -------
    list
        A limited selected node indices in a (nested or not) list.
    
    Notes
    -----
    1. The function will always pick the targets that are closest to the start node first (could be improved)
    2. The function uses a error count to avoid infinite loops, 
        but allows to test each distance for possible targets before exiting loop.
        (could be improved probably)
    """

    intermediate_distances = list(range(apart_length, 0, -2))
    intermediate_distances.sort()

    all_targets = {d:[] for d in intermediate_distances}
    selected_targets = {d:[] for d in intermediate_distances}
    current_distance_index = 0
    index_error_count = 0


    for distance in intermediate_distances:
        all_targets[distance] = list(nx.descendants_at_distance(G, start_node, distance))
        # note for line above: descendants_at_distance returns a set, not a list nor a generator
    
    if max_targets < 0:
        if flat:
            return list(flatten_gen(all_targets.values()))
        else:
            return all_targets

    while len(list(flatten_gen(selected_targets.values()))) < max_targets:
        wrapped_index = current_distance_index % len(intermediate_distances)
        current_distance = intermediate_distances[wrapped_index]

        # bump the current distance index for the next iteration
        current_distance_index += 1

        # if we've tested all distances, but we've not found a single target, then we're done
        if index_error_count == len(intermediate_distances):
            break

        # try to pop a target from the current distance, if it works, add it to the selected_targets and reset error count
        try:
            selected_target = all_targets[intermediate_distances[wrapped_index]].pop()
            selected_targets[current_distance].append(selected_target)
            index_error_count = 0
        
        # if can't pop, then go to the next distance and bump the error count
        except IndexError:
            index_error_count += 1
            continue
            

    if flat:
        return list(flatten_gen(selected_targets.values()))
        
    return selected_targets


def get_all_possible_paths(G, start_node, target, cutoff, max_paths):
    """
    Get the simple paths with itertools.islice
    """

    if max_paths < 0:
        # generate all paths (heavy)
        return list(nx.all_simple_paths(G, start_node, target, cutoff=cutoff))
    else:
        # generate the desired number of paths with itertools.islice
        return list(itertools.islice(nx.all_simple_paths(G, start_node, target, cutoff=cutoff), max_paths))



def get_score_circulation_distance_aparts(graph, aparts):
    """Get the circulation distance scores for each apartment"""
    proximity = []
    proximity_sum = []
    for apart in aparts:
        apart_proxy = []
        for bloc in apart:
            apart_proxy.append(graph.nodes[bloc][CIRCULATION_DISTANCE])

        proximity.append(apart_proxy)
        proximity_sum.append(sum(apart_proxy))
    return proximity, proximity_sum


def get_starting_positions(my_graph):
    """returns a list of indexes that connect with circulation_indexes, where the edge connecting the two meets the
    attribute / value combo"""
    circulation_nodes = [x for x in my_graph.nodes if my_graph.nodes[x][CIRCULATION_DISTANCE] == 0]
    
    # get all the neighbors that connect to the circulation nodes horizontally
    for node in circulation_nodes:
        neighbors = list(my_graph.neighbors(node))
        for neighbor in neighbors:
            if my_graph[node][neighbor]["is_horizontal"]:
                circulation_nodes.append(neighbor)
    
    return circulation_nodes


def reorder_indexes(main_list, sub_list, front=True):
    """Reorder a list of indexes so that the sub_list is at the front or at the back of the main_list, without any duplicates"""
    
    main_list = [x for x in main_list if x not in sub_list]

    if front:
        return sub_list + main_list
    else:
        return main_list + sub_list


def remove_nodes(my_graph, circulation_indexes):
    """remove nodes from a graph (and therefore, the edges to those nodes)"""
    for index in circulation_indexes:
        my_graph.remove_node(index)
    return my_graph


def all_possible_targets_multiple_starts(a_graph, start_indexes, length: list):
    """get all the possible targets from all the possible starts"""
    every_possible_targets = []
    for start in start_indexes:
        every_possible_targets.append(get_all_possible_targets(a_graph, start, length))
    
    # filter the targets that are a circulation bloc
    return [x for x in every_possible_targets if a_graph.nodes[x][CIRCULATION_DISTANCE] > 0]


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


def select_apart_length_first_time_old(desired_proportions, current_apartments):
    for length in desired_proportions.keys():
        if len(current_apartments[length]) < 1:
            return length
        else:
            raise Exception("!!! YIKES !!!")


def select_apart_length_first_time(desired_proportions):
    """
    select the bigger key value, for the biggest values in the dict
    potential improvement: select the biggest key when values are the same
    """
    return max(desired_proportions, key=desired_proportions.get)


def select_apart_length_next_times(desired_proportions, current_apartments):
    """
    select the apart length based on the current proportions and the desired ones
    """
    proportion_ratios = {}
    current_proportions = {}
    ratio_differences = {}
    
    for key in current_apartments.keys():
        current_proportions[key] = len(current_apartments[key])/len(current_apartments.values())
    
    for key in current_proportions.keys():
        proportion_ratios[key] = current_proportions[key]/desired_proportions[key]
        ratio_differences[key] = abs(current_proportions[key] - desired_proportions[key])
    
    # return the length that has the smallest ratio
    return min(proportion_ratios, key=proportion_ratios.get)


def select_apart_length(desired_proportions, current_apartments, main_loop):
    """
    select the apart length based on the current proportions (if exists) and the desired ones
    """

    # if there are no current apartments, return the biggest desired proportion
    if len(current_apartments.values()) == 0:
        return max(desired_proportions, key=desired_proportions.get)

    # proceed with the normal selection, based on the current proportions
    proportion_ratios = {}
    current_proportions = {}
    ratio_differences = {}
            
    for key in current_apartments.keys():
        current_proportions[key] = len(current_apartments[key])/len(current_apartments.values())
    
    for key in current_proportions.keys():
        proportion_ratios[key] = current_proportions[key]/desired_proportions[key]
        ratio_differences[key] = abs(current_proportions[key] - desired_proportions[key])

    if main_loop:
        # return the length that has the smallest ratio
        return min(proportion_ratios, key=proportion_ratios.get)
    
    else:
        # return a stack of the length that has the smallest ratio
        return sorted(proportion_ratios, key=proportion_ratios.get)


def select_apart_length_later_check(desired_proportions, current_apartments):
    """
    return a list of desired_proportion keys, sorted so the most needed proportion is first.
    #copilot
    """
    proportion_ratios = {}
    current_proportions = {}
    ratio_differences = {}
    
    for key in current_apartments.keys():
        current_proportions[key] = len(current_apartments[key])/len(current_apartments.values())
    
    for key in current_proportions.keys():
        proportion_ratios[key] = current_proportions[key]/desired_proportions[key]
        ratio_differences[key] = abs(current_proportions[key] - desired_proportions[key])
    
    # return the length that has the smallest ratio
    return sorted(ratio_differences, key=ratio_differences.get, reverse=True)


def filter_targets(my_graph, targets, start, max_floor):
    # filter the targets so they are not to many floors above start
    filtered_targets = [x for x in targets if my_graph.nodes[x]["branch_A"] - my_graph.nodes[start]["branch_A"] < max_floor]
    # filter the targets that are below the start (!! doens't accout for aparts that go down then up (-1+1=0 duh))
    filtered_targets = [x for x in filtered_targets if my_graph.nodes[x]["branch_A"] >= my_graph.nodes[start]["branch_A"]]
    return filtered_targets


def get_limited_number_of_apartments(graph, start, target, max_length, max_count):
    # if max_count == -1, get all the possible paths
    if max_count == -1:
        return list(nx.all_simple_paths(graph, start, target, cutoff=int(max_length)))
    # if max_count > 0, get the first max_count paths
    elif max_count > 0:
        apart_generator = nx.all_simple_paths(graph, start, target, cutoff=int(max_length))
        return list(itertools.islice(apart_generator, 0, int(max_count)))
    else:
        raise Exception(f"You can't search for {max_count} apartments, change the max_paths value\n")


def get_limited_number_of_targets(list_of_targets, max_number_targets):
    
    if max_number_targets == -1:
        return list_of_targets
    elif max_number_targets > 0:
        return list_of_targets[:int(max_number_targets)]
    else:
        raise Exception(f"You can't search for {max_number_targets} targets, change the max_targets value\n")





def select_apart_from_dict(ranking_dict, iteration_count):
    """
    returns the key and index of the best apartment overall
    the key is the source_target tuple, and the index is the index of the apartment in the list of apartments
    """

    # get the keys and the values of the ranking dict
    keys = list(ranking_dict.keys())
    values = list(ranking_dict.values())
    values_length = [len(x) for x in values]

    flatten_gened_values = [item for sublist in values for item in sublist]

    # get the hightest value, then get all the indices of the highest values
    highest_value = max(flatten_gened_values)
    highest_value_indices = [i for i, x in enumerate(flatten_gened_values) if x == highest_value]

    # if many highest values, select one randomly using the iteration count
    if len(highest_value_indices) > 1:
        random.seed(iteration_count)
        selected_index = random.choice(highest_value_indices)
    else:
        selected_index = highest_value_indices[0]
    
    # find from which list the selected index is
    selected_key = keys[0]
    for i in range(len(values_length)):
        if selected_index < sum(values_length[:i+1]):
            selected_key = keys[i]
            break
    
    # get the index of the selected apartment in the list of apartments
    selected_index = selected_index - sum(values_length[:keys.index(selected_key)])
    

    return selected_key, selected_index


def remove_higher_targets(graph, start_node, possible_targets, maximum_floor):
    """
    Remove the targets that are higher or lower than the maximum floor compared to the start node
    """

    # get the start node floor
    start_node_floor = graph.nodes[start_node]["branch_A"]

    # for all possible targets
    for target in possible_targets:
            
            # get the target floor
            target_floor = graph.nodes[target]["branch_A"]
    
            # if the target floor is higher than the maximum floor, remove the target from the list
            possible_targets = [x for x in possible_targets if graph.nodes[x]["branch_A"] < start_node_floor + maximum_floor]
            possible_targets = [x for x in possible_targets if graph.nodes[x]["branch_A"] > start_node_floor - maximum_floor]

    return possible_targets


def print_recap(dict_final_aparts, desired_proportions):
    """
    Print a recap of the generated apartments
    """
    print("\n\n\n###############################################################################")
    print("############################# FINAL RESULTS ###################################")
    print("###############################################################################\n")
    
    print(f"total number of generated apartments: {len([apart for sublist in dict_final_aparts.values() for apart in sublist])}")
    print(f"total number of blocs used: {len(list(flatten_gen(dict_final_aparts.values())))}\n")
    print(f"desired proportions were: {json.loads(desired_proportions)}")
    print(f"actual proportions are:   {get_apartments_proportions(dict_final_aparts)}\n\n")
    
    for key in dict_final_aparts.keys():
        print(f"number of apartments of length {key}: {len(dict_final_aparts[key])}")
        for apart in dict_final_aparts[key]:
            print(f"    {apart}")
        print("\n")
    pass


def get_best_apartment_dict(dict_of_dicts):
    """
    return the dict with the longest values list, and the key of that dictionary
    """
    best_dict = {"0": {}}
    best_key = 0
    for key in dict_of_dicts.keys():

        # print(dict_of_dicts[key])
        current_flatten_gened_aparts = [apart for sublist in dict_of_dicts[key].values() for apart in sublist]
        stored_flatten_gened_aparts = [apart for sublist in best_dict.values() for apart in sublist]
        if len(current_flatten_gened_aparts) > len(stored_flatten_gened_aparts):
            best_dict = dict_of_dicts[key]
            best_key = key
    return best_dict


def get_apartments_proportions(apartment_dict):
    """
    Get the proportions of the generated apartments
    """
    total_aparts = len([x for sublist in apartment_dict.values() for x in sublist])
    proportions = {key: round(len(apartment_dict[key])/total_aparts, 2) for key in apartment_dict.keys()}
    
    return proportions


def select_apart_from_dict_ex(rank_dict):
    """
    select the best apartment from the dict #copilot
    """

    all_values = list(flatten_gen(rank_dict.values()))
    if len(all_values) < 1:
        return None, None
    
    # get the best ranking
    best_ranking = max(all_values)
    # get the key of the best ranking
    best_ranking_key = [k for k, v in rank_dict.items() if best_ranking in v]
    
    # get the index of the best ranking
    best_ranking_index = rank_dict[best_ranking_key[0]].index(best_ranking)
    # select the key and index of the best ranking
    selected_key = best_ranking_key[0]
    
    return selected_key, best_ranking_index 


def build_empty_dict(desired_proportions):
    """
    build an empty dict using the desired_proportions keys
    """
    empty_dict = {}
    for key in desired_proportions.keys():
        empty_dict[key] = []
    return empty_dict


def reorder_possible_apartments(possible_apartments, possible_apartment_rankings):
    """
    reorder the possible apartments based on the ranking of the possible apartments
    """
    # sort the possible apartments based on the ranking
    possible_apartments = [x for _, x in sorted(zip(possible_apartment_rankings, possible_apartments))]
    return possible_apartments


def remove_invalid_apartments(possible_apartments, filter_list):
    # filter out the apartments based on the boolean filter_list
    possible_apartments = [x for x, y in zip(possible_apartments, filter_list) if y]
    
    return possible_apartments


def remove_nodes_with_attribute(graph, attribute, value):
    """
    Remove all nodes that have the attribute "circulation_distance" equal to 0
    """
    nodes_to_remove = [node for node in graph.nodes if graph.nodes[node][attribute] == value]
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def define_apartment_validity(graph, possible_apartments, selected_apart_length, max_floor_count):
    # create a list of booleans that indicate if the apartment length is equal to the selected apartment length
    apartments_length_validity = [int(len(x)) == int(selected_apart_length) for x in possible_apartments]
    lengthes = [len(x) for x in possible_apartments]
    
    # create a list of booleans that indicate if the apartment height delta is lower or equal to max_floor_count
    apartments_height_validity = [max_floor_count - 1 >= get_apartment_height_delta(graph, x) for x in possible_apartments]
    
    # combine the two lists so that the apartments that are both valid in length and height are kept
    apartments_validity = [x and y for x, y in zip(apartments_length_validity, apartments_height_validity)]
    
    return apartments_validity


def get_apartment_height_delta(the_graph, apartment):
    """
    return the height delta of an apartment
    """
    # get the nodes that match the apartment
    apartment_nodes = [node for node in the_graph.nodes if node in apartment]
    branch_A = nx.get_node_attributes(the_graph, "branch_A")
    
    floors = [branch_A[x] for x in apartment_nodes]
    
    # get the highest delta in the floors
    highest_floor, lowest_floor = max(floors), min(floors)
    
    return highest_floor - lowest_floor


def compute_post_generation_rankings(final_graph_aparts, generated_apartments, bounds, max_floor, compacity_data):
    
    # rank the aparts based on the sum of the circulation distance of each nodes
    circulation_rankings = rank_circulation(final_graph_aparts, generated_apartments, circulation_bounds=bounds)
    
    # rank the aparts based on the computed compacity divide by 2 cause Digraph
    compacity_rankings = rank_compacity(final_graph_aparts, generated_apartments, max_floor=max_floor, compacity_data=compacity_data)
    
    # rank the aparts based on window count
    window_rankings_before, window_rankings_after = compute_window_rankings(final_graph_aparts, generated_apartments)

    # rank the apart based on the number of floors (use node attribute "branch_A")
    branch_A = nx.get_node_attributes(final_graph_aparts, "branch_A")
    floors_rankings = [len(set([branch_A[x] for x in apart])) for apart in generated_apartments]
    
    # return all the rankings in a dictionary
    return {
        "circulation_rankings": circulation_rankings,
        "compacity_rankings": compacity_rankings,
        "window_rankings_before": window_rankings_before,
        "window_rankings_after": window_rankings_after,
        "floors_rankings": floors_rankings
        }


def compute_window_rankings(graph, aparts):
    # rank the aparts based on the "window_count" node attribute
    window_rankings_before = [sum([graph.nodes[x]["window_count"] for x in apart]) for apart in aparts]
    # print("-- window_rankings_before ", window_rankings_before)
    
    window_rankings_after = []
    for apart in aparts:
        apart_window = []
        for node in apart:
            # get the edges of attribute "is_horizontal" == "True"
            horizontal_edges = [x for x in graph.edges(node) if graph.edges[x]["is_horizontal"] == "True"]
            window_count = 4 - len(horizontal_edges)
            apart_window.append(window_count)
        window_rankings_after.append(sum(apart_window))
    
    # print("-- window_rankings_after ", window_rankings_after)
    
    return window_rankings_before, window_rankings_after


def compute_circulation_minmax_precise(graph, settings, apart_sizes, max_floor):
    """"
    (can be massively improved)

    This function finds the minimum and maximum circulation values possible for each apartment for
    - the current volume
    - the circulation nodes
    - the apart sizes
    - the max floor

    The function runs prior to the generation, so that the circulation values can be ussed to rank 
    the circulation values of the generated apartments.

    The function returns a dictionary with the minimum and maximum circulation values for each 
    apartment size.
    """

    print("\n........................................")
    print("...Initiating circulation computation...")
    print("........................................")

    circulation_nodes = [x for x in graph.nodes if graph.nodes[x]["CIRCULATION_DISTANCE"] == 0]
    starting_nodes = [x for x in graph.nodes if graph.nodes[x]["CIRCULATION_DISTANCE"] == 1]

    # create a copy of graph remove ciruclation nodes
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(circulation_nodes)
    
    # create a list of empty list based on apart_sizes length
    apart_dict = dict.fromkeys(apart_sizes, [])
    targets = dict.fromkeys(apart_sizes, [])
    circulations = dict.fromkeys(apart_sizes, [])
    
    # get list of all descendants at distance from all starting_nodes
    possible_distances = dict.fromkeys(apart_sizes, [])
    # for each key, create a list of values that decrease at a rate of 2, but above 0
    for key in apart_dict.keys():
        possible_distances[key] = [x for x in range(int(key), 0, -2)]


    # for each key and for each element in list, get the simple paths from node to descendant and add it to target
    print("computing targets")
    for key in apart_dict.keys():
        for node in starting_nodes:
            for distance in possible_distances[key]:
                descendants = list(nx.descendants_at_distance(graph_copy, node, distance))
                for descendant in descendants:
                    if descendant not in targets[key]:
                        targets[key].append(descendant)

    # for each key, filter the number of targets to max_targets
    print("filtering targets")
    for key in apart_dict.keys():
        targets[key] = get_limited_number_of_targets(targets[key], settings["max_targets"])


    # for each key and each element in list, get all the possible paths from starting_nodes to targets and store them in apart_dict
    print("computing paths")
    for key in apart_dict.keys():
        for node in starting_nodes:
            for target in targets[key]:
                paths = get_limited_number_of_apartments(graph_copy, node, target, int(key), settings["max_paths"])
                for path in paths:
                    apart_dict[key].append(path)

    # get the total number of apartments
    total_apartments = sum([len(x) for x in apart_dict.values()])
    print("total apartments tested: ", total_apartments)
    # filter the apartments based on a validity check
    print("filtering apartments")
    for key in apart_dict.keys():
        validity_list = define_apartment_validity(graph_copy, apart_dict[key], key, max_floor)
        apart_dict[key] = [x for x, y in zip(apart_dict[key], validity_list) if y]

    # for each key, for each element in list, get the sum of circulation distance and store it in circulations
    print("computing circulations")
    for key in apart_dict.keys():
        for path in apart_dict[key]:
            circulations[key].append(sum([graph.nodes[x][CIRCULATION_DISTANCE] for x in path]))

    # for each key, get the min and max circulation values and store them in min_max_circulations
    print("computing min and max circulations")
    min_max_circulations = dict.fromkeys(apart_sizes, [])
    for key in apart_dict.keys():
        min_max_circulations[key] = [min(circulations[key]), max(circulations[key])]

    return min_max_circulations


def circulation_estimation(graph, apart_sizes):
    """
    This function estimates the minimum and maximum circulation values possible for each apartment for
    - total node count
    - total ciruclation node count
    - apart sizes
    - max floor
    """

    # initiate our circulation estimation dictionary
    circulation_estimation_dict = dict.fromkeys(apart_sizes)

    # get the total number of nodes and circulation nodes
    total_nodes = len(graph.nodes)
    circulation_one_nodes = [x for x, y in graph.nodes(data=True) if y[CIRCULATION_DISTANCE] == 1]
    total_circulation_nodes = len(circulation_one_nodes)


    # the minimum circulation value is the key as integer
    # if the key is larger than the total number of nodes with CIRCULATION_DISTANCE == 1, then, the minimum circulation is the sum of 
    # the number of nodes with CIRCULATION_DISTANCE == 1 and the difference between the key and the total number of nodes with CIRCULATION_DISTANCE == 1

    for key in circulation_estimation_dict.keys():
        int_key = int(key)
        if int_key > total_circulation_nodes:
            circulation_estimation_dict[key] = [total_circulation_nodes + (int_key - total_circulation_nodes)]
        else:
            circulation_estimation_dict[key] = [int_key]


    for key in circulation_estimation_dict.keys():
        int_key = int(key)
        # the maximum circulation value is the sum of values of the circulation nodes, starting at 1, and increasing by 1 for the length of the apartment.
        # The maximum value for a node is the maximum CIRCULATION_DISTANCE value.
        # example : if the maximum CIRCULATION_DISTANCE is 4 and the apartment is 6 nodes long. 
        # the maximum circulation value is 1 + 2 + 3 + 4 + 4 + 4 = 18
        max_circulation = 0
        for i in range(int_key):
            max_circulation += min(i + 1, max([y[CIRCULATION_DISTANCE] for x, y in graph.nodes(data=True)]))
        # print(f"maximum found value in graph is {max([y[CIRCULATION_DISTANCE] for x, y in graph.nodes(data=True)])}")
        # print(f"max circulation for {key} is {max_circulation}")
        circulation_estimation_dict[key].append(max_circulation)
    
    return circulation_estimation_dict

    
def flatten_gen(L):
    """
    flatten_gen any level of nested lists
    returns a generator
    """
    for item in L:
        try:
            yield from flatten_gen(item)
        except TypeError:
            yield item
    

def get_nodes_with_attribute(a_graph, attribute, val=0, more=True):
    """get the nodes that meet the attribute-value combo. Either above (more=True) or below (more=False)"""
    nodes_at = []

    for (p, d) in a_graph.nodes(data=True):
        if more:
            if d[attribute] > val:
                nodes_at.append(p)
        else:
            if d[attribute] < val:
                nodes_at.append(p)

    return nodes_at


def graph_from_subgraph(original_graph, subgraph):
    """ create new graph from subgraph with all attributes"""
    
    # check if subgraph is digraph() or graph()
    if isinstance(subgraph, nx.classes.digraph.DiGraph):
        apart_graph = nx.DiGraph()
    else:
        apart_graph = nx.Graph()
    
    # build the graph
    for node in subgraph.nodes:
        apart_graph.add_node(node, **original_graph.nodes[node])
        for edge in subgraph.edges:
            apart_graph.add_edge(*edge, **original_graph.edges[edge])

    return apart_graph


def graph_copy_from_indices(original_graph, node_index_list):
    """
    create a new graph, with all the nodes from the list of node indices
    all the nodes and edges must come from the original graph
    """
    # get the node indices in original_graph where BLOCK_TYPE attribute is "CIRCULATION"
    circulation_nodes = [x for x in original_graph.nodes if original_graph.nodes[x][BLOCK_TYPE] == "CIRCULATION"]
    
    # add the circulation_indices to the node_index_list
    node_index_list = node_index_list + [circulation_nodes]
    
    # check if subgraph is digraph() or graph()
    if isinstance(original_graph, nx.classes.digraph.DiGraph):
        new_graph = nx.DiGraph()
    else:
        new_graph = nx.Graph()
    
    # build the graph
    for apart in node_index_list:
        for node in apart:
            new_graph.add_node(node, **original_graph.nodes[node])
            
    # copy the edges from the nodes in original graph that are in apart_graph
    for edge in original_graph.edges:
        if edge[0] in new_graph.nodes and edge[1] in new_graph.nodes:
            new_graph.add_edge(*edge, **original_graph.edges[edge])
    
    
    return new_graph


def check_defaults_generation_settings(settings_dict):
    # check if all the required keys are in settings_dict. They are : compacity, max_targets and max_paths
    # if they are not, add them with a default value. Respectively : 0.3, 10 and 10
    required_keys = ["compacity", "max_targets", "max_paths"]
    default_values = [0.3, 10, 10]
    
    for key, value in zip(required_keys, default_values):
        if key not in settings_dict:
            settings_dict[key] = value
    
    return settings_dict


def string_to_graph(a_string_graph):
    a_graph = json.loads(a_string_graph)
    return json_graph.node_link_graph(a_graph, directed=True, multigraph=False,
                                      attrs={"link": "edges", "source": "from", "target": "to"})


def graph_to_string(a_graph):
    jsongraph = json_graph.node_link_data(a_graph, attrs={"link": "edges", "source": "from", "target": "to"})
    return json.dumps(jsongraph, indent=4)





