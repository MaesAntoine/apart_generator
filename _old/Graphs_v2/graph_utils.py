import networkx as nx
import random
import itertools
import json
import time

from multiprocessing.dummy import current_process
from xml.dom.expatbuilder import makeBuilder
from networkx.readwrite import json_graph
from collections import defaultdict


# CONSTANTS
BLOCK_TYPE = "BLOCK_TYPE"
CIRCULATION_TYPE = "CIRCULATION"
APARTMENT_TYPE = "APARTMENT"
CIRCULATION_DISTANCE = "CIRCULATION_DISTANCE"


# create a function decorator to time a function
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r  %2.2f sec' % \
              (method.__name__, te-ts))
        return result

    return timed


def frange(start, stop=None, step=None, decimals=2):
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
    distance_dictionary = {} 
    
    circulation_nodes = [x for x in node_graph.nodes if x in circulation_indices]
    
    for node in node_graph.nodes:
        distance_dictionary[node] = smallest_distance_source_targets(node_graph, node, circulation_nodes)
    
    return distance_dictionary


def smallest_distance_source_targets(a_graph, origin, targets):
    # compute the smallest distance of a source node to many target nodes
    distance = float("inf")
    
    for target in targets:
        current_distance = nx.shortest_path_length(a_graph, origin, target)
        distance = min(distance, current_distance)
        
    return distance


def merge_two_dict(numero_uno: dict, numero_duo: dict) -> dict:
    # https://stackoverflow.com/a/5946322/10235237
    
    dd = defaultdict(list)
    
    for d in (numero_uno, numero_duo):
        for key, value in d.items():
            dd[key].append(value)
    
    return dd


def build_list_dict_attribute(list_of_attributes: list, labels: list) -> list:
    
    dict_list = []
    for paired_values in list_of_attributes:
        current_dict = {}
        
        for label_idx in range(len(labels)):
            current_dict[labels[label_idx]] = paired_values[label_idx]
            
        dict_list.append(current_dict)

    return dict_list


def edges_attr_list(node_list: list, attributes: list):
    # create a list of tuples (start, end)
    
    tuples_node_attr = []
    
    for index in range(len(node_list)):
        tuples_node_attr.append((node_list[index][0], node_list[index][1], attributes[index]))
        
    return tuples_node_attr


def read_nodes_attr_from_graph(a_graph, some_attributes: list, some_indices: list):
    # read specified attributes from selected nodes. If no nodes then read specified attributes from all nodes !
    
    if some_indices[0] < 0:
        # if no indices provided, get all indices possible
        some_indices = list(range(len(a_graph.nodes)))
    value_list = []

    for att in some_attributes:
        for node in some_indices:
            value_list.append(a_graph.nodes[node][att])

    return value_list, len(some_indices)


def read_edges_attr_from_graph(a_graph, attributes_in: list, indexes_in: list) -> list:
    value_list_out = []
    all_indexes = [list(x) for x in a_graph.edges]
    
    if indexes_in == [[-1]]:
        indexes_in = all_indexes
    
    size_ok = all([len(x) == 2 for x in indexes_in])
    edges_included = all([x in a_graph.edges for x in indexes_in])
    
    attr_included = True

    if indexes_in[0] == -1:
        indexes_in = all_indexes
    
    if not size_ok:
        raise Exception("Your edge indexes should be a pair of node index per branch. Not more, not less")
    if not edges_included:
        raise Exception("At least one of your provided edges is not part of your graph")
    if not attr_included:
        raise Exception("At least one of your provided attributes doesn't seem to be in your graph")
    
    if size_ok:
        for edge in indexes_in:
            for att in attributes_in:
                value_list_out.append(a_graph.edges[edge][att])
    
    return value_list_out


def dict_val_to_nested(a_dict):
    
    nested_out = []
    for i in a_dict.keys():
        nested_out.append(a_dict[i])
    return nested_out


def remove_edges_constrains(my_graph, edge_attributes, edge_values):
    """Remove all the edges from a graph where each value is matched for each attributes"""
    
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


def get_all_possible_targets_old(my_graph, start_index: int, apart_length, flat=False) -> list:
    """Get all the possible targets from a start node"""
    all_targets = []
    intermediate_distances = list(range(apart_length, 0, -2))
    intermediate_distances.sort()
    
    for distance in intermediate_distances:
        current_targets = list(nx.descendants_at_distance(my_graph, start_index, distance))
        if len(current_targets) > 0:
            # filter potential circulation bloc from targets
            filtered_targets = [x for x in current_targets if my_graph.nodes[x][CIRCULATION_DISTANCE] > 0]
            all_targets.append(filtered_targets)
    
    if flat:
        return list(flatten(all_targets))
    
    return all_targets


def get_all_possible_targets(my_graph, start_index, apart_lenght, max_targets, flat=False):
    """Get all the possible targets from a start node as a dictionary"""
    all_targets = {}
    intermediate_distances = list(range(apart_lenght, 0, -2))
    intermediate_distances.sort()
    
    for distance in intermediate_distances:

        if max_targets < 0:
            current_targets = list(nx.descendants_at_distance(my_graph, start_index, distance))
        else:
            target_generator = nx.descendants_at_distance(my_graph, start_index, distance)
            current_targets = list(itertools.islice(target_generator, 0, max_targets))
            
        if len(current_targets) > 0:
            # filter potential circulation bloc from targets
            filtered_targets = [x for x in current_targets if my_graph.nodes[x][CIRCULATION_DISTANCE] > 0]
            all_targets[distance + 1] = filtered_targets
    
    if flat:
        return list(flatten(all_targets.values()))
    
    return all_targets



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


def generate_apartments_v2(graph, desired_proportions, generation, maximum_floor, fill, iteration):
    """
    better logic?
    
    define empty dict of apartments
    select a start
    
    for target in all_possible_targets
        apartment_dict[target] = [nx.all_simple_paths(graph, start, target, selected_length)]
        
    """
    
    # I know this is terrible but I need it
    global MAX_FLOOR, COMPACITY
    MAX_FLOOR = maximum_floor
    COMPACITY = generation["compacity"]
    
    circulation_nodes = [x for x in graph.nodes if graph.nodes[x][CIRCULATION_DISTANCE] == 0]
    start_nodes = get_neighbors_constrained(graph, circulation_nodes, "is_horizontal", True) # !!! could be the line above but == 1
    windows = get_nodes_with_attribute(graph, "window_count", val=1, more=False)
    priority_start = list(set.intersection(set(start_nodes), set(windows)))
    start_nodes_ordered = reorder_indexes(start_nodes, priority_start, front=True)
    
    min_max_circulation = compute_circulation_minmax_precise(graph, generation, circulation_nodes, start_nodes_ordered, list(desired_proportions.keys()), maximum_floor)
    print(f"min_max_circulation: {min_max_circulation}")
    
    apartment_dict = build_empty_dict(desired_proportions)
    apartment_list = []
    later_check_starts = []
    apartment_graphs = []
    compacity_rankings = []
    circulation_rankings = []
    random.seed(iteration)

    # remove the circulation blocs from the graph
    graph.remove_nodes_from(circulation_nodes)
    
    print("\n==================================")
    print(f"= INITIALISATION OF GENERATION {iteration} =")
    print("==================================\n")
    
    while len(start_nodes_ordered) > 0:
        temporary_apart_dict = {}
        temporary_rank_dict = {}
        absolute_all_apart = []
        start = start_nodes_ordered.pop(0)

        # find and select the desired apart length for the current loop
        if len(list(flatten(apartment_dict.values()))) == 0:
            # selected_apart_length = list(desired_proportions.keys())[0]
            selected_apart_length = select_apart_length_first_time(desired_proportions)
        else:
            # select apart length depending on desired proportions and current generation proportions
            selected_apart_length = select_apart_length_next_times(desired_proportions, apartment_dict)
        
        # get the possible targets in one flatten list, since we will create potential aparts + ranks for each
        flatten_possible_targets = get_all_possible_targets(graph, start, int(selected_apart_length)-1, int(generation["max_targets"]), flat=True)
        # print(f"lenght of possible targets: {len(flatten_possible_targets)}")
        
        #shuffle the flatten_possible_tagrets
        #random.shuffle(flatten_possible_targets)
        # filter the list to keep the generation["max_targets"] number of targets
        # flatten_possible_targets = get_limited_number_of_targets(flatten_possible_targets, generation["max_targets"])
        
        # start loop to get all apart for all targets
        for target in flatten_possible_targets:
            source_target = (start, target)
            potential_aparts = get_limited_number_of_apartments(graph, start, target, selected_apart_length, int(generation["max_paths"]))
            #potential_aparts = list(nx.all_simple_paths(graph, start, target, cutoff=int(selected_apart_length)))
            absolute_all_apart.extend(potential_aparts)
            # filter the aparts that might be smaller
            filtering_list = define_apartment_validity(graph, potential_aparts, selected_apart_length, MAX_FLOOR)
            possible_apartments = remove_invalid_apartments(potential_aparts, filtering_list)

            temporary_apart_dict[source_target] = possible_apartments
            
            # remove key, value if the key has no value
            temporary_apart_dict = {k: v for k, v in temporary_apart_dict.items() if len(v) > 0}
            temporary_rank_dict = rank_apart_in_dict(graph, temporary_apart_dict, min_max_circulation)
        
        # debug
        # print(f"temporary_apart_dict: {temporary_apart_dict}")
        # print(f"temporary_rank_dict: {temporary_rank_dict}")

        selected_key, selected_index = select_apart_from_dict(temporary_rank_dict)
        
        # print("temporary_apart_dict = ", temporary_apart_dict)

        if not temporary_apart_dict:
            # if no potential aparts for this start/length combo, add start to later check
            later_check_starts.append(start)
            continue
        

        # debug
        # print(f"selected_key: {selected_key}")
        # print(f"selected_index: {selected_index}")

        # select the current apartment
        selected_apartment = temporary_apart_dict[selected_key][selected_index]
        apartment_dict[selected_apart_length].append(selected_apartment)
        apartment_list.append(selected_apartment)

        graph.remove_nodes_from(selected_apartment)
        start_nodes_ordered = [x for x in start_nodes_ordered if x not in selected_apartment]
        
        number_apart_valid = len([x for sublist in temporary_apart_dict.values() for x in sublist])
        print(f"tested apartments {len(absolute_all_apart)},    valid apartments: {number_apart_valid},    selected apart: {selected_apartment}")
        # print(f"current generated aparts: {apartment_dict}")


    if fill and later_check_starts:
        # start the later check
        # for each start, create a stack of lengths from the prefered proportions vs the current proportions
        # try to create an apartment and if it succed, add it to the list of apartments
        print("\n------------------------------------------------------------------------------")
        print("|First pass of the generation completed, now trying to fill the eventual gaps|")
        print("------------------------------------------------------------------------------\n")
        
        print(f"current number of generated apartments is: {len([apart for sublist in apartment_dict.values() for apart in sublist])}")
        print(f"potential gap starts:\n{later_check_starts}")

        for start in later_check_starts:
            # as we check for potential new filling aparts, check if it's actually still in the graph
            if start in list(flatten(apartment_dict.values())):
                continue
            
            # create the stack
            stack = [*select_apart_length_later_check(desired_proportions, apartment_dict)]
            # !!!!! SOMETHING IS FISHY HERE !!!!!
            # print("loop stack => ", stack)
            
            while len(stack) > 0:
                
                selected_apart_length = stack.pop(0)
                
                temporary_apart_dict = {}
                temporary_rank_dict = {}
                
                flatten_possible_targets = get_all_possible_targets(graph, start, int(selected_apart_length)-1, int(generation["max_targets"]) , flat=True)
                
                for target in flatten_possible_targets:
                    source_target = (start, target)
                    potential_aparts = list(nx.all_simple_paths(graph, start, target, cutoff=int(selected_apart_length)))
                    
                    # filter the aparts that might be smaller
                    filtering_list = define_apartment_validity(graph, potential_aparts, selected_apart_length, MAX_FLOOR)
                    possible_apartments = remove_invalid_apartments(potential_aparts, filtering_list)
                    
                    temporary_apart_dict[source_target] = possible_apartments
                    
                    # remove key, value if the key has no value
                    temporary_apart_dict = {k: v for k, v in temporary_apart_dict.items() if len(v) > 0}
                    temporary_rank_dict = rank_apart_in_dict(graph, temporary_apart_dict, min_max_circulation)

                selected_key, selected_index = select_apart_from_dict(temporary_rank_dict)
                
                if not temporary_apart_dict:
                    continue
                
                # select the current apartment
                selected_apartment = temporary_apart_dict[selected_key][selected_index]
                apartment_dict[selected_apart_length].append(selected_apartment)
                apartment_list.append(selected_apartment)
                
                graph.remove_nodes_from(selected_apartment)
                later_check_starts = [x for x in later_check_starts if x not in selected_apartment]
                print(f"Space successfully filled with apartment {selected_apartment}")
                break
    
    # print the final results
    print("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print(f"||||||||||||||||||||| GENERATION {iteration} COMPLETED ||||||||||||||||||||||")
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"final number of generated apartment is: {len([apart for sublist in apartment_dict.values() for apart in sublist])}")
    print(f"total number of blocs used is: {len(list(flatten(apartment_dict.values())))}\n")
    
    print(f"final generated apartments:\n{apartment_dict}")

    return apartment_dict


def print_recap(dict_final_aparts, desired_proportions):
    """
    Print a recap of the generated apartments
    """
    print("\n\n\n###############################################################################")
    print("############################# FINAL RESULTS ###################################")
    print("###############################################################################\n")
    
    print(f"total number of generated apartments: {len([apart for sublist in dict_final_aparts.values() for apart in sublist])}")
    print(f"total number of blocs used: {len(list(flatten(dict_final_aparts.values())))}\n")
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
        current_flattened_aparts = [apart for sublist in dict_of_dicts[key].values() for apart in sublist]
        stored_flattened_aparts = [apart for sublist in best_dict.values() for apart in sublist]
        if len(current_flattened_aparts) > len(stored_flattened_aparts):
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


def select_apart_from_dict(rank_dict):
    """
    select the best apartment from the dict #copilot
    """
    all_values = list(flatten(rank_dict.values()))
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


def define_apartment_validity(graph, possible_apartments, selected_apart_length, max_floor_count):
    # filter the aparts that might be smaller than the selected_apart_length
    apartments_length_validity = [len(x) == int(selected_apart_length) for x in possible_apartments]
    
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
    
    

def rank_possible_apartments(graph, possible_apartments, circulation_min_max):

    # rank the aparts based on the computed compacity divide by 2 cause Digraph
    compacity_rankings = compute_compacity(graph, possible_apartments)
    # print("-- compacity_rankings ", compacity_rankings)

    # rank the aparts based on the sum of the circulation distance of each nodes
    # circulation_rankings = compute_circulation(graph, possible_apartments)
    circulation_rankings = compute_circulation_2(graph, possible_apartments, circulation_min_max)
    # print("-- circulation_rankings ", circulation_rankings)
    

    
    # sort the paths based on a rule between the two previous rankings
    # interpolate between circulation and compacity rankings depending on COMPACITY
    possible_apartment_rankings = [x * (1 - COMPACITY) + y * COMPACITY for x, y in zip(circulation_rankings, compacity_rankings)]
    # possible_apartment_rankings = [x + 5*y for x, y in zip(circulation_rankings, compacity_rankings)]
         
    return possible_apartment_rankings


def compute_post_generation_rankings(final_graph_aparts, generated_apartments):
    
    # rank the aparts based on the sum of the circulation distance of each nodes
    circulation_rankings = compute_circulation(final_graph_aparts, generated_apartments)
    
    # rank the aparts based on the computed compacity divide by 2 cause Digraph
    compacity_rankings = compute_compacity(final_graph_aparts, generated_apartments)
    
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

def rank_apart_in_dict(graph, apart_dict, min_max_circulation):
    """
    create a dict of the same structure as apart_dict, but with the ranking of each possible apartment
    """
    rank_dict = {}
    for key, value in apart_dict.items():
        rank_dict[key] = rank_possible_apartments(graph, value, min_max_circulation)

    return rank_dict

def compute_compacity(graph, apartments):
    # precalulated values for the 27 first blocks
    block_count =           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    min_compacity =         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    
    max_compacity_flat =    [0, 1, 2, 4, 5, 7, 8, 10, 12, 13, 15 ,17, 18, 20, 22, 24, 25, 27, 29, 31, 32, 34, 36, 38, 40, 41, 43]
    max_compacity_duplex =  [0, 1, 2, 4, 5, 7, 9, 12, 13, 15, 17, 20, 21, 23, 25, 28, 30, 33, 34, 36, 38, 41, 43, 46, 47, 49, 51]
    max_compacity_triplex = [0, 1, 2, 4, 5, 7, 9, 12, 13, 15, 17, 20, 21, 23, 25, 28, 30, 33, 34, 36, 38, 41, 43, 46, 48, 51, 54]
    max_compacities = [max_compacity_flat, max_compacity_duplex, max_compacity_triplex]
    
    compacity_index = min(len(max_compacities), MAX_FLOOR) - 1 # get the global variable
    max_compacity = max_compacities[compacity_index]
    
    compacity_list = []
    for apartment in apartments:
        edge_count = round(len(graph.subgraph(apartment).edges()) / 2)
        index = len(apartment) - 1
        
        divide_by_zero_check = (max_compacity[index] - min_compacity[index]) == 0
        
        # depending on the number of nodes in the apartment, evaluate a compacity value between 0 and 1 by checking min_compacity and max_compacity and edge count
        if len(apartment) < 28 and not divide_by_zero_check:
            compacity = (edge_count - min_compacity[index]) / (max_compacity[index] - min_compacity[index])
            compacity_list.append(compacity)
        
        elif len(apartment) < 28 and divide_by_zero_check:
            compacity = 1
            compacity_list.append(compacity)
        else:
             # raise exception message
             raise Exception("The apartment has more than 27 nodes, the compacity can't be computed")
        
    return compacity_list


def compute_circulation(graph, apartments):
    # precalulated values for the 27 first blocks
    block_count =     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    min_circulation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    max_circulation = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378]

    
    circulation_list = []
    for apartment in apartments:
        circulation_value = sum([graph.nodes[x][CIRCULATION_DISTANCE] for x in apartment])
        index = len(apartment) - 1
        
        divide_by_zero_check = (max_circulation[index] - min_circulation[index]) == 0
        
        # depending on the number of nodes in the apartment, evaluate a circulation value between 0 and 1 by checking min_circulation and max_circulation
        if len(apartment) < 28 and not divide_by_zero_check:
            circulation = (circulation_value - min_circulation[index]) / (max_circulation[index] - min_circulation[index])
            circulation_list.append(circulation)
        elif len(apartment) < 28 and divide_by_zero_check:
            circulation = 1
            circulation_list.append(circulation)
        else:
            # raise exception message
            raise Exception("The apartment has more than 27 nodes, the circulation can't be computed")
        
    return circulation_list


def compute_circulation_2(graph, apartments, circulation_dict):
    '''
    block_count = list(circulation_dict.keys())
    # get the first value of the first key
    min_circulation = list(circulation_dict.values())[0]
    # get the last value of the last key
    max_circulation = list(circulation_dict.values())[-1]
    '''

    # print("\n---- Entering compute_circulation_2 ----\n")
    # print(f"apartments: {apartments}")

    block_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    # circulation range = list of values from the first key to the last key
    circulation_ranges = dict.fromkeys(list(circulation_dict.keys()), [])

    # print("-- circulation_ranges ", circulation_ranges)
    # print("-- circulation_dict ", circulation_dict)
    for key, value in circulation_dict.items():
        circulation_ranges[key] = list(frange(value[0], value[1], len(block_count)))
        # print("==== circulation_ranges ", circulation_ranges)


    # let a value X be a circulation value
    # evaluate that circulation value in the circulation range
    # if the value equals the first values of the range, then the circulation value is 0
    # if the value equals the last values of the range, then the circulation value is 1
    # if the value is between the first and last values of the range, then the circulation is a certain ratio between 0 and 1

    circulation_list = []

    for key, values in circulation_dict.items():
        for apartment in apartments:
            current_size = len(apartment)

            circulation_value = sum([graph.nodes[x][CIRCULATION_DISTANCE] for x in apartment])

            # if the circulation value is equal to the first value of the range, then the circulation value is 0
            if circulation_value == values[0]:
                circulation = 0
                circulation_list.append(circulation)
            # if the circulation value is equal to the last value of the range, then the circulation value is 1
            elif circulation_value == values[-1]:
                circulation = 1
                circulation_list.append(circulation)
            # if the circulation value is between the first and last values of the range, then the circulation is a certain ratio between 0 and 1
            elif circulation_value > values[0] and circulation_value < values[-1]:
                circulation = (circulation_value - values[0]) / (values[-1] - values[0])
                circulation_list.append(circulation)

            # print("!!!!! circulation_value ", circulation_value)
            # print("----")
            # print(f"circulation_list {circulation_list}")
    # print("---------------------------\n")
    return circulation_list


@timeit
def compute_circulation_minmax_precise(graph, settings, circulation_nodes, starting_nodes, apart_sizes, max_floor):
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
    

def exclusive_uniform(a, b):
    """returns a random value in the interval  [a, b]"""
    return a+(b-a)*random.random()


def distance_constrained_shuffle(sequence, distance, my_seed, randmoveforward=exclusive_uniform):
    # https://stackoverflow.com/a/30784808/10235237
    # for max distance, use len(sequence)*len(sequence), needs to be triple checked though
    
    def sort_criterion(enumerate_tuple):
        """
        returns the index plus a random offset,
        such that the result can overtake at most 'distance' elements
        """
        indx, _ = enumerate_tuple
        
        # set new seed for each random value
        random.seed(my_seed + indx)
        
        return indx + randmoveforward(0, distance + 1)
    
    # get enumerated, shuffled list
    enumerated_result = sorted(enumerate(sequence), key = sort_criterion)
    # remove enumeration
    result = [x for i, x in enumerated_result]
    return result

    
def flatten(L):
    """
    flatten any level of nested lists
    returns a generator
    """
    for item in L:
        try:
            yield from flatten(item)
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
    return json_graph.node_link_data(a_graph, attrs={"link": "edges", "source": "from", "target": "to"})


def digraph_to_string(a_graph):
    return json_graph.node_link_graph(a_graph, directed=True, multigraph=False,
                                      attrs={"link": "edges", "source": "from", "target": "to"})



