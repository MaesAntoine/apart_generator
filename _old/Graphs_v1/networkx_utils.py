import networkx as nx
import random
from networkx.readwrite import json_graph
import json


class DiGraph_GH(nx.DiGraph):
    def __repr__(self):
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges",
            ]
        )


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
        except TypeError:
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


def read_nodes_attr_from_graph(a_graph, some_attributes: list, some_indexes: list):
    # read specified attributes from selected nodes. If no nodes then read specified attributes from all nodes !

    if some_indexes[0] < 0:
        # if no indices provided, get all indices possible
        some_indexes = list(range(len(a_graph.nodes)))
    value_list = []

    for att in some_attributes:
        for node in range(len(some_indexes)):
            value_list.append(a_graph.nodes[node][att])

    return value_list, len(some_indexes)


def read_edges_attr_from_graph(a_graph, attributes_in: list, indexes_in: list) -> list:
    value_list_out = []
    all_indexes = [list(x) for x in a_graph.edges]
    size_ok = all([len(x) == 2 for x in indexes_in])
    edges_included = all([x in a_graph.edges for x in indexes_in])
    attr_included = True

    if -1 in indexes_in[0]:
        indexes_in = all_indexes
    if not size_ok:
        print("Your edge indexes should be a pair of node index per branch. Not more, not less.")
        return []
    if not edges_included:
        print("At least one of your provided edges is not part of your graph")
        return []
    if not attr_included:
        print("At least one of your provided attributes doesn't seem to be in your graph")
        return []

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


def get_all_possible_targets_old(my_graph, start_index: int, apart_length: int) -> list:
    """all reachable indexes are all the descendants at distance apart_length x - 2"""
    possible_descendant_distances = [x for x in range((apart_length - 1) % 2, apart_length, 2) if x > 0]
    possible_target = []

    for intermediate_distance in possible_descendant_distances:

        descendant = list(nx.descendants_at_distance(my_graph, start_index, intermediate_distance))
        if len(descendant) > 0:
            possible_target.append(descendant)

    return possible_target


def get_all_possible_targets(my_graph, start_index: int, apart_length: list) -> list:

    possible_descendant_distances = list(range(apart_length[0] - 1, apart_length[1], 1)) # triggers "int object not subscriptable"
    possible_target = []

    for intermediate_distance in possible_descendant_distances:

        descendant = list(nx.descendants_at_distance(my_graph, start_index, intermediate_distance))
        if len(descendant) > 0:
            possible_target.append(descendant)
    return possible_target


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
            if my_graph[index][neighbor][attribute] == value:
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


def all_possible_targets_multiple_starts_old(a_graph, start_indexes, length):
    every_possible_targets = []
    for start in start_indexes:
        every_possible_targets.append(get_all_possible_targets(a_graph, start, length))

    return every_possible_targets


def all_possible_targets_multiple_starts(a_graph, start_indexes, length: list):
    every_possible_targets = []
    for start in start_indexes:
        every_possible_targets.append(get_all_possible_targets(a_graph, start, length))
    print("All the start indexes = ", start_indexes)
    print("every_possible_targets = ", every_possible_targets)
    return every_possible_targets


def recursive_remove_empty_nested_list(my_list):
    for index, value in enumerate(reversed(my_list)):
        if isinstance(value, list) and value != []:
            recursive_remove_empty_nested_list(value)
        elif isinstance(value, list) and len(value) == 0:
            my_list.remove(value)


def generate_apartments_v4(a_graph, source_positions, compacity_factor, apart_lengths: list, my_seed):
    copied_graph = a_graph.copy()
    min_length = apart_lengths[0]
    max_length = apart_lengths[-1]
    apartment_list = []

    possible_targets = all_possible_targets_multiple_starts(copied_graph, source_positions, apart_lengths)
    random.seed(my_seed)

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
        print("possible APARTMENTS =====> ", possible_apartment)

        if len(possible_apartment) < 1:
            possible_targets[0][compacity].remove(target)
            if len(possible_targets[0][compacity]) < 1 and possible_targets[0][compacity] in possible_targets:
                possible_targets.remove(possible_targets[0][compacity])
            continue

        filtered_apartment = possible_apartment
        filtered_apartment = [x for x in possible_apartment if len(x) > min_length - 1]
        if len(filtered_apartment) < 1:
            possible_targets[0][compacity].remove(target)
            continue

        filtered_apartment.sort(key=len)
        apartment = filtered_apartment[0]

        apartment_list.append(apartment)
        a_graph = remove_nodes(copied_graph, apartment)     # modify the graph in place, used in next iteration !
        source_positions.remove(source)
        possible_targets.remove(possible_targets[0])

    return copied_graph, apartment_list


def generate_apartments_v5(a_graph, source_positions, compacity_factor, apart_lengths: list, my_seed):
    pass


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


def string_to_graph(a_string_graph):
    a_graph = json.loads(a_string_graph)
    return json_graph.node_link_graph(a_graph, directed=True, multigraph=False,
                                      attrs={"link": "edges", "source": "from", "target": "to"})


def graph_to_string(a_graph):
    return json_graph.node_link_data(a_graph, attrs={"link": "edges", "source": "from", "target": "to"})


def digraph_to_string(a_graph):
    return json_graph.node_link_graph(a_graph, directed=True, multigraph=False,
                                      attrs={"link": "edges", "source": "from", "target": "to"})


def get_nodes_with_attribute(a_graph, attribute, val=0, more=True):
    nodes_at = []

    for (p, d) in a_graph.nodes(data=True):
        if more:
            if d[attribute] > val:
                nodes_at.append(p)
        else:
            if d[attribute] < val:
                nodes_at.append(p)

    return nodes_at


def get_nodes_least_most_attribute(a_graph, attribute, most=True):
    pass


def reorder_indexes(main_list, sub_list, front=True):

    main_list = [x for x in main_list if x not in sub_list]

    if front:
        return sub_list + main_list
    else:
        return main_list + sub_list


