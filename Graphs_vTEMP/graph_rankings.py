# This file contains the functions that are used to rank the apartments
# The ranking functions are selected from Grasshopper (when the user sets a value in the specific ranking input)

# To add a new ranking function you will need to:
# 1. Add a new stand alone ranking function to this file
# 2. Add the function to the get_ranking_functions function
# 3. Back in the main file (graph_components.py), under SETTINGS>ranking_settings component, 
#   add a new input for the ranking function and check if it is selected


from graph_data import CIRCULATION_DISTANCE, min_compacity, max_compacities_dict
import graph_utils



def wieghted_sum_ranking_results(ranking_dict, ranking_settings):
    """
    Sum the rankings of the ranking dict based on the ranking settings

    each ranking function has a weight, which is defined in the ranking settings (values from grasshopper)
    the ranking functions are summed up, and the apartment with the highest sum will be selected after this function
    """
    
    summed_rankings = ranking_dict.copy() # cannot dict.fromkeys cause nested dicts.

    for source_target in ranking_dict.keys():

        for function in ranking_settings.keys():

            # this is the weight of the ranking function, comming from the grasshopper
            value = ranking_settings[function]
            updated_values = [x * value for x in ranking_dict[source_target][function]]

            # replace the old values with the new values
            summed_rankings[source_target][function] = updated_values
    
        # sum the values of the ranking functions, no matter how many there are
        summed_rankings[source_target] = [sum(x) for x in zip(*summed_rankings[source_target].values())]

    return summed_rankings


def get_ranking_functions(the_graph, rank_settings, possible_apartments, circulation_bounds, max_floor):
    """
    THIS IS WHERE YOU CAN ADD NEW RANKING FUNCTIONS

    for each ranking settings, add a new ranking function to the ranking_functions dict
    the items in the dict are the function itself, and the values are the arguments that need to be passed to the function
    """

    ranking_functions = {}

    if "_compacity" in rank_settings:
        compacities = (min_compacity, max_compacities_dict)
        ranking_functions[rank_compacity] = [the_graph, possible_apartments, max_floor, compacities]
        # print(f"ranking functions: {ranking_functions}")
    
    if "_circulation" in rank_settings:
        ranking_functions[rank_circulation] = [the_graph, possible_apartments, circulation_bounds]
        # print(f"ranking functions: {ranking_functions}")
    
    if "_windows" in rank_settings:
        ranking_functions[rank_windows] = [the_graph, possible_apartments]
        # print(f"ranking functions: {ranking_functions}")

    return ranking_functions


def rank_apartments(apartment_list, ranking_functions, ranking_settings):
    """
    Rank the apartments in the apartment dict
    Returns a dict 
    """

    # initiate empty ranking dict
    rankings = dict.fromkeys(ranking_settings.keys(), {})

    # for function in the ranking_functions dict
    for index in range(len(ranking_settings.keys())):
        key = list(ranking_functions.keys())[index]
        cute_name_key = list(ranking_settings.keys())[index]

        # for apartment in the apartment dict
        for _ in apartment_list:

            # get the ranking for the current apartment
            ranking = key(*ranking_functions[key])

            # add the ranking to the ranking dict
            rankings[cute_name_key] = ranking
            
    return rankings


def rank_compacity(graph, apartments, max_floor, compacity_data):

    # get the compacity data from tuple (min compacity is a list, max compacity is a dict cause multiple lengthes)
    min_compacity, max_compacities_dict = compacity_data
    
    compacity_index = min(len(max_compacities_dict), max_floor) - 1
    max_compacity = max_compacities_dict[compacity_index]
    
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


def rank_circulation(graph, apartments, circulation_bounds):
    '''
    block_count = list(circulation_bounds.keys())
    # get the first value of the first key
    min_circulation = list(circulation_bounds.values())[0]
    # get the last value of the last key
    max_circulation = list(circulation_bounds.values())[-1]
    '''

    # print("\n---- Entering compute_circulation_2 ----\n")
    # print(f"apartments: {apartments}")

    block_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    # circulation range = list of values from the first key to the last key
    circulation_ranges = dict.fromkeys(list(circulation_bounds.keys()), [])

    # print("-- circulation_ranges ", circulation_ranges)
    # print("-- circulation_bounds ", circulation_bounds)
    for key, value in circulation_bounds.items():
        circulation_ranges[key] = list(graph_utils.float_range(value[0], value[1], len(block_count)))
        # print("==== circulation_ranges ", circulation_ranges)


    # let a value X be a circulation value
    # evaluate that circulation value in the circulation range
    # if the value equals the first values of the range, then the circulation value is 0
    # if the value equals the last values of the range, then the circulation value is 1
    # if the value is between the first and last values of the range, then the circulation is a certain ratio between 0 and 1

    circulation_list = []


    for apartment in apartments:
        current_size = str(len(apartment))
        values = circulation_bounds[current_size]

        circulation_value = sum([graph.nodes[x][CIRCULATION_DISTANCE] for x in apartment])

        # if the circulation value is equal to the first value of the range, then the circulation value is 0
        if circulation_value == values[0]:
            circulation = 0
            circulation_list.append(circulation)
        # if the circulation value is equal to the last value of the range, then the circulation value is 1
        elif circulation_value >= values[-1]:
            circulation = 1
            circulation_list.append(circulation)
        # if the circulation value is between the first and last values of the range, 
        # then the circulation is a certain ratio between 0 and 1
        elif circulation_value > values[0] and circulation_value < values[-1]:
            circulation = (circulation_value - values[0]) / (values[-1] - values[0])
            circulation_list.append(circulation)

    return circulation_list


def rank_windows(graph, apartments):
    """
    Rank the apartments based on the number of windows
    """

    # initiate empty list
    window_list = []

    for apartment in apartments:
        # get the number of windows in the apartment
        window_count = sum([graph.nodes[x]["window_count"] for x in apartment])
        # add the number of windows to the list
        window_list.append(window_count)

    return window_list