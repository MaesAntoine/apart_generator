from graph_data import CIRCULATION_DISTANCE

from graph_utils import select_apart_length, get_all_possible_targets, remove_higher_targets, get_all_possible_paths, define_apartment_validity
from graph_utils import remove_invalid_apartments, select_apart_from_dict

from graph_rankings import wieghted_sum_ranking_results, get_ranking_functions, rank_apartments

def generate_apartments(graph, desired_proportions, generation_settings, ranking_settings, circulation_bounds, maximum_floor, fill, iteration):
    """
    Parameters
    ----------
    graph : networkx graph
        The graph that contains the nodes and edges.
    desired_proportions : dict
        The desired proportions of the apartments.
    generation_settings : dict
        The settings for the generation of the apartments.
    ranking_settings : dict
        The settings for the ranking of the apartments.
    circulation_bounds : dict
        The bounds for the circulation of the apartments.
    maximum_floor : int
        The maximum floors that the apartments can have.
    fill : bool
        If True, a second pass is made to fill the enventual gaps.
    iteration : int
        The iteration of the generation.
    
    Performs
    --------
        DESCRIPTION.
    
    Returns
    -------
    apartment_dict : dict
        The generated apartments.
    
    """

    # initiate empty dicts for both apartments and rankings and define the start nodes
    apartment_dict = dict.fromkeys(desired_proportions.keys(), [])
    rankings = dict.fromkeys(desired_proportions.keys(), [])
    start_nodes = [x for x in graph.nodes if graph.nodes[x][CIRCULATION_DISTANCE] == 1]
    # print(f"start nodes: {start_nodes}")

    main_loop = True
    later_check_starts = []

    # while there are start nodes left, generate apartments
    while len(start_nodes) > 0:

        # select the first node
        start_node = start_nodes.pop(0)
        print(f"\nstart node: {start_node}")

        # check if the start node is still in the graph
        if start_node not in graph.nodes:
            # print(f"start node {start_node} is not in the graph anymore, skipping")
            continue

        # select the length of the apartment
        selected_apart_length = select_apart_length(desired_proportions, apartment_dict, main_loop=main_loop)
        # print(f"selected apartment length: {selected_apart_length}")

        # get all possible targets
        possible_targets = get_all_possible_targets(graph, start_node, int(selected_apart_length) - 1, int(generation_settings["max_targets"]), flat=True)
        print(f"possible targets: {possible_targets}")

        # remove the targets that are higher than the maximum floor compared to the start node
        possible_targets = remove_higher_targets(graph, start_node, possible_targets, maximum_floor)

        # create empty temporary dicts for apartments and rankings
        temporary_apart_dict = {}
        temporary_rank_dict = {}

        # for all possible targets
        for target in possible_targets:

            # print(f"\n  target: {target}")
            # get a certain number of possible paths from the start node to the target with a certain length
            potential_apartments = get_all_possible_paths(graph, start_node, target, int(selected_apart_length), int(generation_settings["max_paths"]))
            print(f"  number of potential apartments for target {target}: {len(potential_apartments)}")

            # filter the paths based on validity (not rankings)
            filtering_list = define_apartment_validity(graph, potential_apartments, selected_apart_length, maximum_floor)
            possible_apartments = remove_invalid_apartments(potential_apartments, filtering_list)
            # print(f"possible apartments: {possible_apartments}")

            # check if there are any valid paths left, if not, continue
            if len(possible_apartments) == 0:
                continue


            source_target = (start_node, target)
            temporary_apart_dict[source_target] = possible_apartments

            # rank the valid paths
            ranking_functions = get_ranking_functions(graph, ranking_settings, possible_apartments, circulation_bounds, maximum_floor)
            rankings = rank_apartments(possible_apartments, ranking_functions, ranking_settings)

            # add the valid paths to the temporary dicts for apartments and rankings
            temporary_rank_dict[source_target] = rankings

            
        # if there are no valid paths, add the start node to the list of nodes to check later
        if not temporary_apart_dict:
            # print(f"\nno valid paths found for start node {start_node}, adding to later check list")
            # print(f"temporary apartment dict: {temporary_apart_dict}\n")
            later_check_starts.append(start_node)
            continue

        # sum the rankings of the ranking dict based on the ranking settings (as weights)
        weighted_ranking_result_dict = wieghted_sum_ranking_results(temporary_rank_dict, ranking_settings)

        # select the best apartment (based on rankings, and with a random choice if there are multiple bests)
        selected_key, selected_index = select_apart_from_dict(weighted_ranking_result_dict, iteration)
        selected_apartment = temporary_apart_dict[selected_key][selected_index]

        # add the selected apartment to the apartment dict
        current_apartments = apartment_dict[selected_apart_length]
        updated_apartments = current_apartments + [selected_apartment]
        apartment_dict[selected_apart_length] = updated_apartments

        # remove the selected apartment from the graph
        graph.remove_nodes_from(selected_apartment)


    # =========== MAIN LOOP END ===========
    # Let's try to fill the remaing gaps !

    if fill and len(later_check_starts) > 0:
        main_loop = False

        for start_node in later_check_starts:
            # as we check for potential new filling aparts, check if it's actually still in the graph
            if start_node not in graph.nodes:
                continue
            
            # create a stack of the desired apartment lengths to satisfy the proportions (ordered lengthes)
            stack = select_apart_length(desired_proportions, apartment_dict, main_loop=main_loop)

            while stack:
                selected_apart_length = stack.pop(0)

                # get all possible targets
                possible_targets = get_all_possible_targets(graph, start_node, int(selected_apart_length) - 1, int(generation_settings["max_targets"]), flat=True)
                # print(f"possible targets: {possible_targets}")

                # remove the targets that are higher than the maximum floor compared to the start node
                possible_targets = remove_higher_targets(graph, start_node, possible_targets, maximum_floor)

                # create empty temporary dicts for apartments and rankings
                temporary_apart_dict = {}
                temporary_rank_dict = {}

                # for all possible targets
                for target in possible_targets:

                    # get a certain number of possible paths from the start node to the target with a certain length
                    potential_apartments = get_all_possible_paths(graph, start_node, target, int(selected_apart_length), int(generation_settings["max_paths"]))
                    # print(f"\npotential apartments: {potential_apartments}")

                    # filter the paths based on validity (not rankings)
                    filtering_list = define_apartment_validity(graph, potential_apartments, selected_apart_length, maximum_floor)
                    possible_apartments = remove_invalid_apartments(potential_apartments, filtering_list)
                    # print(f"possible apartments: {possible_apartments}")

                    # check if there are any valid paths left, if not, continue
                    if len(possible_apartments) == 0:
                        continue

                    source_target = (start_node, target)
                    temporary_apart_dict[source_target] = possible_apartments

                    # rank the valid paths
                    ranking_functions = get_ranking_functions(graph, ranking_settings, possible_apartments, circulation_bounds, maximum_floor)
                    rankings = rank_apartments(possible_apartments, ranking_functions, ranking_settings)

                    # add the valid paths to the temporary dicts for apartments and rankings
                    temporary_rank_dict[source_target] = rankings

                    
                # if there are no valid paths, add the start node to the list of nodes to check later
                if not temporary_apart_dict:
                    later_check_starts.append(start_node)
                    continue

                # sum the rankings of the ranking dict based on the ranking settings (as weights)
                weighted_ranking_result_dict = wieghted_sum_ranking_results(temporary_rank_dict, ranking_settings)

                # select the best apartment (based on rankings, and with a random choice if there are multiple bests)
                selected_key, selected_index = select_apart_from_dict(weighted_ranking_result_dict, iteration)
                selected_apartment = temporary_apart_dict[selected_key][selected_index]

                # add the selected apartment to the apartment dict
                current_apartments = apartment_dict[selected_apart_length]
                updated_apartments = current_apartments + [selected_apartment]
                apartment_dict[selected_apart_length] = updated_apartments

                # remove the selected apartment from the graph
                graph.remove_nodes_from(selected_apartment)
                # break to check the next start node
                break

            # remove start_node from the graph. Can trigger endless loop if not  done
            if start_node in graph.nodes:
                graph.remove_node(start_node)

    return apartment_dict
