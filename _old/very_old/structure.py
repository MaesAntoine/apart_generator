'''
def generate_apartment(graph, list of start indexes, compacity, length)

    select one position from the list of start indexes
    evaluate all possible targets
    select a target randomly from the list of targets (also based on compacity)
    evaluate all possible apartments for the selected target

    if list of apartments NOT empty
        select one of the apartments (randomly for now)
        remove nodes from the graph
        return the graph and the freshly created apartment and selected start

    remove target from list

    if length of list of target NOT empty
        generate_apartment(graph, list of start indexes, compacity, length) #ouuuh, big boi recursion, hashtag flex !
    else
        remove selected start from list of start indexes
        if length of list of starts NOT empty
            generate_apartment(graph, list of start indexes, compacity, length) #daaamn, serious business !
        else
            return



    VERSION 2

    def generate_apartment(graph, list of starts, super list of targets, compacity, length)
        select one position from the list of start indexes
        get the index on that selected start position
        select the group of possible targets from super list of targets (use the index previously created)
        define compacity
        select sub-group of targets using compacity
        select one target randomly
        evaluate all possible apartments for the selected target

        if length of all possible apartment is NOT empty
            select one apartment (randomly for now)
            remove the nodes from the graph
            remove selected start position from list of starts          |
            remove selected target from super list of targets           |   function ??
            remove empty list in group of targets if necessary !!       |
            return updated graph, list of starts, super list of targets
        else
            remove selected start position from list of starts          |
            remove selected target from super list of targets           |   function ??
            remove empty list in group of targets if necessary !!       |

        if length of list of sub list of target NOT empty
            generate_apartment(graph, list of starts, super list of targets, compacity, length)
        else
            remove selected start from list of starts
            if length of list of start NOT empty
                generate_apartment(graph, list of starts, super list of targets, compacity, length)
            else
                return




    VERSION 3 fuck recursion?

    def generate_apartment()

        define empty apartment list
        evaluate all possible targets for each possible starts (super target list)

        while start in possible starts

            select the first available start
            select the possible targets for that start

            if no target for that start (for any compacity)
                remove that start from the list of starts
                clean any possible empty lists in super target list
                continue

            define compacity (using compacity factor and the possible targets for the specific start)
            select sub group of targets with the compacity value
            select one of the possible targets ( based on selected start and compacity)
            evaluate all possible apartments

            if all possible apartments empty
                remove selected target from super target list
                clean any possible empty lists in super target list
                continue

            select random apartment
            remove nodes from graph
            remove selected target from super target list
            remove selected start
            add selected apartment to the apartment list

        return apartment list







------------------
program
-------


get all the possible starting positions
create empty list of apartments

while start positions available:

    graph, apartment, used start = generate_apartment(graph, list of start indexes, compacity, length)

    add apartment to the list of apartments
    remove used start from start positions
    remove used target from list of targets
    '''