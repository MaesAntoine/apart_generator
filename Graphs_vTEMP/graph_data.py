

## COMPACITY DATA FOR RANKINGS ##

# precalulated values for the 27 first blocks
min_compacity = list(range(27))

max_compacity_flat =    [0, 1, 2, 4, 5, 7, 8, 10, 12, 13, 15 ,17, 18, 20, 22, 24, 25, 27, 29, 31, 32, 34, 36, 38, 40, 41, 43]
max_compacity_duplex =  [0, 1, 2, 4, 5, 7, 9, 12, 13, 15, 17, 20, 21, 23, 25, 28, 30, 33, 34, 36, 38, 41, 43, 46, 47, 49, 51]
max_compacity_triplex = [0, 1, 2, 4, 5, 7, 9, 12, 13, 15, 17, 20, 21, 23, 25, 28, 30, 33, 34, 36, 38, 41, 43, 46, 48, 51, 54]

max_compacities_dict = {0: max_compacity_flat, 1: max_compacity_duplex, 2: max_compacity_triplex}

# CONSTANTS
BLOCK_TYPE = "BLOCK_TYPE"
CIRCULATION_TYPE = "CIRCULATION"
APARTMENT_TYPE = "APARTMENT"
CIRCULATION_DISTANCE = "CIRCULATION_DISTANCE"