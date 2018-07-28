from enum import Enum

class SaliencyMethod(Enum):
    # VANILLA = 1
    # GUIDED = 2
    # DECONV = 3
    # TEST = 4
    # PERTURBATION = 5
    PERTURBATION_2 = 1
    #LIME = 7
    #EXCITATIONBP = 6


class MapType(Enum):
    POSITIVE = 1
    NEGATIVE = 2
    ABSOLUTE = 3
    ORIGINAL = 4
    INPUT = 5
