from enum import Enum

class SaliencyMethod(Enum):
    VANILLA = 1
    GUIDED = 2
    DECONV = 3
    # EXCITATION_BP = 4

class MapType(Enum):
    POSITIVE = 1
    NEGATIVE = 2
    ABSOLUTE = 3
    ORIGINAL = 4
