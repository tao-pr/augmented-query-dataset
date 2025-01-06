from enum import Enum

class RunMode(Enum):
    UNKNOWN = -1
    GENERATOR = 1
    AUGMENTOR = 2
    VALIDATOR = 8
    MERGER = 16
