"""Implementing the ID3 algorithm"""


class DecisionTree:
    def __init__(self):
        pass



"""
per level:
- determine node value (subset, column) with entropy
- split in options per node value
- if option[subset] !has the same class:
    each option == new level
"""

"""
SO:

read trainset file
use trainset to build tree
    - entropy
    - 

read testset file
see how well the tree classifies testset values
print the results (row by row in an output file, but print total accuracy)

"""