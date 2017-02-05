from abc import ABC, abstractmethod


class Node(ABC):

    @abstractmethod
    def foo(self):
        pass


class TerminalNode(Node):

    def __init__(self, classification):
        self._classification = classification

    def get_classification(self):
        return self._classification

    def foo(self):
        pass



class RootNode(Node):

    def __init__(self, right_node, left_node):
        self.left_node = left_node
        self.right_node = right_node

    def foo(self):
        pass
