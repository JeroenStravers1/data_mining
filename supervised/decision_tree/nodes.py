
class Node:

    def __init__(self):
        self._decision_rule = ""
        self._children = dict()
        self._class = None

    def get_decision_rule(self):
        return self._decision_rule

    def set_decision_rule(self, feature):
        self._decision_rule = feature

    def get_children(self):
        return self._children

    def add_children(self, name, value):
        self._children[name] = value

    def get_class(self):
        return self._class

    def set_class(self, value):
        self._class = value
