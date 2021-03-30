import gum
from elements.networks import BayesianBKTNetwork

class Node:

    def __init__(self, network=BayesianBKTNetwork()):
        self.network = network


class KnowledgeComponentNode(Node):
    def __init__(self, kc, timestamp):
        gum.LabelizedVariable(f"C{kc.id}_{timestamp}", '', 2)


class ExerciseNode(Node):
    def __init__(self, exercise, timestamp):
        gum.LabelizedVariable(f"E{exercise.kc.id}_{timestamp}", '', 2)

