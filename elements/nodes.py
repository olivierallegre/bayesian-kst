import pyAgrum as gum


class Node:

    def __init__(self, network):
        self.network = network


class KnowledgeComponentNode(Node):
    def __init__(self, network, kc, timestamp):
        Node.__init__(self, network)
        self.variable = gum.LabelizedVariable(f"C{kc.name}_{timestamp}", '', 2)


class ExerciseNode(Node):
    def __init__(self, network, exercise, timestamp):
        Node.__init__(self, network)
        print(exercise)
        self.variable = gum.LabelizedVariable(f"E{exercise.kc.name}_{timestamp}", '', 2)


class EvidenceNode(Node):
    def __init__(self, network, exercise, timestamp):
        Node.__init__(self, network)
        print(exercise)
        self.variable = gum.LabelizedVariable(f"Ev{exercise.kc.name}_{timestamp}", '', 2)
