import gum
from elements.layers import PriorNetworkLayer, IntermediateNetworkLayer


class BayesianBKTNetwork:

    def __init__(self, domain_graph, evaluations):
        self.network = gum.BayesNet('LearnerGraph')
        self.domain_graph = domain_graph
        self.knowledge_components = domain_graph.kc_list
        self.intermediate_network_layers = []
        self.prior_network_layer = PriorNetworkLayer(self)
        self._set_network_layers(evaluations)

    def _set_network_layers(self, evaluations):
        for i in range(len(evaluations)):
            self.intermediate_network_layers.append(IntermediateNetworkLayer(self, evaluations[i], i))
