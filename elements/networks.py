import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn

from elements.layers import PriorNetworkLayer, IntermediateNetworkLayer


class BayesianBKTNetwork:

    def __init__(self, domain_graph, evaluations):
        self.bayes_net = gum.BayesNet('LearnerGraph')
        self.domain_graph = domain_graph
        self.evaluations = evaluations
        self.knowledge_components = domain_graph.knowledge_components
        self.intermediate_network_layers = []
        self.prior_network_layer = PriorNetworkLayer(self)
        self._set_network_layers()

    def __str__(self):
        gdyn.showTimeSlices(self.bayes_net)
        evs = {}
        for i, evaluation in enumerate(self.evaluations):
            evs[f"Ev{evaluation.get_kc().name}_{i+1}"] = 1
        print(evs)
        if evs:
            gnb.showInference(self.bayes_net)
        else:
            gnb.showInference(self.bayes_net, evs=evs)

        return f"BayesianBKT Network with {len(self.intermediate_network_layers)+1} on " \
               f"{len(self.knowledge_components)} knowledge_components."

    def _set_network_layers(self):
        for i in range(len(self.evaluations)):
            self.intermediate_network_layers.append(IntermediateNetworkLayer(self, self.evaluations[i], i+1))
