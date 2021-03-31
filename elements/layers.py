import gum
from elements.nodes import Node, KnowledgeComponentNode, ExerciseNode


class NetworkLayer:

    def __init__(self, network):
        self.network = network
        self.nodes = network.nodes


class PriorNetworkLayer(NetworkLayer):

    def __init__(self, network):
        NetworkLayer.__init__(self, network)
        self._set_layer_internal_nodes()
        self._set_layer_internal_links()

    def _set_layer_internal_nodes(self):
        for kc in self.network.domain_graph.kc_list:
            node = KnowledgeComponentNode(kc, 0)
            self.network.add(node)

    def _set_layer_internal_links(self):
        links = self.network.domain_graph.link_model.links
        assert links.keys().issubset(self.network.knowledge_components)
        for kc in self.network.knowledge_components:
            if links[kc]["from_parents"]:
                parents = links[kc]["from_parents"].linked_knowledge_components
                n_parents = len(parents)
                for parent in parents:
                    self.network.addArc(*(f"C{parent.name}_{0}", f"C{kc.name}_{0}"))
                p_vec = links[kc]["from_parents"].probability_vector
                truthtable = expert_layer.truthtable(n_parents)
                for i in range(len(truthtable)):
                    self.network.cpt(f"C{kc.name}_{0}")[{f"C{parents[k].name}_{0}": truthtable[i][k] for k in range(
                        n_parents)}] = [1-p_vec[i], p_vec[i]]


class IntermediateNetworkLayer(NetworkLayer):

    def __init__(self, network, exercise, step):
        NetworkLayer.__init__(self, network)
        self._set_layer_internal_nodes(exercise, step)
        self._set_layer_internal_links(exercise)
        self._set_layer_external_links()

    def _set_layer_internal_nodes(self, exercise, step):
        for kc in self.network.domain_graph.kc_list:
            node = KnowledgeComponentNode(kc, step)
            self.network.add(node)
        self.network.add(ExerciseNode(exercise, step))

    def _set_layer_internal_links(self, exercise, step):
        evaluated_kc = exercise.kc
        links = self.network.domain_graph.link_model.links
        if links[evaluated_kc]["from_parents"]:
            parents = links[evaluated_kc]["from_parents"].linked_knowledge_components
            n_parents = len(parents)
            for parent in parents:
                self.network.addArc(*(f"C{parent.name}_{step}", f"E{evaluated_kc.name}_{step}"))
            p_vec = links[evaluated_kc]["from_parents"].probability_vector
            truthtable = expert_layer.truthtable(n_parents)
            for i in range(len(truthtable)):
                self.network.cpt(f"E{evaluated_kc.name}_{step}")[{f"C{parents[k].name}_{step}": truthtable[i][k] for k in range(n_parents)}] = [1-p_vec[i], p_vec[i]]

    def _set_layer_external_links(self, step):
        for kc in self.network.domain_graph.kc_list:
            self.network.addArc(*(f"C{kc.name}_{step-1}", f"C{kc.name}_{step}"))
            self.network.cpt(f"{kc.name}_{step}")[{f"{kc.name}_{step-1}": True}] = [0, 1]
            self.network.cpt(f"{kc.name}_{step}")[{f"{kc.name}_{step-1}": False}] = [1, 0]
