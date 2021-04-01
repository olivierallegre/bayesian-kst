from elements.nodes import KnowledgeComponentNode, ExerciseNode, EvidenceNode
from kgraph.helpers.truthtable import truthtable


forget = 0.
prf = 1
prl = 1

class NetworkLayer:

    def __init__(self, network):
        self.network = network


class PriorNetworkLayer(NetworkLayer):

    def __init__(self, network):
        NetworkLayer.__init__(self, network)
        self._set_layer_internal_nodes()
        self._set_layer_internal_links()

    def _set_layer_internal_nodes(self):
        for kc in self.network.domain_graph.knowledge_components:
            node = KnowledgeComponentNode(self.network, kc, 0)
            print(node)
            self.network.bayes_net.add(node.variable)
            print(self.network.bayes_net.nodes())
            m_pba = 0.5
            self.network.bayes_net.cpt(f"C{kc.name}_0").fillWith([1 - m_pba, m_pba])

    def _set_layer_internal_links(self):
        links = self.network.domain_graph.link_model.links
        for kc in self.network.knowledge_components:
            if kc in links.keys():
                if links[kc]["from_parents"]:
                    parents = links[kc]["from_parents"].linked_knowledge_components
                    n_parents = len(parents)
                    for parent in parents:
                        self.network.bayes_net.addArc(*(f"C{parent.name}_{0}", f"C{kc.name}_{0}"))
                    p_vec = links[kc]["from_parents"].probability_vector
                    truthtable_lst = truthtable(n_parents)
                    for i in range(len(truthtable_lst)):
                        self.network.bayes_net.cpt(f"C{kc.name}_{0}")[{f"C{parents[k].name}_{0}": truthtable_lst[i][k] for k in range(n_parents)}] = [1-p_vec[i], p_vec[i]]


class IntermediateNetworkLayer(NetworkLayer):

    def __init__(self, network, evaluation, step):
        NetworkLayer.__init__(self, network)
        self._set_layer_internal_nodes(evaluation, step)
        """        self._set_layer_internal_links(evaluation, step)
                self._set_layer_external_links(step)
        """
        self._set_layer_links(evaluation, step)
    def _set_layer_internal_nodes(self, evaluation, step):
        for kc in self.network.domain_graph.knowledge_components:
            node = KnowledgeComponentNode(self.network, kc, step)
            self.network.bayes_net.add(node.variable)
        self.network.bayes_net.add(ExerciseNode(self.network, evaluation.exercise_family, step).variable)
        #self.network.bayes_net.add(EvidenceNode(self.network, evaluation.exercise_family, step).variable)
        m_pba = 0.8
        kc = evaluation.get_kc()
        self.network.bayes_net.cpt(f"E{kc.name}_{step}").fillWith([1 - m_pba, m_pba])

    def _set_layer_links(self, evaluation, step):
        evaluated_kc = evaluation.get_kc()
        links = self.network.domain_graph.link_model.links
        self.network.bayes_net.addArc(*(f"E{evaluated_kc.name}_{step}", f"C{evaluated_kc.name}_{step}"))
        #self.network.bayes_net.addArc(*(f"E{evaluated_kc.name}_{step}", f"Ev{evaluated_kc.name}_{step}"))

        for kc in self.network.domain_graph.knowledge_components:
            self.network.bayes_net.addArc(*(f"C{kc.name}_{step - 1}", f"C{kc.name}_{step}"))
            if kc is evaluated_kc:
                self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False,
                                                                  f"E{kc.name}_{step}": False}] = [1, 0]
                self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False,
                                                                  f"E{kc.name}_{step}": True}] = [1 - prl, prl]
                self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True,
                                                                  f"E{kc.name}_{step}": False}] = [1-(1-forget)*(1-prf),
                                                                                                   (1-forget)*(1-prf)]
                self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True,
                                                                  f"E{kc.name}_{step}": True}] = [0, 1]
            elif kc in links.keys():
                if evaluated_kc in links[kc]["from_parents"].linked_knowledge_components:
                    self.network.bayes_net.addArc(*(f"E{evaluated_kc.name}_{step}", f"C{kc.name}_{step}"))
                    #p_cond = get_conditional_probability()
                    p_cond = [0.1, 0.9]  # in shape [p(Y|noX), p(Y|X)]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False,
                                                                      f"E{evaluated_kc.name}_{step}": False}] = [1, 0]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False,
                                                                      f"E{evaluated_kc.name}_{step}": True}] = [
                        1 - prl*p_cond[1], prl*p_cond[1]]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True,
                                                                      f"E{evaluated_kc.name}_{step}": False}] = [
                        1 - (1-forget)* prf * p_cond[0], (1-forget)* prf * p_cond[0]]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True,
                                                                      f"E{evaluated_kc.name}_{step}": True}] = [
                        1 - (1-forget) * p_cond[1], (1-forget) * p_cond[1]]
                elif evaluated_kc in links[kc]["from_children"].linked_knowledge_components:
                    # p_cond = get_conditional_probability()
                    p_cond = [0.1, 0.9]  # in shape [p(Y|noX), p(Y|X)]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False,
                                                                      f"E{evaluated_kc.name}_{step}": False}] = [1, 0]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False,
                                                                      f"E{evaluated_kc.name}_{step}": True}] = [
                        1 - prl*p_cond[1], prl*p_cond[1]]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True,
                                                                      f"E{evaluated_kc.name}_{step}": False}] = [
                        1 - (1-forget)* prf * p_cond[0], (1-forget)* prf * p_cond[0]]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True,
                                                                      f"E{evaluated_kc.name}_{step}": True}] = [
                        1 - (1-forget) * p_cond[1], (1-forget) * p_cond[1]]
                else:
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False}] = [1, 0]
                    self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True}] = [forget, 1-forget]

            else:
                self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step - 1}": False}] = [1, 0]
                self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step - 1}": True}] = [forget, 1 - forget]


        """if evaluated_kc in links.keys():
            if links[evaluated_kc]["from_parents"]:
                parents = links[evaluated_kc]["from_parents"].linked_knowledge_components
                n_parents = len(parents)
                for parent in parents:
                    self.network.bayes_net.addArc(*(f"C{parent.name}_{step}", f"E{evaluated_kc.name}_{step}"))


                p_vec = links[evaluated_kc]["from_parents"].probability_vector
                truthtable_lst = truthtable(n_parents)
                for i in range(len(truthtable_lst)):

                    self.network.bayes_net.cpt(f"E{evaluated_kc.name}_{step}")[{
                        **{f"C{evaluated_kc.name}_{step}": True},
                        **{f"C{parents[k].name}_{step}": truthtable_lst[i][k] for k in range(n_parents)}}] = [
                        1 - p_vec[i], p_vec[i]]
                    self.network.bayes_net.cpt(f"E{evaluated_kc.name}_{step}")[{
                        **{f"C{evaluated_kc.name}_{step}": False},
                        **{f"C{parents[k].name}_{step}": truthtable_lst[i][k] for k in range(n_parents)}}] = [1, 0]
        else:
            self.network.bayes_net.cpt(f"E{evaluated_kc.name}_{step}")[{f"C{evaluated_kc.name}_{step}": True}] = [0, 1]
            self.network.bayes_net.cpt(f"E{evaluated_kc.name}_{step}")[{f"C{evaluated_kc.name}_{step}": False}] = [1, 0]
        s_pba = evaluation.get_success_probability()
        self.network.bayes_net.cpt(f"Ev{evaluated_kc.name}_{step}")[{f"E{evaluated_kc.name}_{step}": True}] = [
            1 - s_pba, s_pba]
        self.network.bayes_net.cpt(f"Ev{evaluated_kc.name}_{step}")[{f"E{evaluated_kc.name}_{step}": False}] = [1, 0]
    
    def _set_layer_external_links(self, step):
        for kc in self.network.domain_graph.knowledge_components:
            self.network.bayes_net.addArc(*(f"C{kc.name}_{step-1}", f"C{kc.name}_{step}"))
            self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": True}] = [0, 1]
            self.network.bayes_net.cpt(f"C{kc.name}_{step}")[{f"C{kc.name}_{step-1}": False}] = [1, 0]
        """
