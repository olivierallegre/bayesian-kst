import copy
import tqdm
import numpy as np
from hyperopt import hp, tpe, fmin
from kgraph.expert_layer.knowledge_components import KnowledgeComponent, DeclarativeKnowledgeComponent, \
    ProceduralKnowledgeComponent
from kgraph.helpers.expected_scoring import expected_declarative_kc_scoring, expected_procedural_kc_scoring
from kgraph.helpers.truthtable import get_representative_answers, truthtable


class LearnerGraph(object):

    def __init__(self, learner, domain_graph):
        """
        Constructor of LearnerGraph object.
        :param learner: the learner of who belong the LearnerGraph
        :param domain_graph: the DomainGraph on which the LearnerGraph is based.
        """
        self.learner = learner
        self.kc_dict = {kc: {"m_pba": .2,
                             "params": {"learn": .1, "delta": -.9, "gamma": 2.2},
                             "diagnosis": False}
                        for kc in domain_graph.knowledge_components}
        self.link_model = domain_graph.link_model

    def __str__(self):
        """
        String method for LearnerGraph
        :return: str, listing the probabilities contained in the LearnerGraph
        """
        string = f'LearnerGraph of Learner {self.learner.id}\n'
        for kc in self.kc_dict.keys():
            string += f"P({kc.id}) = {self.get_node_value_from_kc(kc)} " \
                      f"(params: {self.kc_dict[kc]['params']})\n"
        return string

    def __len__(self):
        """
        Len method for LearnerGraph
        :return: the number of KnowledgeComponents represented in the LearnerGraph
        """
        return len(self.kc_dict.keys())

    def set_learner_graph_from_learner_pool(self, learner_pool):
        """
        Instanciate the self LearnerGraph from the default LearnerGraph of a given LearnerPool
        :param learner_pool: the LearnerPool from which the LearnerGraph is
        :return: None, only changes the params of self LearnerGraph
        """
        # TODO: check if some tests are necessary (e.g. compare the kcs)
        self.kc_dict = {key: copy.deepcopy(learner_pool.default_learner.learner_graph.kc_dict[key])
                        for key in learner_pool.default_learner.learner_graph.kc_dict.keys()}

    def initialize_learner_graph_params(self, verbose=False):
        """
        Initialize the learner graph parameters with optimization for all KCs.
        :param verbose: bool, True if prints are wanted, False otherwise
        :return: None, only changes the params of self LearnerGraph
        """
        for kc in tqdm.tqdm(self.kc_dict.keys()):
            if verbose:
                print(f"Initializing parameters for:\n"
                      f"- {'P' if isinstance(kc, ProceduralKnowledgeComponent) else 'D'}"
                      f"KnowledgeComponent #{kc.id}: {kc.name}\n"
                      f"- KartableDocument #{kc.exercise_family.id}: {kc.exercise_family.name}")
            if isinstance(kc, ProceduralKnowledgeComponent):
                if kc.exercise_family:
                    self._initialize_learn_param(kc, verbose)
            else:
                if kc.exercise_family:
                    self._initialize_delta_and_gamma_params(kc, verbose)

    def _initialize_learn_param(self, kc, verbose=False):
        """
        Compute learn parameter for a given Procedural KC through bayesian optimization.
        :param kc: the KC for which the learn parameter is computed
        :param verbose: bool, True if prints are wanted, False otherwise
        :return: None, only changes learn parameter with its optimal value given the behavior wanted
        """
        from kgraph.learner_layer.evaluation import Evaluation

        def objective(x, knowledge_component):
            """
            Objective function to minimize to compute the learn param.
            :param x: learn param to optimize
            :param knowledge_component: the corresponding kc
            :return: the optimized value
            """
            self.change_learn_param(knowledge_component, x)
            ex_fam = knowledge_component.exercise_family

            possible_answers = get_representative_answers(ex_fam, 'complex')  # 2^n possible answers on ex_fam
            least_squares = []  # list of least square for every answer
            for answer in possible_answers:
                # TODO: create a method for Learner that tests an evaluation
                evaluation = Evaluation(0, ex_fam, 0,
                                        {ex_fam.exercise_list[i]: {'success': answer[i], 'length': 0}
                                         for i in range(len(ex_fam.exercise_list))})
                m_pba = self.learner.predict_evaluation(evaluation)
                least_squares.append(abs(m_pba - expected_procedural_kc_scoring(answer)) ** 2)

            return sum(least_squares)

        ex_fam = kc.exercise_family
        if not ex_fam.exercise_list:
            self.kc_dict[kc]["params"]["learn"] = 0

        else:
            # Computing the optimal value with bayesian optimization
            best = fmin(fn=lambda x: objective(x, kc),
                        space=hp.uniform('x', 0, 1), algo=tpe.suggest,
                        max_evals=500,
                        verbose=int(verbose))

            assert 0 <= best['x'] <= 1, "Learn parameter must be in [0, 1]."
            if verbose:
                print(f"learn for kc {kc.id} is {best}")
            self.kc_dict[kc]["params"]["learn"] = best['x']

    def _initialize_delta_and_gamma_params(self, kc, verbose=False):
        """
        Compute delta and gamma parameters for a given Declarative KC through bayesian optimization.
        :param kc: the KC for which the learn parameter is computed
        :param verbose: bool, True if prints are wanted, False otherwise
        :return: None, only changes delta and gamma parameter with their optimal value given the behavior wanted
        """
        from kgraph.learner_layer.evaluation import Evaluation

        def objective(x, y, knowledge_component):
            """
            Objective function to minimize to compute delta and gamma params.
            :param x: delta param to optimize
            :param y: gamma param to optimize
            :param knowledge_component: the corresponding kc
            :return: the optimized value
            """
            self.change_delta_param(knowledge_component, x)
            self.change_gamma_param(knowledge_component, y)
            ex_fam = knowledge_component.exercise_family

            possible_answers = get_representative_answers(ex_fam, 'simple')  # 2^n possible answers on ex_fam
            least_squares = []  # list of least square for every answer
            for answer in possible_answers:
                evaluation = Evaluation(0, ex_fam, 0,
                                        {ex_fam.exercise_list[i]: {'success': answer[i], 'length': 0}
                                         for i in range(len(ex_fam.exercise_list))})
                m_pba = self.learner.predict_evaluation(evaluation)
                least_squares.append(abs(m_pba - expected_declarative_kc_scoring(answer)) ** 2)

            return sum(least_squares)

        ex_fam = kc.exercise_family
        if not ex_fam.exercise_list:
            self.kc_dict[kc]["params"]["delta"], self.kc_dict[kc]["params"]["gamma"] = 0, 0

        else:
            # Computing the optimal value with bayesian optimization
            best = fmin(fn=lambda x: objective(x[0], x[1], kc),
                        space=[hp.uniform('x', -2, 0), hp.uniform('y', 0, 5)], algo=tpe.suggest,
                        max_evals=500,
                        verbose=int(verbose))

            if verbose:
                print(f"[delta, gamma] for kc {kc.id} is {best}")
            assert 0 >= best['x'], "Delta parameter must be negative"
            assert 0 <= best['y'], "Gamma parameter must be positive"
            self.kc_dict[kc]["params"]["delta"], self.kc_dict[kc]["params"]["gamma"] = best['x'], best['y']

    def change_learn_param(self, kc, learn):
        """
        Changes the learn parameter of a given kc to a given value
        :param kc: the KnowledgeComponent
        :param learn: the wished learn value
        :return: None, only changes the learn param of the given kc
        """
        self.kc_dict[kc]['params']['learn'] = learn

    def change_delta_param(self, kc, delta):
        """
        Changes the delta parameter of a given kc to a given value
        :param kc: the KnowledgeComponent
        :param delta: the wished delta value
        :return: None, only changes the delta param of the given kc
        """
        self.kc_dict[kc]['params']['delta'] = delta

    def change_gamma_param(self, kc, gamma):
        """
        Changes the gamma parameter of a given kc to a given value
        :param kc: the KnowledgeComponent
        :param gamma: the wished gamma value
        :return: None, only changes the gamma param of the given kc
        """
        self.kc_dict[kc]['params']['gamma'] = gamma

    def change_kc_params(self, kc, **kwargs):
        assert all(key in ('learn', 'delta', 'gamma') for key in kwargs.keys())
        if 'learn' in kwargs.keys():
            self.change_learn_param(kc, kwargs['learn'])
        if 'delta' in kwargs.keys():
            self.change_delta_param(kc, kwargs['delta'])
        if 'gamma' in kwargs.keys():
            self.change_gamma_param(kc, kwargs['gamma'])

    def get_node_value_from_kc(self, kc):
        return self.kc_dict[kc]['m_pba']

    def get_params_from_kc(self, kc):
        return self.kc_dict[kc]['params']['learn'], \
               self.kc_dict[kc]['params']['slip'], \
               self.kc_dict[kc]['params']['guess']

    def get_mastering_probability(self, kc: KnowledgeComponent):
        assert kc in self.kc_dict.keys(), f"KC #{kc.id} not in the LearnerGraph kc_dict."
        return self.kc_dict[kc]["m_pba"]

    def set_mastering_probability(self, kc: KnowledgeComponent, m_pba):
        self.kc_dict[kc]["m_pba"] = m_pba

    def update_mastering_probability(self, kc, answer, additional_params):
        """
        Method to update probability to master self from the result of an exercise about self.
        :param answer: dict that contains the result and content of an answer to an exercise
        :param params: params of the exercise -- must contain learn, guess and slip params (ref. kgraph content)
        :return: update the value of the mastering probability of self
        """
        m_pba = self.get_mastering_probability(kc)
        params = {**self.kc_dict[kc]['params'], **additional_params}
        self.set_mastering_probability(kc, kc.update_mastering_probability(m_pba, answer, params))

    def diffuse_to_children(self, kc):
        children = self.link_model.get_children(kc)
        if children:
            m_pba = self.get_mastering_probability(kc)
            for child in children:
                if m_pba < .5:
                    child_m_pba = min(self.get_mastering_probability(child), .5 * m_pba)
                else:
                    child_m_pba = self.get_mastering_probability(child)
                self.set_mastering_probability(child, child_m_pba)
                self.diffuse_to_children(child)

    def diffuse_to_parents(self, kc):
        parents = self.link_model.get_parents(kc)
        if parents:
            for parent in parents:
                trigger = .5
                parent_children = self.link_model.get_children(parent)
                max_m_pba = max([self.get_mastering_probability(child) for child in parent_children])
                if max_m_pba > trigger:
                    parent_m_pba = max_m_pba + .5 * (1 - max_m_pba)
                else:
                    parent_m_pba = self.get_mastering_probability(parent)
                self.set_mastering_probability(parent, parent_m_pba)
                self.diffuse_to_parents(parent)

    def display(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        # Graph initialization
        G = nx.DiGraph()

        # Nodes definition
        nodes = [kc for kc in self.kc_dict]
        G.add_nodes_from(nodes)
        labels = {n: n.id for n in nodes}
        # Edges definition
        edges = self.link_model.get_all_links()
        G.add_edges_from(edges)

        # For now, random layout
        # TODO : intelligent layout
        pos = nx.random_layout(G)

        print("INITIAL SKILL GRAPH")
        node_sizes = [1000 * self.get_mastering_probability(node) ** 4 for node in nodes]

        nx.draw_networkx(G, pos=pos, arrows=True, with_labels=True, **{'node_size': node_sizes, 'labels': labels})
        plt.show()

    def bayesian_process(self, evaluation, model=1):
        evaluated_knowledge_component = evaluation.exercise_family.kc
        self.learner.process_evaluation(evaluation)
        if model == 1:
            self._bayesian_diffuse_to_parents(evaluated_knowledge_component, dynamic=False)
            roots = self.link_model.get_roots()
            for root in roots:
                self._bayesian_diffuse_to_children(root, dynamic=False)
        if model == 2:
            self._bayesian_diffuse_to_parents(evaluated_knowledge_component, dynamic=True)
            roots = self.link_model.get_roots()
            for root in roots:
                self._bayesian_diffuse_to_children(root, dynamic=False)
        if model == 3:
            self._bayesian_diffuse_to_parents(evaluated_knowledge_component, dynamic=True)
            roots = self.link_model.get_roots()
            for root in roots:
                self._bayesian_diffuse_to_children(root, dynamic=True)
        if model == 4:
            self._bayesian_diffuse_to_parents(evaluated_knowledge_component, dynamic=False)
            self._bayesian_diffuse_to_children(evaluated_knowledge_component, dynamic=False)
        if model == 5:
            self._bayesian_diffuse_to_parents(evaluated_knowledge_component, dynamic=True)
            self._bayesian_diffuse_to_children(evaluated_knowledge_component, dynamic=False)
        if model == 6:
            self._bayesian_diffuse_to_parents(evaluated_knowledge_component, dynamic=True)
            self._bayesian_diffuse_to_children(evaluated_knowledge_component, dynamic=True)

    def _bayesian_diffuse_to_parents(self, kc, dynamic=False):
        parents = self.link_model.get_parents(kc)
        if parents:
            for parent in parents:
                parent_children = self.link_model.get_children(parent)
                parent_children_truthtable = truthtable(len(parent_children))
                m_pba_truthtable = [
                    np.prod([self.get_mastering_probability(parent_children[i])
                             if parent_children_truthtable[j][i] else 1 - self.get_mastering_probability(
                        parent_children[i])
                             for i in range(len(parent_children))]) for j in range(len(parent_children_truthtable))]
                cond_pbas = self.link_model.get_links(parent)['from_children'].probability_vector
                if dynamic:
                    alpha = 0.8
                    prior_parent_pba = self.get_mastering_probability(parent)
                    parent_m_pba = alpha * (
                                prior_parent_pba + (1 - prior_parent_pba) * np.dot(m_pba_truthtable, cond_pbas)) \
                                   + (1 - alpha) * np.dot(m_pba_truthtable, cond_pbas)
                else:
                    parent_m_pba = np.dot(m_pba_truthtable, cond_pbas)
                self.set_mastering_probability(parent, parent_m_pba)
                self._bayesian_diffuse_to_parents(parent, dynamic)

    def _bayesian_diffuse_to_children(self, kc, dynamic=False):
        children = self.link_model.get_children(kc)
        if children:
            for child in children:
                child_parents = self.link_model.get_parents(child)
                child_parents_truthtable = truthtable(len(child_parents))
                m_pba_truthtable = [
                    np.prod([self.get_mastering_probability(child_parents[i])
                             if child_parents_truthtable[j][i] else 1 - self.get_mastering_probability(child_parents[i])
                             for i in range(len(child_parents))]) for j in range(len(child_parents_truthtable))]
                cond_pbas = self.link_model.get_links(child)['from_parents'].probability_vector
                if dynamic:
                    alpha = 0.8
                    prior_child_pba = self.get_mastering_probability(child)
                    child_m_pba = alpha * (
                                prior_child_pba + (1 - prior_child_pba) * np.dot(m_pba_truthtable, cond_pbas)) \
                                   + (1 - alpha) * np.dot(m_pba_truthtable, cond_pbas)
                else:
                    child_m_pba = np.dot(m_pba_truthtable, cond_pbas)
                self.set_mastering_probability(child, child_m_pba)
                self._bayesian_diffuse_to_children(child)
