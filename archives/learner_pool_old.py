import numpy as np
import dill
from kgraph.expert_layer.domain_graph import DomainGraph
from kgraph.learner_layer.learner import Learner
from archives.learner_graph import LearnerGraph
from kgraph.helpers.random_generation import get_random_integers_list_summing_to_given_integer


class LearnerPool(object):
    learner: Learner

    def __init__(self, domain_graph: DomainGraph):
        """
        Initialization of the LearnerPool class
        :param learners : list of Learner objects that corresponds to the learners belonging to the pool
        """
        self.learners = []
        self.domain_graph = domain_graph
        self.default_learner = Learner(0, self)

    def __str__(self):
        string = f'The LearnerPool contains {len(self.learners)} learners' \
                 f' on {len(self.domain_graph.knowledge_components)} KCs DomainGraph.'
        return string

    def add_learner(self, learner):
        if learner not in self.learners and learner.id != 0:
            self.learners.append(learner)

    def get_knowledge_components(self):
        return self.domain_graph.knowledge_components

    def get_test_learner(self):
        return Learner(-1, self)

    def get_new_learner_graph(self, learner):
        assert learner in self.learners or learner.id == 0, "Given learner not in learner_pool."
        learner.set_learner_graph(LearnerGraph(learner, self.domain_graph))
        if learner.id == 0:
            learner.learner_graph.initialize_learner_graph_params()
        else:
            learner.learner_graph.set_learner_graph_from_learner_pool(self)

    def get_learner_ids(self):
        return [learner.id for learner in self.learners]

    def get_learner_from_id(self, learner_id):
        """
        Get a Learner object that corresponds to learner_id id.
        :param learner_id: id of the searched learner
        :return: Learner object that has learner_id as id
        """
        return [learner for learner in self.learners if learner.id == learner_id]

    def compute_mean_mastering_from_kc_id(self, kc_id):
        """
        Compute the mean over learner of LearnerPool of the probability of mastering of the Knowledge Component kc_id.
        :param kc_id: the id of the KnowledgeComponent
        :return: the mean of the probabilities of mastering of learners from LearnerPool
        """
        return np.mean([learner.learner_mastering["prob"][kc_id] for learner in self.learners])

    def save_object(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            dill.dump(self, output, dill.HIGHEST_PROTOCOL)

    def get_ex_fam_ids(self):
        return self.domain_graph.get_ex_fam_ids()

    def get_random_learner(self):
        from random import seed
        from random import randint
        # seed random number generator
        seed(1)
        return self.learners[randint(0, len(self.learners))]

    def setup_random_learners(self, n_learners):
        import random
        for i in range(1, n_learners + 1):
            learner = Learner(i, self)
            if learner not in self.learners:
                self.learners.append(learner)
            p_a = random.uniform(0, 1)
            learner.set_mastering_probability(self.domain_graph.get_kc_by_name("A"), p_a)
            p_b = (1 - p_a) * .15 + p_a * .1
            learner.set_mastering_probability(self.domain_graph.get_kc_by_name("B"), p_b)

    def simulate_evaluations_from_learners(self, n_evaluations):
        import random
        # generating the evaluation repartition between users and kcs
        eval_repartition = []
        n_learners = len(self.learners)
        n_kc = len(self.domain_graph.knowledge_components)
        n_evaluation_of_learner = get_random_integers_list_summing_to_given_integer(n_learners, n_evaluations)

        for i in range(n_learners):
            learner_eval_repartition = get_random_integers_list_summing_to_given_integer(
                n_kc, n_evaluation_of_learner[i])
            eval_repartition.append(learner_eval_repartition)
        simulated_evals = [[] for _ in range(n_learners)]
        # generating the evals one by one
        for i in range(n_learners):
            for j in range(n_kc):
                for k in range(eval_repartition[i][j]):
                    simulated_evals[i].append(self.learners[i].simulate_evaluation(
                        self.domain_graph.knowledge_components[j].exercise_family))
        shuffled_sim_evals = [[simulated_evals[i][j] for j in random.sample([
            k for k in np.arange(len(simulated_evals[i]))], len(simulated_evals[i]))]
                        for i in range(n_learners)]
        return shuffled_sim_evals



    def print_default_learner_graph(self):
        print(self.default_learner.learner_graph)
