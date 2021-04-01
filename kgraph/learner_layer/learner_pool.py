import numpy as np
import dill
from kgraph.expert_layer.domain_graph import DomainGraph
from kgraph.learner_layer.learner import Learner
from kgraph.learner_layer.learner_graph import LearnerGraph


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