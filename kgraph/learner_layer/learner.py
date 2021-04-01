import numpy as np
import matplotlib.pyplot as plt
from kgraph.learner_layer.learner_graph import LearnerGraph


class Learner(object):
    learner_graph: LearnerGraph

    def __init__(self, learner_id, learner_pool):
        """
        Initialization of Learner object.
        :param learner_id: id of the learner
        :param domain_graph: domain graph of the domain that the Learner studies
        """
        self.id = learner_id  # TODO change self.learner_id in self.id
        self.learner_pool = learner_pool
        self.learner_pool.add_learner(self)
        self.learner_pool.get_new_learner_graph(self)

    def __str__(self):
        string = f"Learner #{self.id} belonging to LearnerPool {self.learner_pool}\n" \
                 f"LearnerGraph:" + self.learner_graph.__str__()
        return string

    def set_learner_graph(self, learner_graph):
        self.learner_graph = learner_graph

    def print_learner_graph(self):
        """
        Method to print the mastering probability of every knowledge component of the self learner graph.
        :return: print of the probabilities
        """
        print(self.learner_graph)

    @staticmethod
    def get_guess_from_exercise(exercise):
        guess = exercise.get_guess()
        # TODO: adapt the guess parameter to the self learner
        return guess

    @staticmethod
    def get_slip_from_exercise(exercise):
        slip = exercise.get_slip()
        # TODO: adapt the slip parameter to the self learner
        return slip

    def get_learn_parameter(self, kc):
        return self.learner_graph.kc_dict[kc]['params']['learn']

    def predict_evaluation(self, evaluation):
        kc = evaluation.exercise_family.kc
        answers = evaluation.answers
        # LOCAL DIAGNOSIS
        prior_pba = self.get_mastering_probability(kc)
        self._compute_diagnosis(kc, answers)
        updated_pba = self.get_mastering_probability(kc)
        self.set_mastering_probability(kc, prior_pba)
        return updated_pba

    # def has_done_a_new_evaluation(self, evaluation):
    #     kc = evaluation.exercise_family.kc
    #     answers = evaluation.answers
    #     # LOCAL DIAGNOSIS
    #     self._compute_diagnosis(kc, answers)
    #     self.learner_graph.kc_dict[kc]['diagnosis'] = True  # we store the state of diagnosis
    #     # GLOBAL DIAGNOSIS -- former propagation
    #     self._diffuse_from_kc(kc)

    def process_evaluation(self, evaluation, local_diagnosis=True, global_diagnosis=False):
        from kgraph.learner_layer.evaluation import Evaluation

        if isinstance(evaluation, (list, np.ndarray)):
            for element in evaluation:
                self.process_evaluation(element, local_diagnosis, global_diagnosis)
        else:
            assert isinstance(evaluation, Evaluation), f"Evaluation {evaluation} must be an Evaluation object."
            kc = evaluation.exercise_family.kc
            answers = evaluation.answers
            # LOCAL DIAGNOSIS
            if local_diagnosis:
                self._compute_diagnosis(kc, answers)
                self.learner_graph.kc_dict[kc]['diagnosis'] = True  # we store the state of diagnosis
            # GLOBAL DIAGNOSIS -- former propagation
            if global_diagnosis:
                self._diffuse_from_kc(kc)

    def _compute_diagnosis(self, kc, answers):
        """
        Method declaring a new evaluation done by self and modifing the probability of mastering the associated
        knowledge component.
        :param evaluation: Evaluation, evaluation done by self
        :return: modify the values of LearnerGraph
        """
        # TODO: shuffle the exercises
        for exercise in answers.keys():
            add_params = {'guess': self.get_guess_from_exercise(exercise),
                          'slip': self.get_slip_from_exercise(exercise)}
            self.learner_graph.update_mastering_probability(kc, answers[exercise], add_params)

    def _diffuse_from_kc(self, kc):
        """
        Method declaring a new evaluation done by self and modifing the probability of mastering the associated
        knowledge component.
        :param evaluation: Evaluation, evaluation done by self
        :return: modify the values of LearnerGraph
        """
        self.learner_graph.diffuse_to_children(kc)
        self.learner_graph.diffuse_to_parents(kc)

    def get_mastering_probability(self, kc):
        return self.learner_graph.kc_dict[kc]['m_pba']

    def set_mastering_probability(self, kc, m_pba):
        self.learner_graph.kc_dict[kc]['m_pba'] = m_pba

    def test_given_eval(self, evaluation):
        kc = evaluation.get_kc()
        initial_m_pba = self.get_mastering_probability(kc)
        self.has_done_a_new_evaluation(evaluation)
        m_pba = self.get_mastering_probability(kc)
        self.set_mastering_probability(kc, initial_m_pba)
        return m_pba

