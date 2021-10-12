import numpy as np
import matplotlib.pyplot as plt
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from kgraph.learner_layer.evaluation import Evaluation
import sklearn.metrics as sk_metrics


class Learner(object):

    def __init__(self, learner_id, learner_pool=None):
        """
        Initialization of Learner object.
        :param learner_id: id of the learner
        :param domain_graph: domain graph of the domain that the Learner studies
        """
        self.id = learner_id
        self.learner_pool = learner_pool
        if self.learner_pool:
            self.learner_pool.add_learner(self)
            knowledge_components = self.learner_pool.get_knowledge_components()
            self.parameters = {kc: {'learn': self.learner_pool.get_learn(kc),
                                    'slip': self.learner_pool.get_slip(kc),
                                    'guess': self.learner_pool.get_guess(kc)} for kc in knowledge_components}
            self.mastering_probabilities = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}

    def change_learner_pool(self, new_learner_pool):
        self.__init__(self.id, new_learner_pool)

    @staticmethod
    def get_guess_from_exercise(exercise):
        guess = exercise.get_guess()
        # TODO: adapt the guess parameter to the self learner
        return guess

    @staticmethod
    def get_learn_parameter(self, kc):
        return self.parameters[kc]['learn']

    def get_slip_parameter(self, kc):
        return self.parameters[kc]['slip']

    def get_guess_parameter(self, kc):
        return self.parameters[kc]['guess']

    def set_mastering_probability(self, kc, prior):
        assert kc in self.mastering_probabilities.keys()
        self.mastering_probabilities[kc] = prior

    def get_mastering_probability(self, kc):
        return self.mastering_probabilities[kc]

    def set_priors(self, priors):
        for kc in self.learner_pool.get_knowledge_components():
            self.set_mastering_probability(kc, priors[kc])

    def get_priors(self):
        return {kc.name: self.get_mastering_probability(kc) for kc in self.learner_pool.get_knowledge_components()}

    def predict_next_step(self, priors, evaluation=None, pred_mode='one_kc'):
        evaluated_kc = evaluation[0] if evaluation else evaluation
        bn = self.learner_pool.two_step_bn(priors, evaluated_kc, pred_mode)
        if evaluated_kc:
            evidence = {
                f"({evaluation[0].name})t": [self.learner_pool.get_guess(evaluated_kc),
                                             1 - self.learner_pool.get_guess(evaluated_kc)] if evaluation[1]
                else [1 - self.learner_pool.get_slip(evaluated_kc), self.learner_pool.get_slip(evaluated_kc)]}
        else:
            evidence = {}
        ie = gum.LazyPropagation(bn)
        ie.setEvidence(evidence)
        ie.makeInference()
        knowledge_components = self.learner_pool.get_knowledge_components()
        next_state_prediction = {
            **{f"{kc.name}": ie.posterior(bn.idFromName(f"({kc.name})t"))[1] for kc in knowledge_components},
            **{f"eval({kc.name})": ie.posterior(bn.idFromName(f"eval({kc.name})t"))[1] for kc in
               knowledge_components}}
        return next_state_prediction

    def predict_sequence(self, evaluations, floor_idx=0, verbose=False):
        n_eval = len(evaluations)
        knowledge_state = {kc.name: self.learner_pool.get_prior(kc) for kc in self.learner_pool.get_knowledge_components()}
        predicted_values = {}
        for i in range(n_eval-1):
            knowledge_state = self.predict_next_step(knowledge_state, evaluations[i])
            if i > floor_idx:
                for kc in self.learner_pool.get_knowledge_components():
                    predicted_values[f"({kc.name}){i}"] = knowledge_state[f"{kc.name}"]
                    predicted_values[f"eval({kc.name}){i}"] = knowledge_state[f"eval({kc.name})"]
        return predicted_values
