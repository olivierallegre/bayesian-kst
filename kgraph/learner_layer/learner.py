import numpy as np
import matplotlib.pyplot as plt
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import sklearn.metrics as sk_metrics
from kgraph.learner_layer.inference_model import NoisyANDInferenceModel, NoisyORInferenceModel

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
            self.mastering_probabilities = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}

    def change_learner_pool(self, new_learner_pool):
        self.__init__(self.id, new_learner_pool)

    @staticmethod
    def get_guess_from_exercise(exercise):
        guess = exercise.get_guess()
        # TODO: adapt the guess parameter to the self learner
        return guess

    @staticmethod
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

    def predict_sequence(self, learner_traces, inference_model_type, params):
        n_eval = len(learner_traces)
        if inference_model_type == 'NoisyAND':
            inference_model = NoisyANDInferenceModel(self.learner_pool, {'c': params['c'], 's': params['s']})
        elif inference_model_type == 'NoisyOR':
            inference_model = NoisyORInferenceModel(self.learner_pool, params['c'])
        else:
            return Exception('This type of inference model is not handled.')
        correct_predictions, exercises = [], []
        for trace in learner_traces:
            exercise = trace.get_exercise()
            if exercise not in exercises:
                exercises.append(exercise)

        for i in range(n_eval):
            knowledge_states = inference_model.predict_learner_knowledge_states_from_learner_traces(learner_traces[:i])
            slip = self.learner_pool.slips[learner_traces[i].get_exercise()]
            guess = self.learner_pool.guesses[learner_traces[i].get_exercise()]
            m_pba = knowledge_states[f'{learner_traces[i].get_kc().id}']
            correct_predictions.append(m_pba*(1-slip) + (1-m_pba)*guess)  # pba to answer correctly
        return correct_predictions
