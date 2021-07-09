import numpy as np
import matplotlib.pyplot as plt
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb


class Learner(object):

    def __init__(self, learner_id, learner_pool=None):
        """
        Initialization of Learner object.
        :param learner_id: id of the learner
        :param domain_graph: domain graph of the domain that the Learner studies
        """
        self.id = learner_id  # TODO change self.learner_id in self.id
        self.learner_pool = learner_pool
        if self.learner_pool:
            self.learner_pool.add_learner(self)
            knowledge_components = self.learner_pool.get_knowledge_components()
            self.parameters = {kc: {'learn': self.learner_pool.get_learn(kc),
                                    'slip': self.learner_pool.get_slip(),
                                    'guess': self.learner_pool.get_guess()} for kc in knowledge_components}
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

    def predict_answers(self, evaluations, verbose=False):
        bn = self.learner_pool.dbn_inference()
        """
        # HARD EVIDENCE ON EVAL KC
        evidences = {f"eval({evaluations[i][0].name}){i+1}": int(evaluations[i][1]) for i in range(len(evaluations))}

        """
        # SOFT EVIDENCE ON KC
        evidences = {
            f"({evaluations[i][0].name}){i + 1}": [self.learner_pool.get_guess(), 1 - self.learner_pool.get_guess()] if
            evaluations[i][1] else [
                1 - self.learner_pool.get_slip(), self.learner_pool.get_slip()] for i in
            range(len(evaluations))}  # soft evidences directly on KC nodes
        #"""
        unrolled_bn = gdyn.unroll2TBN(bn, len(evaluations) + 1)
        ie = gum.LazyPropagation(unrolled_bn)
        ie.setEvidence(evidences)
        ie.makeInference()
        if verbose:
            if len(evidences) < 5:
                gnb.showInference(unrolled_bn, evs=evidences)
            gdyn.plotFollowUnrolled(["(A)", "(B)"], unrolled_bn, T=len(evaluations) + 1, evs=evidences)
        return [ie.posterior(unrolled_bn.idFromName(
            f"eval({evaluations[i][0].name}){i + 1}"))[1] for i in range(len(evaluations))]
