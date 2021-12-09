import numpy as np
import dill
from kgraph.expert_layer.domain import Domain
from kgraph.learner_layer.learner import Learner
import pyAgrum as gum
import itertools
from lmfit import Minimizer, Parameters, fit_report
import sklearn.metrics as sk_metrics


class LearnerPool(object):
    learner: Learner

    def __init__(self, domain: Domain, link_strengths, params=None, desc='unspecified'):
        """
        The LearnerPool class is supposed to represent groups of Learners that shares similar characteristics. They
        belong on a given domain on which they behave in the same way. We declare a default_learner that will emphasize
        the behavior of a random learner of this pool.
        :param domain : the domain on which the learner of the LearnerPool study.
        """
        self.desc = desc
        self.learners = []
        self.domain = domain
        self.knowledge_components = self.domain.get_knowledge_components()
        if params is None:
            self.priors = {kc: .2 for kc in self.knowledge_components}
            self.learns = {kc: .1 for kc in self.knowledge_components}
            self.forgets = {kc: .05 for kc in self.knowledge_components}

            self.slips = {x: 0.1 for kc in self.knowledge_components for x in kc.get_exercises()}
            self.guesses = {x: 0.1 for kc in self.knowledge_components for x in kc.get_exercises()}
        else:
            self.priors = {kc: params.loc[f'{kc.id}', 'prior', 'default'].value for kc in self.knowledge_components}
            self.learns = {kc: params.loc[f'{kc.id}', 'learns', f'{kc.id}'].value for kc in self.knowledge_components}
            self.forgets = {kc: params.loc[f'{kc.id}', 'forgets', f'{kc.id}'].value for kc in self.knowledge_components}
            self.slips = {ex: params.loc[f'{kc.id}', 'slips', f'{ex.id}'].value for kc in self.knowledge_components for ex in kc.get_exercises()}
            self.guesses = {ex: params.loc[f'{kc.id}', 'guesses', f'{ex.id}'].value for kc in self.knowledge_components for ex in kc.get_exercises()}

        self.link_strengths = link_strengths

    def __str__(self):
        string = f'The LearnerPool {self.desc} contains {len(self.learners)} learners' \
                 f' on {len(self.domain.get_knowledge_components())} KCs DomainGraph.'
        return string

    def add_learner(self, learner):
        """
        Add a learner in the LearnerPool.
        :param learner: Learner object, the learner to be added
        """
        if isinstance(learner, list):
            for elt in learner:
                self.add_learner(elt)
        if learner not in self.learners and learner.id != 0:
            self.learners.append(learner)
            learner.change_learner_pool(self)

    def get_exercise_from_id(self, exercise_id):
        exercise_list = [exercise for kc in self.knowledge_components for exercise in kc.get_exercises()]
        return [exercise for exercise in exercise_list if exercise.id == exercise_id][0]

    def set_link_strength(self, source_kc, target_kc, strength):
        """
        Set the strength of the linked between source_kc and target_kc to given value.
        :param source_kc: the source kc of the link
        :param target_kc: the target kc of the link
        :param strength: the wished strength of the link
        """
        assert strength in ['strong', 'weak']
        self.link_strengths[source_kc][target_kc] = strength

    def get_link_strength(self, source_kc, target_kc):
        """
        Returns the strength of a prerequisite link given its source and target.
        :param source_kc: KnowledgeComponent object, the source of the studied prerequisite link
        :param target_kc: KnowledgeComponent object, the target of the studied prerequisite link
        :return: str, the strength of the prerequisite link
        """
        if source_kc in self.link_strengths.keys():
            if target_kc in self.link_strengths[source_kc].keys():
                return self.link_strengths[source_kc][target_kc]
        return 'not existing'

    def get_link_strengths(self):
        """
        Returns the self link strengths argument.
        :return: dict, the strength of every prerequisite link of the learner pool.
        """
        return self.link_strengths

    def get_conditional_probability(self, consequence, condition):
        assert isinstance(condition, dict), "The condition must be a dict of the form {kc: value}"
        assert len(list(condition.keys())) == 1, "Only one condition for now"

        source_kc = list(condition.keys())[0]
        link_strength = self.get_link_strength(source_kc, consequence)
        if condition[source_kc] == 1:
            if source_kc in self.get_learner_pool_kc_parents(consequence):  # the condition is a parent
                if link_strength == 'strong':  # the condition is a strong parent
                    return .4
                elif link_strength == 'weak':
                    return .25
                else:
                    return ValueError("link_strength shouldn't be 'not existing'.")
            elif source_kc in self.get_learner_pool_kc_children(consequence):  # the condition is a child
                if link_strength == 'strong':  # the condition is a strong parent
                    return .9

                elif link_strength == 'weak':
                    return .7
                else:
                    return ValueError("link_strength shouldn't be 'not existing'.")
            else:
                return ValueError("condition is not in kc's parents nor children.")
        else:
            assert source_kc in self.get_learner_pool_kc_parents(consequence), \
                "we don't infer on not mastering children" # the condition is a child
            if link_strength == 'strong':  # the condition is a strong parent
                return .05
            elif link_strength == 'weak':
                return .1
            else:
                return ValueError("link_strength shouldn't be 'not existing'.")

    def get_knowledge_components(self):
        """
        Return all KCs of the domain associated to the LearnerPool.
        :return: the list of Domain's KCs.
        """
        return self.domain.get_knowledge_components()

    def get_learner_ids(self):
        """
        Return the ids of the learners that belong to LearnerPool.
        :return: the list of learners' ids.
        """
        return [learner.id for learner in self.learners]

    def get_learner_from_id(self, learner_id):
        """
        Get a Learner object that corresponds to learner_id id.
        :param learner_id: id of the searched learner
        :return: Learner object that has learner_id as id
        """
        return [learner for learner in self.learners if learner.id == learner_id]

    def get_random_learner(self):
        """
        Return a random learner among the learners belonging to the LearnerPool.
        :return: Learner object, a random learner
        """
        from random import seed
        from random import randint
        # seed random number generator
        seed(1)
        return self.learners[randint(0, len(self.learners))]

    def get_learner_pool_kc_parents(self, kc):
        return [parent for parent in self.link_strengths[kc].keys() if self.get_link_strength(parent, kc)!='not existing']

    def get_learner_pool_kc_children(self, kc):
        return [child for child in self.link_strengths.keys() if kc in list(self.link_strengths[child].keys())]

    def set_learn(self, kc, val):
        self.learns[kc] = val

    def set_prior(self, kc, val):
        self.priors[kc] = val

    def set_slip(self, exercise, val):
        from kgraph.resources_layer.exercise import Exercise
        assert isinstance(exercise, Exercise), "Exercise expected"
        self.slips[exercise] = val

    def set_guess(self, exercise, val):
        from kgraph.resources_layer.exercise import Exercise
        assert isinstance(exercise, Exercise), "Exercise expected"
        self.guesses[exercise] = val

    def set_forget(self, kc, val):
        self.forgets[kc] = val

    def get_learn(self, kc):
        return self.learns[kc]

    def get_prior(self, kc):
        return self.priors[kc]

    def get_slip(self, exercise):
        return self.slips[exercise]

    def get_guess(self, exercise):
        return self.guesses[exercise]

    def get_forget(self, kc):
        return self.forgets[kc]

    def is_kc_learnable(self, kc, evaluated_kc, learn_prop):
        if learn_prop == 'all':
            return True
        if kc is evaluated_kc:
            return True
        return False

    def get_recursive_parents(self, root_kc):
        parents = []

        def _get_parents_leaf_nodes(kc, cond_pba):
            if kc is not None:
                if len(self.get_learner_pool_kc_parents(kc)) == 0:
                    parents.append([kc, cond_pba])
                for parent in self.get_learner_pool_kc_parents(kc):
                    _get_parents_leaf_nodes(parent, cond_pba * self.get_conditional_probability(parent, {kc: True}))

        _get_parents_leaf_nodes(root_kc, 1.)
        return parents

    def get_recursive_children(self, root_kc):
        children = []

        def _get_children_leaf_nodes(kc, cond_pba):
            if kc is not None:
                if len(self.get_learner_pool_kc_children(kc)) == 0:
                    children.append([kc, cond_pba])
                for child in self.get_learner_pool_kc_children(kc):
                    _get_children_leaf_nodes(child, cond_pba * self.get_conditional_probability(kc, {child: True}))

        _get_children_leaf_nodes(root_kc, 1.)
        return children

    def get_kc_parents(self, kc):
        if kc not in self.link_strengths.keys():
            return []
        else:
            return [parent for parent in self.link_strengths[kc].keys()]

    def get_optimized_parameters(self, learner_traces, inference_model_type):
        if inference_model_type == 'NoisyAND':
            cs_params = Parameters()

            for kc in self.link_strengths.keys():
                for parent in self.link_strengths[kc]:
                    cs_params.add(f'c_{parent.id}_{kc.id}', value=1, vary=True, min=0.5, max=1, brute_step=10e-2)
                    cs_params.add(f's_{parent.id}_{kc.id}', value=0, vary=True, min=0, max=.5, brute_step=10e-2)

            def f(p, **kwargs):
                par = p.valuesdict()
                learner_traces = kwargs["learner_traces"]
                c, s = {}, {}
                for kc in self.link_strengths.keys():
                    for parent in self.link_strengths[kc].keys():
                        c[parent], s[parent] = {}, {}
                for kc in self.link_strengths.keys():
                    for parent in self.link_strengths[kc].keys():
                        c[parent][kc] = par[f'c_{parent.id}_{kc.id}']
                        s[parent][kc] = par[f's_{parent.id}_{kc.id}']
                        print('c: ', par[f'c_{parent.id}_{kc.id}'])
                        print('s: ', par[f's_{parent.id}_{kc.id}'])

                exp, pred = [], []
                for learner in learner_traces.keys():
                    exp = np.concatenate((exp, [
                        int(trace.get_success()) for trace in learner_traces[learner]
                    ]))
                    pred = np.concatenate((pred,
                                           learner.predict_sequence([trace for trace in learner_traces[learner]],
                                                                    inference_model_type, {'c': c, 's': s})
                                          ))
                cohen_kappa = max([
                    sk_metrics.cohen_kappa_score(np.array(exp),
                                                 [1 if pred[i] > j else 0 for i in range(len(pred))])
                    for j in np.linspace(0, 1, 100)
                ])

                print(sk_metrics.roc_auc_score(exp, pred), cohen_kappa)
                return 1 - cohen_kappa

            kws = {'learner_traces': learner_traces}
            fitter = Minimizer(f, cs_params, fcn_kws=kws)
            result = fitter.minimize(method='brute')

        elif inference_model_type == 'NoisyOR':
            c_params = Parameters()

            for kc in self.link_strengths.keys():
                for parent in self.link_strengths[kc]:
                    c_params.add(f'c_{parent.id}_{kc.id}', value=1, vary=True, min=.5, max=1, brute_step=10e-2)

            def f(p, **kwargs):
                par = p.valuesdict()
                learner_traces = kwargs["learner_traces"]
                c = {}
                for kc in self.link_strengths.keys():
                    for parent in self.link_strengths[kc].keys():
                        c[parent] = {}
                for kc in self.link_strengths.keys():
                    for parent in self.link_strengths[kc].keys():
                        c[parent][kc] = par[f'c_{parent.id}_{kc.id}']
                        print('c: ', par[f'c_{parent.id}_{kc.id}'])
                exp, pred = [], []
                for learner in learner_traces.keys():
                    exp = np.concatenate((exp, [
                        int(trace.get_success()) for trace in learner_traces[learner]
                    ]))
                    pred = np.concatenate((pred,
                                           learner.predict_sequence([trace for trace in learner_traces[learner]],
                                                                    inference_model_type, {'c': c})
                                           ))
                cohen_kappa = max([
                    sk_metrics.cohen_kappa_score(np.array(exp),
                                                 [1 if pred[i] > j else 0 for i in range(len(pred))])
                    for j in np.linspace(0, 1, 100)
                ])

                print('auc',sk_metrics.roc_auc_score(exp, pred))
                print('cohen kappa', cohen_kappa)
                return 1 - cohen_kappa

            kws = {'learner_traces': learner_traces}
            fitter = Minimizer(f, c_params, fcn_kws=kws)
            result = fitter.minimize(method='brute')
        else:
            return Exception('Inference model type not handled.')
        return result
