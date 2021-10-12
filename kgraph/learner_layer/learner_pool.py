import numpy as np
import dill
from kgraph.expert_layer.domain import Domain
from kgraph.learner_layer.learner import Learner
import pyAgrum as gum
import itertools



class LearnerPool(object):
    learner: Learner

    def __init__(self, domain: Domain, links_strengths, desc='unspecified'):
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
        self.priors = {kc: .2 for kc in self.knowledge_components}
        self.slips = {kc: .1 for kc in self.knowledge_components}
        self.guesses = {kc: .1 for kc in self.knowledge_components}
        self.learns = {kc: .1 for kc in self.knowledge_components}
        self.forgets = {kc: .05 for kc in self.knowledge_components}
        self.links_strengths = links_strengths

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

    def set_link_strength(self, source_kc, target_kc, strength):
        """
        Set the strength of the linked between source_kc and target_kc to given value.
        :param source_kc: the source kc of the link
        :param target_kc: the target kc of the link
        :param strength: the wished strength of the link
        """
        assert strength in ['strong', 'weak', 'not existing']
        self.links_strengths[source_kc][target_kc] = strength

    def get_link_strength(self, source_kc, target_kc):
        if source_kc in self.links_strengths.keys():
            if target_kc in self.links_strengths[source_kc].keys():
                return self.links_strengths[source_kc][target_kc]
        return 'not existing'

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
        return [
            parent for parent in self.domain.get_kc_parents(kc)
            if self.get_link_strength(parent, kc) != 'not existing']

    def get_learner_pool_kc_children(self, kc):
        return [child for child in self.domain.get_kc_children(kc)
                if self.get_link_strength(kc, child) != 'not existing']

    def set_learn(self, kc, val):
        self.learns[kc] = val

    def set_prior(self, kc, val):
        self.priors[kc] = val

    def set_slip(self, kc, val):
        self.slips[kc] = val

    def set_guess(self, kc, val):
        self.guesses[kc] = val

    def set_forget(self, kc, val):
        self.forgets[kc] = val

    def get_learn(self, kc):
        return self.learns[kc]

    def get_prior(self, kc):
        return self.priors[kc]

    def get_slip(self, kc):
        return self.slips[kc]

    def get_guess(self, kc):
        return self.guesses[kc]

    def get_forget(self, kc):
        return self.forgets[kc]

    def dbn_inference(self, initial_state={}, evaluated_kc=None):
        """
        Return the parameterized 2TBN that corresponds to learning into the Domain for the learners that belong to
        LearnerPool.
        :return: the 2TBN pyAgrum object
        """
        knowledge_components = self.get_knowledge_components()
        bn = gum.BayesNet()

        # setting the general structure
        for kc in knowledge_components:
            bn.add(gum.LabelizedVariable(f"({kc.name})0", f"({kc.name})0", 2))
            bn.cpt(f"({kc.name})0").fillWith([1-self.priors[kc], self.priors[kc]])
            bn.add(gum.LabelizedVariable(f"({kc.name})t", f"({kc.name})t", 2))
            bn.add(gum.LabelizedVariable(f"eval({kc.name})t", f"eval({kc.name})t", 2))
            bn.addArc(f"({kc.name})0", f"({kc.name})t")
            bn.addArc(f"({kc.name})t", f"eval({kc.name})t")
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 0}] = [1 - self.guesses[kc], self.guesses[kc]]
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 1}] = [self.slips[kc], 1 - self.slips[kc]]

        for kc in knowledge_components:
            kc_parents = self.get_learner_pool_kc_parents(kc)

            for parent in kc_parents:
                bn.addArc(f"({parent.name})0", f"({kc.name})t")
                bn.addArc(f"({kc.name})t", f"({parent.name})t")

        for kc in knowledge_components:
            kc_parents = self.get_learner_pool_kc_parents(kc)
            kc_children = self.get_learner_pool_kc_children(kc)

            n_parents, n_children = len(kc_parents), len(kc_children)
            combinations = [{**{f"({kc.name})0": comb[0]},
                             **{f"({kc_parents[i].name})0": comb[i + 1] for i in range(n_parents)},
                             **{f"({kc_children[j].name})t": comb[j + n_parents + 1] for j in
                                range(n_children)}}
                            for comb in list(itertools.product([0, 1],
                                                               repeat=n_parents + n_children + 1))]

            for comb in combinations:
                comb_values = list(comb.values())  # array of all boolean values of the condition
                if comb[f"({kc.name})0"] == 1:  # the KC was mastered at time t-1
                    cond_pba = 1 - self.forgets[kc]
                elif sum(comb_values[n_parents + 1:]) >= 1:  # one of the children of kc has been mastered
                    # on cherche les children qui sont maîtrisés à temps t
                    mastered_child_kcs_idx = [
                        i - (n_parents + 1) for i in range(n_parents + 1, len(comb_values)) if comb_values[i] == 1]
                    children_conditional_probabilities = [
                        self.get_conditional_probability(kc, {child: 1}) for child in
                        [kc_children[j] for j in mastered_child_kcs_idx]
                    ]
                    cond_pba = max(children_conditional_probabilities)  # TODO: change for real formula
                elif n_parents > 0:  # kc have parents
                    not_mastered_parent_kcs_idx = [i - 1 for i in range(1, n_parents + 1) if comb_values[i] == 0]
                    cond_pba = self.learns[kc] * np.prod([
                        self.get_learn(parent) * self.get_conditional_probability(kc, {parent: 1})
                        + self.get_conditional_probability(kc, {parent: 0}) for parent in
                        [kc_parents[j] for j in not_mastered_parent_kcs_idx]])

                else:
                    cond_pba = self.learns[kc] if (evaluated_kc is None or evaluated_kc == kc) else 0

                bn.cpt(f"({kc.name})t")[comb] = [1 - cond_pba, cond_pba]

        return bn

    def is_kc_learnable(self, kc, evaluated_kc, learn_prop):
        if learn_prop == 'all':
            return True
        if kc is evaluated_kc:
            return True
        if kc in self.get_learner_pool_kc_children(evaluated_kc) or self.get_learner_pool_kc_children(evaluated_kc):
            return True
        return False

    def two_step_bn(self, initial_state, evaluated_kc=None, learn_prop='one_kc'):
        """
        Return the parameterized 2TBN that corresponds to learning into the Domain for the learners that belong to
        LearnerPool.
        :return: the 2TBN pyAgrum object
        """
        knowledge_components = self.get_knowledge_components()
        bn = gum.BayesNet()

        # setting the general structure
        for kc in knowledge_components:
            bn.add(gum.LabelizedVariable(f"({kc.name})0", f"({kc.name})0", 2))
            bn.cpt(f"({kc.name})0").fillWith([1-initial_state[kc.name], initial_state[kc.name]])
            bn.add(gum.LabelizedVariable(f"({kc.name})t", f"({kc.name})t", 2))
            bn.add(gum.LabelizedVariable(f"eval({kc.name})0", f"eval({kc.name})0", 2))
            bn.add(gum.LabelizedVariable(f"eval({kc.name})t", f"eval({kc.name})t", 2))
            bn.addArc(f"({kc.name})0", f"({kc.name})t")
            bn.addArc(f"({kc.name})0", f"eval({kc.name})0")
            bn.addArc(f"({kc.name})t", f"eval({kc.name})t")
            bn.cpt(f"eval({kc.name})0")[{f"({kc.name})0": 0}] = [1 - self.guesses[kc], self.guesses[kc]]
            bn.cpt(f"eval({kc.name})0")[{f"({kc.name})0": 1}] = [self.slips[kc], 1 - self.slips[kc]]
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 0}] = [1 - self.guesses[kc], self.guesses[kc]]
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 1}] = [self.slips[kc], 1 - self.slips[kc]]

        for kc in knowledge_components:
            kc_parents = self.get_learner_pool_kc_parents(kc)

            for parent in kc_parents:
                bn.addArc(f"({parent.name})0", f"({kc.name})t")
                bn.addArc(f"({kc.name})t", f"({parent.name})t")
        for kc in knowledge_components:
            kc_children = self.get_learner_pool_kc_children(kc)
            kc_parents = self.get_learner_pool_kc_parents(kc)

            n_parents, n_children = len(kc_parents), len(kc_children)
            combinations = [{**{f"({kc.name})0": comb[0]},
                             **{f"({kc_parents[i].name})0": comb[i + 1] for i in range(n_parents)},
                             **{f"({kc_children[j].name})t": comb[j + n_parents + 1] for j in
                                range(n_children)}}
                            for comb in list(itertools.product([0, 1],
                                                               repeat=n_parents + n_children + 1))]
            for comb in combinations:
                comb_values = list(comb.values())  # array of all boolean values of the condition
                if comb[f"({kc.name})0"] == 1:  # the KC was mastered at time t-1
                    cond_pba = 1 - self.forgets[kc]
                elif sum(comb_values[n_parents + 1:]) >= 1:  # one of the children of kc has been mastered
                    # on cherche les children qui sont maîtrisés à temps t
                    mastered_child_kcs_idx = [
                        i - (n_parents + 1) for i in range(n_parents + 1, len(comb_values)) if comb_values[i] == 1]
                    children_conditional_probabilities = [
                        self.get_conditional_probability(kc, {child: 1}) for child in
                        [kc_children[j] for j in mastered_child_kcs_idx]
                    ]
                    cond_pba = max(children_conditional_probabilities)  # TODO: change for real formula
                elif n_parents > 0:  # kc have parents
                    not_mastered_parent_kcs_idx = [i - 1 for i in range(1, n_parents + 1) if comb_values[i] == 0]
                    cond_pba = self.learns[kc] * np.prod([
                        self.get_learn(parent) * self.get_conditional_probability(kc, {parent: 1})
                        + self.get_conditional_probability(kc, {parent: 0}) for parent in
                        [kc_parents[j] for j in not_mastered_parent_kcs_idx]]) if self.is_kc_learnable(kc, evaluated_kc, learn_prop) else 0
                else:
                    cond_pba = self.learns[kc] if self.is_kc_learnable(kc, evaluated_kc, learn_prop) else 0

                bn.cpt(f"({kc.name})t")[comb] = [1 - cond_pba, cond_pba]

        return bn

    def get_evidences(self, evaluation, direction='all'):
        kc = evaluation[0]
        # parents evidences
        evidences = {
                f"({evaluation[0].name})t": [self.get_guess(kc),
                                             1 - self.get_guess(kc)] if evaluation[1]
                else [1 - self.get_slip(kc), self.get_slip(kc)]
        }

        if direction in ('all', 'parents'):
            if len(self.get_learner_pool_kc_parents(kc)) > 0:
                recursive_parents = self.get_recursive_parents(kc)
                for i in range(len(recursive_parents)):
                    parent, cond_pba = recursive_parents[i]
                    evidences[f"({parent.name})t"] = [self.get_guess(kc), (1-self.get_guess(kc))*cond_pba] if evaluation[1] else [.6, .4]
        if direction in ('all', 'children'):
            if len(self.get_learner_pool_kc_children(kc)) > 0:
                recursive_children = self.get_recursive_children(kc)
                for i in range(len(recursive_children)):
                    child, cond_pba = recursive_children[i]
                    evidences[f"({child.name})t"] = [.6, .4] if evaluation[1] else [(1-self.get_slip(kc))*cond_pba, self.get_slip(kc)]

        return evidences

    def get_recursive_parents(self, root_kc):
        parents = []

        def _get_leaf_nodes(kc, cond_pba):
            if kc is not None:
                if len(self.get_learner_pool_kc_parents(kc)) == 0:
                    parents.append([kc, cond_pba])
                for parent in self.get_learner_pool_kc_parents(kc):
                    _get_leaf_nodes(parent, cond_pba * self.get_conditional_probability(parent, {kc: True}))

        _get_leaf_nodes(root_kc, 1.)
        return parents

    def get_recursive_children(self, root_kc):
        children = []

        def _get_leaf_nodes(kc, cond_pba):
            if kc is not None:
                if len(self.get_learner_pool_kc_children(kc)) == 0:
                    children.append([kc, cond_pba])
                for child in self.get_learner_pool_kc_children(kc):
                    _get_leaf_nodes(child, cond_pba * self.get_conditional_probability(kc, {child: True}))

        _get_leaf_nodes(root_kc, 1.)
        return children

    def success_two_step_bn(self, initial_state, evaluated_kc=None, learn_prop='one_kc'):
        """
        Return the parameterized 2TBN that corresponds to learning into the Domain for the learners that belong to
        LearnerPool.
        :return: the 2TBN pyAgrum object
        """
        knowledge_components = self.get_knowledge_components()
        bn = gum.BayesNet()

        # setting the general structure
        for kc in knowledge_components:
            bn.add(gum.LabelizedVariable(f"({kc.name})0", f"({kc.name})0", 2))
            bn.cpt(f"({kc.name})0").fillWith([1-initial_state[kc.name], initial_state[kc.name]])
            bn.add(gum.LabelizedVariable(f"({kc.name})t", f"({kc.name})t", 2))
            bn.add(gum.LabelizedVariable(f"eval({kc.name})t", f"eval({kc.name})t", 2))
            bn.addArc(f"({kc.name})0", f"({kc.name})t")
            bn.addArc(f"({kc.name})t", f"eval({kc.name})t")
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 0}] = [1 - self.guesses[kc], self.guesses[kc]]
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 1}] = [self.slips[kc], 1 - self.slips[kc]]

        for kc in knowledge_components:
            kc_children = self.get_learner_pool_kc_children(kc)
            for child in kc_children:
                bn.addArc(f"({kc.name})t", f"({child.name})t")

        for kc in knowledge_components:
            kc_parents = self.get_learner_pool_kc_parents(kc)
            n_parents= len(kc_parents)
            combinations = [{**{f"({kc.name})0": comb[0]},
                             **{f"({kc_parents[i].name})t": comb[i + 1] for i in range(n_parents)}}
                            for comb in list(itertools.product([0, 1], repeat=n_parents + 1))]

            for comb in combinations:
                comb_values = list(comb.values())  # array of all boolean values of the condition
                if comb[f"({kc.name})0"] == 1:  # the KC was mastered at time t-1
                    cond_pba = 1 - self.forgets[kc]
                elif n_parents > 0:  # kc have parents
                    not_mastered_parent_kcs_idx = [i - 1 for i in range(1, n_parents + 1) if comb_values[i] == 0]
                    cond_pba = self.learns[kc] * np.prod([
                        self.get_learn(parent) * self.get_conditional_probability(kc, {parent: 1})
                        + self.get_conditional_probability(kc, {parent: 0}) for parent in
                        [kc_parents[j] for j in not_mastered_parent_kcs_idx]]) if self.is_kc_learnable(kc, evaluated_kc, learn_prop) else 0
                else:
                    cond_pba = self.learns[kc] if self.is_kc_learnable(kc, evaluated_kc, learn_prop) else 0
                bn.cpt(f"({kc.name})t")[comb] = [1 - cond_pba, cond_pba]
        return bn


    def fail_two_step_bn(self, initial_state, evaluated_kc=None, learn_prop='one_kc'):
        """
        Return the parameterized 2TBN that corresponds to learning into the Domain for the learners that belong to
        LearnerPool.
        :return: the 2TBN pyAgrum object
        """
        knowledge_components = self.get_knowledge_components()
        bn = gum.BayesNet()

        # setting the general structure
        for kc in knowledge_components:
            bn.add(gum.LabelizedVariable(f"({kc.name})0", f"({kc.name})0", 2))
            bn.cpt(f"({kc.name})0").fillWith([1-initial_state[kc.name], initial_state[kc.name]])
            bn.add(gum.LabelizedVariable(f"({kc.name})t", f"({kc.name})t", 2))
            bn.add(gum.LabelizedVariable(f"eval({kc.name})t", f"eval({kc.name})t", 2))
            bn.addArc(f"({kc.name})0", f"({kc.name})t")
            bn.addArc(f"({kc.name})t", f"eval({kc.name})t")
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 0}] = [1 - self.guesses[kc], self.guesses[kc]]
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 1}] = [self.slips[kc], 1 - self.slips[kc]]

        for kc in knowledge_components:
            kc_parents = self.get_learner_pool_kc_parents(kc)
            for parent in kc_parents:
                bn.addArc(f"({kc.name})t", f"({parent.name})t")

        for kc in knowledge_components:
            kc_children = self.get_learner_pool_kc_children(kc)
            n_children = len(kc_children)

            combinations = [{**{f"({kc.name})0": comb[0]},
                             **{f"({kc_children[j].name})t": comb[j + 1] for j in
                                range(n_children)}}
                            for comb in list(itertools.product([0, 1],
                                                               repeat=n_children + 1))]
            for comb in combinations:
                comb_values = list(comb.values())  # array of all boolean values of the condition
                if comb[f"({kc.name})0"] == 1:  # the KC was mastered at time t-1
                    cond_pba = 1 - self.forgets[kc]
                elif sum(comb_values[1:]) >= 1:  # one of the children of kc has been mastered
                    # on cherche les children qui sont maîtrisés à temps t
                    mastered_child_kcs_idx = [
                        i - 1 for i in range(1, len(comb_values)) if comb_values[i] == 1]
                    children_conditional_probabilities = [
                        self.get_conditional_probability(kc, {child: 1}) for child in
                        [kc_children[j] for j in mastered_child_kcs_idx]
                    ]
                    cond_pba = max(children_conditional_probabilities)  # TODO: change for real formula
                else:
                    cond_pba = self.learns[kc] if self.is_kc_learnable(kc, evaluated_kc, learn_prop) else 0

                bn.cpt(f"({kc.name})t")[comb] = [1 - cond_pba, cond_pba]

        return bn
