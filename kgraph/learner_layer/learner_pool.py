import numpy as np
import dill
from kgraph.expert_layer.domain import Domain
from kgraph.learner_layer.learner import Learner
import pyAgrum as gum
import itertools

link_strength_2_cond_pba = {'strong': .9,
                            'weak': .7}


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
        self.prior = .2
        self.slip = .1
        self.guess = .1
        self.learn = .2
        self.forget = .01
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

    def get_link_strength(self, source_kc, target_kc):
        return self.links_strengths[source_kc][target_kc]

    def get_conditional_probability(self, source_kc, target_kc):
        link_strength = self.get_link_strength(source_kc, target_kc)
        return link_strength_2_cond_pba[link_strength]

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
        return [parent for parent in self.domain.get_kc_parents(kc) if self.get_link_strength(parent, kc) is not None]

    def get_learner_pool_kc_children(self, kc):
        return [child for child in self.domain.get_kc_children(kc) if self.get_link_strength(kc, child) is not None]

    def get_learn(self, kc):
        return self.learn

    def get_prior(self, kc):
        return self.prior

    def get_slip(self):
        return self.slip

    def get_guess(self):
        return self.guess

    def dbn_inference(self):
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
            bn.cpt(f"({kc.name})0").fillWith([.8, .2])
            bn.add(gum.LabelizedVariable(f"({kc.name})t", f"({kc.name})t", 2))
            bn.add(gum.LabelizedVariable(f"eval({kc.name})t", f"eval({kc.name})t", 2))
            bn.addArc(f"({kc.name})0", f"({kc.name})t")
            bn.addArc(f"({kc.name})t", f"eval({kc.name})t")
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 0}] = [1 - self.guess, self.guess]
            bn.cpt(f"eval({kc.name})t")[{f"({kc.name})t": 1}] = [self.slip, 1 - self.slip]

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
                if sum(comb_values[n_parents + 1:]) < 1:  # there is no mastered child of kc
                    if n_parents > 0:  # kc have parents
                        parent_subcondition = {k: comb[k] for k in list(comb)[1:n_parents + 1]}
                        unmastered_parent_kcs = [parent for parent in kc_parents
                                                 if not parent_subcondition[f"({parent.name})0"]]
                        # in order to master kc, the learner will have to learn all unmastered parents and then to
                        # learn kc

                        parent_coef = np.prod([self.get_learn(parent) for parent in unmastered_parent_kcs])
                        self_coef = self.get_learn(kc) if not comb[f"({kc.name})0"] else 1 - self.forget

                        cond_pba = self_coef * parent_coef
                    else:  # kc has no parents
                        cond_pba = self.get_learn(kc) if not comb[f"({kc.name})0"] else 1 - self.forget

                else:  # there is children
                    children_subcondition = {k: comb[k] for k in list(comb)[1 + n_parents:]}
                    mastered_child_kcs = [child for child in kc_children
                                          if children_subcondition[f"({child.name})t"]]
                    cond_pba = max([self.get_conditional_probability(child, kc) for child in mastered_child_kcs]
                                   + [self.get_learn(kc) if not comb[f"({kc.name})0"] else 1 - self.forget])
                bn.cpt(f"({kc.name})t")[comb] = [1 - cond_pba, cond_pba]
        return bn
