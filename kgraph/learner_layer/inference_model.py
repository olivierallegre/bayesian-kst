import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import numpy as np


def get_root_nodes(knowledge_components, link_strengths):
    nodes_with_parents = [key for key in link_strengths.keys()]
    root_nodes = [kc for kc in knowledge_components if kc not in nodes_with_parents]
    return root_nodes


def get_leaf_nodes(knowledge_components, link_strengths):
    nodes_with_children = [y for x in [list(link_strengths[key].keys()) for key in link_strengths.keys()] for y in x]
    leaf_nodes = [kc for kc in knowledge_components if kc not in nodes_with_children]
    return leaf_nodes


def showBN(bn):
    print('---------------------------------')
    for i in bn.nodes():
        print('{0} : {1}'.format(i, str(bn.variable(i))))
    print('---------------------------------')

    for (i, j) in bn.arcs():
        print('{0}->{1}'.format(bn.variable(i).name(), bn.variable(j).name()))
    print('---------------------------------')


def unroll_2tbn(temp_bn, n_steps):
    dbn = gdyn.unroll2TBN(temp_bn, n_steps)
    return dbn


class NoisyANDInferenceModel:

    def __init__(self, learner_pool, params):
        self.associated_learner_pool = learner_pool
        self.bn = gum.BayesNet()
        self.params = params if params else self._set_default_params()
        self.setup_dbn()

    def _set_default_params(self):
        link_strengths = self.associated_learner_pool.get_link_strengths()
        params = {'c': {}, 's': {}}
        for kc in link_strengths.keys():
            for parent in link_strengths[kc].keys():
                params['c'][parent], params['s'][parent] = {}, {}
        for kc in link_strengths.keys():
            for parent in link_strengths[kc].keys():
                params['c'][parent][kc] = .9
                params['s'][parent][kc] = .1
        return params

    def get_link_strengths(self):
        return self.associated_learner_pool.get_link_strengths()

    def get_c_param(self, source, target):
        return self.params['c'][source][target]

    def set_c_param(self, source, target, value):
        self.params['c'][source][target] = value

    def get_s_param(self, source, target):
        return self.params['s'][source][target]

    def set_s_param(self, source, target, value):
        self.params['s'][source][target] = value

    def setup_dbn(self):
        knowledge_components = self.associated_learner_pool.get_knowledge_components()
        priors = {kc: self.associated_learner_pool.get_prior(kc) for kc in knowledge_components}
        link_strengths = self.get_link_strengths()

        # Introduce the structure of the temporal relationships between same KC's nodes
        for kc in knowledge_components:
            # Introduce node for KC at time 0
            if kc in link_strengths.keys():
                self.bn.addAND(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.bn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.bn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])

            self.bn.add(gum.LabelizedVariable(f"(Z[({kc.id})0->({kc.id})t])t", f"(Z[({kc.id})0->({kc.id})t])t", 2))
            self.bn.addArc(f"({kc.id})0", f"(Z[({kc.id})0->({kc.id})t])t")

            self.bn.addAND(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.bn.addArc(f"(Z[({kc.id})0->({kc.id})t])t", f"({kc.id})t")

            self.bn.cpt(f"(Z[({kc.id})0->({kc.id})t])t")[{f"({kc.id})0": 0}] = [1, 0]
            self.bn.cpt(f"(Z[({kc.id})0->({kc.id})t])t")[{f"({kc.id})0": 1}] = [0, 1]

        for kc in knowledge_components:
            parents = self.associated_learner_pool.get_kc_parents(kc)
            if parents:
                for parent in parents:
                    self.bn.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])0", f"(Z[{parent.id}->{kc.id}])0", 2))
                    self.bn.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])t", f"(Z[{parent.id}->{kc.id}])t", 2))
                    c, s = self.get_c_param(parent, kc), self.get_s_param(parent, kc)
                    self.bn.addArc(f"({parent.id})0", f"(Z[{parent.id}->{kc.id}])0")
                    self.bn.addArc(f"({parent.id})t", f"(Z[{parent.id}->{kc.id}])t")

                    self.bn.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s, s]
                    self.bn.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c, c]

                    self.bn.cpt(f"(Z[{parent.id}->{kc.id}])t")[{f"({parent.id})t": 0}] = [1 - s, s]
                    self.bn.cpt(f"(Z[{parent.id}->{kc.id}])t")[{f"({parent.id})t": 1}] = [1 - c, c]

                    self.bn.addArc(f"(Z[{parent.id}->{kc.id}])t", f"({kc.id})t")
                    self.bn.addArc(f"(Z[{parent.id}->{kc.id}])0", f"({kc.id})0")

    def predict_learner_knowledge_states_from_learner_traces(self, learner_traces):
        knowledge_components = self.associated_learner_pool.get_knowledge_components()
        bn = unroll_2tbn(self.bn, len(learner_traces)+1)
        # Setup the soft evidences in the BNow
        evidences = {}
        for i, trace in enumerate(learner_traces):
            evaluated_kc = trace.get_kc()
            success = trace.get_success()
            exercise = trace.get_exercise()
            guess, slip = self.associated_learner_pool.get_guess(exercise), self.associated_learner_pool.get_slip(exercise)
            learn, forget = self.associated_learner_pool.get_learn(evaluated_kc), self.associated_learner_pool.get_forget(evaluated_kc)

            bn.add(gum.LabelizedVariable(f"exercise({exercise.id}){i}", f"exercise({exercise.id}){i}", 2))
            bn.addArc(f"({evaluated_kc.id}){i}", f"exercise({exercise.id}){i}")

            bn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 0}] = [1 - guess, guess]
            bn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 1}] = [slip, 1 - slip]
            evidences[f"exercise({exercise.id}){i}"] = int(success)

            bn.cpt(f"(Z[({evaluated_kc.id})0->({evaluated_kc.id})t]){i+1}")[{f"({evaluated_kc.id}){i}": 0}] = [
                1 - learn, learn
            ]
            bn.cpt(f"(Z[({evaluated_kc.id})0->({evaluated_kc.id})t]){i+1}")[{f"({evaluated_kc.id}){i}": 1}] = [
                forget, 1 - forget
            ]
            """
            evidences[f"({evaluated_kc.id}){i}"] = [guess, 1 - guess] if success else [
                1 - slip, slip]
        """
        # Setup the inference
        ie = gum.LazyPropagation(bn)
        ie.setEvidence(evidences)
        ie.makeInference()
        knowledge_states = {}
        for kc in knowledge_components:
            knowledge_states[f"{kc.id}"] = ie.posterior(bn.idFromName(f"({kc.id}){len(evidences.keys())}"))[1]
        return knowledge_states


class NoisyORInferenceModel:

    def __init__(self, learner_pool, c_params):
        self.associated_learner_pool = learner_pool
        self.bn = gum.BayesNet()
        self.c_params = c_params if c_params else self._set_default_c_params()
        self.setup_dbn()

    def _set_default_c_params(self):
        link_strengths = self.associated_learner_pool.get_link_strengths()
        c_params = {}
        for kc in link_strengths.keys():
            for parent in link_strengths[kc].keys():
                c_params[parent] = {}
        for kc in link_strengths.keys():
            for parent in link_strengths[kc].keys():
                c_params[parent][kc] = .9
        return c_params

    def get_link_strengths(self):
        return self.associated_learner_pool.get_link_strengths()

    def get_c_param(self, source, target):
        return self.c_params[source][target]

    def set_c_param(self, source, target, value):
        self.c_params[source][target] = value

    def setup_dbn(self):
        knowledge_components = self.associated_learner_pool.get_knowledge_components()
        priors = {kc: self.associated_learner_pool.get_prior(kc) for kc in knowledge_components}
        link_strengths = self.get_link_strengths()

        # Introduce the structure of the temporal relationships between same KC's nodes
        leaf_nodes = get_leaf_nodes(knowledge_components, link_strengths)
        for kc in knowledge_components:
            # Introduce node for KC at time 0
            if kc in leaf_nodes:
                self.bn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.bn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            else:
                self.bn.addOR(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))

            self.bn.add(gum.LabelizedVariable(f"(Z[({kc.id})0->({kc.id})t])t", f"(Z[({kc.id})0->({kc.id})t])t", 2))
            self.bn.addArc(f"({kc.id})0", f"(Z[({kc.id})0->({kc.id})t])t")

            self.bn.addOR(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.bn.addArc(f"(Z[({kc.id})0->({kc.id})t])t", f"({kc.id})t")

            self.bn.cpt(f"(Z[({kc.id})0->({kc.id})t])t")[{f"({kc.id})0": 0}] = [1, 0]
            self.bn.cpt(f"(Z[({kc.id})0->({kc.id})t])t")[{f"({kc.id})0": 1}] = [0, 1]

        for kc in knowledge_components:
            children = self.associated_learner_pool.get_learner_pool_kc_children(kc)
            if children:
                for child in children:
                    self.bn.add(
                        gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])0", f"(Z[{child.id}->{kc.id}])0", 2))
                    self.bn.add(
                        gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])t", f"(Z[{child.id}->{kc.id}])t", 2))

                    c = self.get_c_param(kc, child)
                    self.bn.addArc(f"({child.id})0", f"(Z[{child.id}->{kc.id}])0")
                    self.bn.addArc(f"({child.id})t", f"(Z[{child.id}->{kc.id}])t")

                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])0")[{f"({child.id})0": 0}] = [1, 0]
                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])0")[{f"({child.id})0": 1}] = [1 - c, c]

                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})t": 0}] = [1, 0]
                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})t": 1}] = [1 - c, c]

                    self.bn.addArc(f"(Z[{child.id}->{kc.id}])t", f"({kc.id})t")
                    self.bn.addArc(f"(Z[{child.id}->{kc.id}])0", f"({kc.id})0")

    def predict_learner_knowledge_states_from_learner_traces(self, learner_traces):
        knowledge_components = self.associated_learner_pool.get_knowledge_components()
        bn = unroll_2tbn(self.bn, len(learner_traces)+1)
        # Setup the soft evidences in the BNow
        evidences = {}
        for i, trace in enumerate(learner_traces):
            evaluated_kc = trace.get_kc()
            success = trace.get_success()
            exercise = trace.get_exercise()
            guess, slip = self.associated_learner_pool.get_guess(exercise), self.associated_learner_pool.get_slip(exercise)
            learn, forget = self.associated_learner_pool.get_learn(evaluated_kc), self.associated_learner_pool.get_forget(evaluated_kc)

            bn.add(gum.LabelizedVariable(f"exercise({exercise.id}){i}", f"exercise({exercise.id}){i}", 2))
            bn.addArc(f"({evaluated_kc.id}){i}", f"exercise({exercise.id}){i}")

            bn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 0}] = [1 - guess, guess]
            bn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 1}] = [slip, 1 - slip]
            evidences[f"exercise({exercise.id}){i}"] = int(success)

            bn.cpt(f"(Z[({evaluated_kc.id})0->({evaluated_kc.id})t]){i+1}")[{f"({evaluated_kc.id}){i}": 0}] = [
                1 - learn, learn
            ]
            bn.cpt(f"(Z[({evaluated_kc.id})0->({evaluated_kc.id})t]){i+1}")[{f"({evaluated_kc.id}){i}": 1}] = [
                forget, 1 - forget
            ]
            """
            evidences[f"({evaluated_kc.id}){i}"] = [guess, 1 - guess] if success else [
                1 - slip, slip]
        """
        # Setup the inference
        ie = gum.LazyPropagation(bn)
        ie.setEvidence(evidences)
        ie.makeInference()
        knowledge_states = {}
        for kc in knowledge_components:
            knowledge_states[f"{kc.id}"] = ie.posterior(bn.idFromName(f"({kc.id}){len(evidences.keys())}"))[1]
        return knowledge_states