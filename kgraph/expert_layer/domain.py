import pandas as pd
import numpy as np
from kgraph.expert_layer.knowledge_components import KnowledgeComponent
import pyAgrum as gum


class Domain(object):

    def __init__(self, knowledge_components=None, prerequisite_links=None):
        """
        The class Domain corresponds to the modelization of the expert knowledge on the learning domain. It is composed
        of two elements: knowledge components (elements that compose the expected knowledge -- overlay model) and the
        prerequisite links
        """
        if knowledge_components is None:
            knowledge_components = []
        self.knowledge_components = knowledge_components
        if prerequisite_links is None:
            prerequisite_links = []
        self.links = prerequisite_links
        self.link_df = pd.DataFrame([[
            i, self.links[i].source, self.links[i].target] for i in range(len(self.links))],
            columns=['index', 'source', 'target'])

    def __str__(self):
        string = f"Domain on {len(self.knowledge_components)} KCs."
        return string

    def add_kc(self, kc: KnowledgeComponent):
        """
        Add a given knowledge component into the Domain's knowledge components.
        :param kc: the knowledge component to be added
        """
        assert isinstance(kc, KnowledgeComponent), "Entered KC is not a KnowledgeComponent object."
        if kc not in self.knowledge_components:
            self.knowledge_components.append(kc)

    def remove_kc(self, kc):
        """
        Remove a given knowledge component from the Domain's knowledge components.
        :param kc: the knowledge component to be removed
        """
        if isinstance(kc, (list, np.ndarray)):
            for k_el in kc:
                self.knowledge_components.remove(k_el)
        else:
            self.knowledge_components.remove(kc)

    def get_knowledge_components(self):
        """
        Return all Domain's knowledge components.
        :return: the list of Domain's knowledge components
        """
        return self.knowledge_components

    def get_links(self):
        """
        Return all Domain's links.
        :return: the list of Domain's links
        """
        return self.links

    def get_exercises(self):
        return [kc.exercise for kc in self.knowledge_components]

    def get_kc_by_name(self, kc_name):
        knowledge_components = [kc for kc in self.knowledge_components if kc.name == kc_name]
        if len(knowledge_components) > 0:
            return knowledge_components[0]
        else:
            return Exception()

    def get_kc_parents(self, kc):
        return pd.unique(self.link_df.loc[self.link_df['target'] == kc]['source'])

    def get_kc_children(self, kc):
        return pd.unique(self.link_df.loc[self.link_df['source'] == kc]['target'])

