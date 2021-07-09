import numpy as np
from kgraph.helpers.truthtable import truthtable, bool_list_to_int, int_to_bool_list
from kgraph.expert_layer.knowledge_components import KnowledgeComponent


class LinkModel(object):

    def __init__(self, links):
        self.links = {}
        # TODO: check if links are consistent
        for link in links:
            kc = link.get_kc()
            if kc not in self.links.keys():
                self.links[kc] = {'from_parents': None, 'from_children': None}
            if isinstance(link, LinkFromParents):
                self.links[kc]['from_parents'] = link
            elif isinstance(link, LinkFromChildren):
                self.links[kc]['from_children'] = link

    def __str__(self):
        string = f"Link model with {len(self.links.keys())} links:\n"
        for link in self.links.keys():
            string += "- " + self.links[link].__str__() + "\n"
        return string

    def add_link(self, link):
        assert isinstance(link, Link), "Given link is not a Link object."
        knowledge_component = link.get_kc()
        if knowledge_component not in self.links.keys():
            self.links[knowledge_component] = {'from_parents': None, 'to_children': None}
        if isinstance(link, LinkFromParents):
            self.links[knowledge_component]['from_parents'] = link
        if isinstance(link, LinkFromChildren):
            self.links[knowledge_component]['from_children'] = link

    def remove_link(self, link):
        if isinstance(link, (list, np.ndarray)):
            for element in link:
                self.links.remove_link(element)
        else:
            knowledge_component = link.get_kc()
            if isinstance(link, LinkFromChildren):
                del self.links[knowledge_component]['from_children']
            elif isinstance(link, LinkFromParents):
                del self.links[knowledge_component]['from_parents']

    def set_links(self, links):
        for link in links:
            self.add_link(link)

    def get_links(self, kc):
        return self.links[kc]

    def get_parents(self, kc):
        if kc in self.links.keys():
            if self.links[kc]["from_parents"]:
                return self.links[kc]["from_parents"].get_parents()
        return []

    def get_children(self, kc):
        if kc in self.links.keys():
            if self.links[kc]["from_children"]:
                return self.links[kc]["from_children"].get_children()
        return []

    def get_all_links(self):
        # TODO : outdated
        all_links = []
        for key in self.links.keys():
            if isinstance(self.links[key], LinkFromChildren):
                for child in self.links[key].linked_knowledge_components:
                    all_links.append((key, child))
        return all_links

    def get_roots(self):
        roots = [key for key in self.links if not self.links[key]["from_parents"]]
        return roots

    def get_all_parents(self, knowledge_components):
        # TODO: refactor
        if not knowledge_components:
            return []
        elif isinstance(knowledge_components, list):
            parents = []
            for kc in knowledge_components:
                parents = list(set(parents + self.get_all_parents(kc)))
            return parents
        else:
            parents = self.get_parents(knowledge_components)
            return list(set(list(parents) + self.get_all_parents(parents)))

    def get_all_children(self, knowledge_components):
        # TODO: refactor
        if not knowledge_components:
            return []
        elif isinstance(knowledge_components, list):
            children = []
            for kc in knowledge_components:
                children = list(set(children + self.get_all_children(kc)))
            return children
        else:
            children = self.get_children(knowledge_components)
            return list(set(list(children) + self.get_all_children(children)))


class Link(object):

    def __init__(self, knowledge_component, linked_knowledge_components=[], probability_vector=[]):
        assert isinstance(knowledge_component, KnowledgeComponent), f"First argument must be a KnowledgeComponent," \
                                                                    f" now {type(knowledge_component)}."
        self.knowledge_component = knowledge_component

        if linked_knowledge_components:
            assert isinstance(linked_knowledge_components, list) and isinstance(linked_knowledge_components[0],
                                                                                KnowledgeComponent), \
                f"Second argument must be a list of KnowledgeComponents, now {type(linked_knowledge_components)} of" \
                f" {type(linked_knowledge_components[0])}."
        self.linked_knowledge_components = linked_knowledge_components
        self.probability_vector = probability_vector

    def get_kc(self):
        return self.knowledge_component

    def get_pc_vec(self):
        pc_vec = []
        conds = truthtable(len(self.parents) + len(self.children))
        for cond in conds:
            cond_p_prob = self.p_vec[bool_list_to_int(cond[:len(self.parents)])] if any(self.p_vec) else 1.
            cond_c_prob = self.c_vec[bool_list_to_int(cond[len(self.parents):])] if any(self.c_vec) else 1.
            print(self.p_vec, self.c_vec)
            print(cond, cond_p_prob, cond_c_prob)
            pc_vec.append(cond_p_prob * cond_c_prob)
        return pc_vec

    def get_conditional_probability_table(self, dynamic=False):
        table = []*2**len(self.linked_knowledge_components)
        return self.probability_vector


class LinkFromParents(Link):
    """
    This class only corresponds to links from a given knowledge component to a set of parents, that is to say the
    prerequisites of the knowledge component.
    """
    def __init__(self, knowledge_component, list_of_parents, probability_vector=[]):
        Link.__init__(self, knowledge_component, list_of_parents, probability_vector)
        self.set_parents(list_of_parents)

    def __str__(self):
        """
        Method to print children of self.
        :return:
        """
        to_print = f"LinkToParents from KC #{self.knowledge_component.id} to "
        for parent in self.linked_knowledge_components:
            to_print += f"KC #{parent.id} "
        return to_print

    def get_parents(self):
        return self.linked_knowledge_components

    def diffuse_to_parent(self, self_mastering_probability):
        pass

    def diffuse_to_child(self, self_mastering_probability):
        pass

    def kartable_diffuse_to_parent(self, self_mastering_probability):
        pass

    def kartable_diffuse_to_child(self, self_mastering_probability):
        pass

    def set_parents(self, list_of_parents):
        """
        Method to declare the parents of self.
        :param list_of_parents: list of the knowledge components that are parents of self
        :return:
        """
        if not self.probability_vector:  # no links already declared
            self.linked_knowledge_components = list_of_parents
            self.probability_vector = np.zeros(2 ** len(self.linked_knowledge_components))
            self.probability_vector[-1] = .5
        else:  # some links have been declared - one adds them, verifying they're not already declared
            for parent in self.linked_knowledge_components:
                self.add_parent(parent)

    def add_parent(self, parent):
        """
        Method to add a parent to a knowledge component.
        :param parent: parent to add to self.parents
        :return:
        """
        assert isinstance(parent, KnowledgeComponent), "Parent must be a KnowledgeComponent."
        if parent not in self.linked_knowledge_components:
            self.linked_knowledge_components.append(parent)
            if not (any(self.probability_vector) or all(self.probability_vector)):
                self.probability_vector = np.concatenate((self.probability_vector, self.probability_vector), axis=None)
            else:
                self.probability_vector = np.zeros(2)

    def set_parents_conditioned_probability_vector(self, vec=None, step_by_step=False):
        """
        Set the conditional probability vector of KnowledgeComponent given the mastering of its parents.
        :param vec: the conditional probability vector knowing the states of parents given by converting the index of
        the probability vector in binary (e.g. for B, C parents of A, vec is 4 length : 1st index corresponds to
        B is False, C is False ; 3rd index is B True, C False)
        :param step_by_step: if the form of linked_knowledge_component is unknown, it is possible to set conditional
        probabilities value by value.
        :return: None, only changes the value of self.probability_vector
        """
        if not step_by_step:
            assert len(vec) == 2 ** len(self.linked_knowledge_components), "Length of the probability vector " \
                                                                           "should be 2^n."
            self.probability_vector = vec
        else:
            combinations = truthtable(len(self.linked_knowledge_components))
            for comb in combinations:
                self.probability_vector[bool_list_to_int(comb)] = input(
                    f"Probability  {self.knowledge_component.name} knowing "
                    f"{[parent.name for parent in self.linked_knowledge_components]} value {comb} is :")

    def declare_p_probability(self, probability, cond):
        """
        Function of link value declaration
        Inputs :
        - probability : float that enphasizes the probability that the student knows self Skill knowing cond
        - cond : dict with the form {'skill_id of a parent': True or False, ...}
        """
        assert cond.keys() == [kc.id for kc in self.linked_knowledge_components], "Error: some parents are lacking"
        index = bool_list_to_int([cond[kc.id] for kc in self.linked_knowledge_components])
        self.probability_vector[index] = probability


class LinkFromChildren(Link):

    def __init__(self, knowledge_component, list_of_children, probability_vector=[]):
        Link.__init__(self, knowledge_component, list_of_children, probability_vector)
        self.set_children(list_of_children)

    def __str__(self):
        """
        Method to print children of self.
        :return:
        """
        print = f"LinkToChildren from KC #{self.knowledge_component.id} to "
        for child in self.linked_knowledge_components:
            print += f"KC #{child.id} "
        return print

    def get_children(self):
        return self.linked_knowledge_components

    def diffuse_to_parent(self, self_mastering_probability):
        pass

    def diffuse_to_child(self, self_mastering_probability):
        pass

    def set_children(self, list_of_children):
        """
        Method to declare the children of self.
        :param list_of_children: list of the knowledge components that are children of self
        :return:
        """

        if not self.probability_vector:
            self.linked_knowledge_components = list_of_children
            self.probability_vector = np.ones(2 ** len(self.linked_knowledge_components))
            self.probability_vector[0] = .5
        else:
            for child in list_of_children:
                self.add_child(child)

    def add_child(self, child):
        """
        Method to add a parent to a knowledge component.
        :param child: parent to add to self.parents
        :return:
        """
        assert isinstance(child, KnowledgeComponent), "Child must be a KnowledgeComponent."
        if child not in self.linked_knowledge_components:
            self.linked_knowledge_components.append(child)
            if not (any(self.probability_vector) or all(self.probability_vector)):
                self.probability_vector = np.concatenate((self.probability_vector,
                                                          np.ones(len(self.probability_vector))), axis=None)
            else:
                self.probability_vector = np.array([.5, 1])

    def set_children_conditioned_probability_vector(self, vec=None, step_by_step=False):
        """
        Set the conditional probability vector of KnowledgeComponent given the mastering of its children.
        :param vec: the conditional probability vector knowing the states of children given by converting the index of
        the probability vector in binary (e.g. for B, C children of A, vec is 4 length : 1st index corresponds to
        B is False, C is False ; 3rd index is B True, C False)
        :param step_by_step: if the form of linked_knowledge_component is unknown, it is possible to set conditional
        probabilities value by value.
        :return: None, only changes the value of self.probability_vector
        """
        if not step_by_step:
            assert len(vec) == 2**len(self.linked_knowledge_components), "Length of the probability vector " \
                                                                         "should be 2^n."
            self.probability_vector = vec
        else:
            combinations = truthtable(len(self.linked_knowledge_components))
            for comb in combinations:
                self.probability_vector[bool_list_to_int(comb)] = input(
                    f"Probability  {self.knowledge_component.name} knowing "
                    f"{[child.name for child in self.linked_knowledge_components]} value {comb} is :")

    def declare_c_probability(self, probability, cond):
        """
        Function of link value declaration
        Inputs :
        - probability : float that enphasizes the probability that the student knows self Skill knowing cond
        - cond : dict with the form {'skill_id of a parent': True or False, ...}
        """
        assert cond.keys() == [kc.id for kc in self.linked_knowledge_components], "Error: some parents are lacking"
        index = bool_list_to_int([cond[kc.id] for kc in self.linked_knowledge_components])
        self.probability_vector[index] = probability

