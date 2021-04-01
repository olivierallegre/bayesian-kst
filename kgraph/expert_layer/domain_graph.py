import pandas as pd
import numpy as np
import tqdm
from kgraph.expert_layer.knowledge_components import KnowledgeComponent, ProceduralKnowledgeComponent, \
    DeclarativeKnowledgeComponent
from kgraph.expert_layer.links import LinkModel
from kgraph.resources_layer.exercise import Exercise
from kgraph.resources_layer.exercise_family import ExerciseFamily
from ast import literal_eval


class DomainGraph(object):

    def __init__(self, kc_list=None, link_model=None):
        """
        Initialization of the DomainGraph object.
        :param kc_list: list of the knowledge components that belongs to DomainGraph
        """
        if not kc_list:
            kc_list = []
        self.knowledge_components = kc_list
        if link_model:
            assert isinstance(link_model, LinkModel)
            self.link_model = link_model
        else:
            self.link_model = LinkModel([])

    def __str__(self):
        string = f"Domain graph contains {len(self.knowledge_components)} KC:\n"
        for kc in self.knowledge_components:
            string += "- " + kc.__str__() + "\n"
        string += f"and has " + self.link_model.__str__()
        return string

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DomainGraph):
            other_kc_ids = [kc.id for kc in other.knowledge_components]
            for kc in self.knowledge_components:
                if kc.id not in other_kc_ids:
                    return False
                else:
                    #TODO : check if the KC/ex_fam/exercises are equals too
                    other_kc_ids.remove(kc.id)
            if other_kc_ids:
                return False
            return True
        return False

    def add_kc(self, kc: KnowledgeComponent):
        assert isinstance(kc, KnowledgeComponent), "Entered KC is not a KnowledgeComponent object."
        if kc not in self.knowledge_components:
            self.knowledge_components.append(kc)
        # TODO: update learner

    def remove_kc(self, kc):
        if isinstance(kc, (list, np.ndarray)):
            for k_el in kc:
                self.knowledge_components.remove(k_el)
        else:
            self.knowledge_components.remove(kc)
        # TODO: update learner

    def set_knowledge_components(self, kc_list):
        for kc in kc_list:
            self.add_kc(kc)
        # TODO: update learner

    def set_link_model(self, link_model):
        self.link_model = link_model

    def get_ex_fam_ids(self):
        return [kc.exercise_family.id for kc in self.knowledge_components]

    def to_csv(self, csv_file):
        """
        Export a DomainGraph into a csv file.
        :param csv_file: the path where the csv file is
        :return:  None, only complete the csv file with the DomainGraph data
        """
        # Every exercise that belongs to DomainGraph
        exercise_tuples = [tuple(kc.exercise_family.exercise_list) for kc in self.knowledge_components]
        exercise_list = [a for tup in exercise_tuples for a in tup]
        # Corresponding ExerciseFamily
        ex_fam_list = [ex.exercise_family for ex in exercise_list]
        # Corresponding KC
        kc_list = [ex_fam.kc for ex_fam in ex_fam_list]

        df = pd.DataFrame({"exercise_family_id": [ex_fam.id for ex_fam in ex_fam_list],
                           "exercise_family_name": [ex_fam.name for ex_fam in ex_fam_list],
                           "kc_id": [kc.id for kc in kc_list],
                           "kc_type": ["procedural" if isinstance(kc, ProceduralKnowledgeComponent)
                                       else "declarative" for kc in kc_list],
                           "kc_name": [kc.name for kc in kc_list],
                           "exercise_id": [ex.id for ex in exercise_list],
                           "exercise_type": [ex.type for ex in exercise_list],
                           "exercise_content": [ex.content for ex in exercise_list],
                           "exercise_params": [ex.params for ex in exercise_list]})
        df.to_csv(csv_file, index=False)

    def from_csv(self, csv_file):
        """
        Import a DomainGraph from a csv file.
        :param csv_file: the path where the csv file is
        :return: None, only sets the DomainGraph with csvfile data
        """
        df = pd.read_csv(csv_file)
        assert {"exercise_family_id", "exercise_family_name", "kc_id", "kc_name", "exercise_id", "exercise_type",
                "exercise_content", "exercise_params"}.issubset(tuple(df.columns.values)), \
            "Missing data to construct DomainGraph."
        exercise_list, kc_list = [], []
        for idx, row in tqdm.tqdm(df.iterrows()):
            if idx == 0 or (idx >= 1 and (df.iloc[idx - 1]["kc_id"]).item() != row["kc_id"]):  # we re in the case of a new kc
                if row["kc_type"] == "procedural":
                    kc = ProceduralKnowledgeComponent(row["kc_id"], row["kc_name"])
                else:
                    kc = DeclarativeKnowledgeComponent(row["kc_id"], row["kc_name"])
                kc_list.append(kc)
                ex_fam = ExerciseFamily(row["exercise_family_id"], row["exercise_family_name"],
                                                         kc, exercise_list)
                kc.declare_associated_ex_fam(ex_fam)
            exercise = Exercise(row["exercise_id"], row["exercise_type"], row["exercise_content"], ex_fam)
            params = literal_eval(row["exercise_params"])
            exercise.set_guess(params["guess"])
            exercise.set_slip(params["slip"])
            exercise_list.append(exercise)

        self.knowledge_components = kc_list

    def get_exercise_families(self):
        return [kc.exercise_family for kc in self.knowledge_components]

    def get_knowledge_components(self):
        return self.knowledge_components
