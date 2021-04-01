import numpy as np
import pandas as pd
import tqdm
from kgraph.expert_layer.domain_graph import DomainGraph
from kgraph.expert_layer.knowledge_components import ProceduralKnowledgeComponent, DeclarativeKnowledgeComponent
from kgraph.expert_layer.links import LinkFromParents, LinkFromChildren, LinkModel
from kgraph.kartable_database.perimeter import process_perimeter
from kgraph.kartable_database.credentials import db_engine, problematic_kd_ids
from kgraph.kartable_database.resources_import import import_ex_fam_from_knowledge_component, \
    import_exercise_family_from_id


def get_link_for_given_knowledge_component(knowledge_component, kc_list):
    """

    :param knowledge_component:
    :return:
    """
    # AT FIRST WE SEARCH FOR PARENTS
    req = f"SELECT * " \
          f"FROM SkillRequirement SR " \
          f"WHERE SR.skill_id = {knowledge_component.id}"
    parent_ids = pd.read_sql(req, db_engine).dropna()["requiredSkill_id"].to_numpy()
    parent_ids = [p_id for p_id in parent_ids if p_id in [kc.id for kc in kc_list]]  # we only keep kc_dict kcs
    if parent_ids:
        parents = [next((kc for kc in kc_list if kc.id == parent_id), None) for parent_id in parent_ids]
        link_from_parents = LinkFromParents(knowledge_component, parents)
    else:
        link_from_parents = None
    # THEN WE SEARCH FOR CHILDREN
    req = f"SELECT * " \
          f"FROM SkillRequirement SR " \
          f"WHERE SR.requiredSkill_id = {knowledge_component.id}"
    child_ids = pd.read_sql(req, db_engine).dropna()["skill_id"].to_numpy()
    child_ids = [c_id for c_id in child_ids if c_id in [kc.id for kc in kc_list]]  # we only keep kc_dict kcs
    if child_ids:
        children = [next((kc for kc in kc_list if kc.id == child_id), None) for child_id in child_ids]
        link_from_children = LinkFromChildren(knowledge_component, children)
    else:
        link_from_children = None
    return link_from_parents, link_from_children


def get_link_model(kc_list):
    return LinkModel([link for kc in tqdm.tqdm(kc_list) for link in get_link_for_given_knowledge_component(kc, kc_list)
                      if link])


# IMPORTING KNOWLEDGE COMPONENTS FROM DATABASE

def import_knowledge_component_from_id(kc_ids):
    """
    Create the KnowledgeComponent from their ids.
    :param kc_ids: list or int, ids of KnowledgeComponent to import
    :return: list of KnowledgeComponent
    """
    if isinstance(kc_ids, (int, np.integer)):
        req = f"SELECT Skill.id as kc_id, Skill.name as kc_name, KD.behavior " \
              f"FROM KartableDocument AS KD " \
              f"LEFT JOIN SkillLink AS SL ON SL.document_id = KD.id " \
              f"LEFT JOIN Skill ON Skill.id = SL.skill_id " \
              f"WHERE Skill.id = {kc_ids}"

        df = pd.read_sql(req, db_engine)

        assert df.iloc[0]["behavior"] in ("exercice_technique",
                                          "exercice_de_connaissance",
                                          "probleme",
                                          "unknown"), f"KD behavior {df.iloc[0]['behavior']} not recognized."
        if df.iloc[0]["behavior"] in ("exercice_technique", "unknown"):  # Exercice technique
            # TODO: treat the 'unknown' behavior case
            kc = ProceduralKnowledgeComponent(df.iloc[0]["kc_id"], df.iloc[0]["kc_name"])
        elif df.iloc[0]["behavior"] == "exercice_de_connaissance":  # Exercice de connaissance
            kc = DeclarativeKnowledgeComponent(df.iloc[0]["kc_id"], df.iloc[0]["kc_name"])
        else:  # Problème
            kc = ProceduralKnowledgeComponent(df.iloc[0]["kc_id"], df.iloc[0]["kc_name"])

        kc.declare_associated_ex_fam(import_ex_fam_from_knowledge_component(kc))
        return kc
    elif isinstance(kc_ids, (list, np.ndarray)):
        kc_list = [import_knowledge_component_from_id(kc_id) for kc_id in tqdm.tqdm(kc_ids)]
        return kc_list
    else:
        print("Type of ids variable is not handled.")


def import_domain_graph_from_perimeter(perimeter):
    """
    Create a DomainGraph covering the given perimeter
    :param perimeter: perimeter
    :return: DomainGraph
    """
    if {'level', 'course', 'schoolyear'}.issubset(perimeter.keys()):
        lhc_id = process_perimeter((perimeter['level'], perimeter['course'], perimeter['schoolyear']))
    else:
        lhc_id = process_perimeter(perimeter)

    chapter_name = perimeter['chapter'] if 'chapter' in perimeter.keys() else None

    req = f'SELECT S.id as kc_id ' \
          f'FROM LevelHasCourse AS LHC ' \
          f'LEFT JOIN Category AS chapter ON LHC.category_id = chapter.root ' \
          f'LEFT JOIN Category AS theme ON theme.id = chapter.parent_id ' \
          f'LEFT JOIN kartabledocument_category AS KC on chapter.id = KC.category_id ' \
          f'LEFT JOIN KartableDocument AS KD ON KD.id = KC.kartabledocument_id ' \
          f'LEFT JOIN SkillLink AS SL ON SL.document_id = KD.id ' \
          f'LEFT JOIN Skill S ON S.id = SL.skill_id ' \
          f"WHERE LHC.id = {lhc_id} AND S.id <> 'NULL' " \
          f'AND KD.type_id<>4 AND KD.displayedOnFront = 1 AND KD.id not in {tuple(problematic_kd_ids)}'
    if chapter_name:
        req += f' AND chapter.fullLabel = "{chapter_name}"'
    kc_ids = pd.read_sql(req, db_engine).dropna()["kc_id"].to_numpy()
    knowledge_components = import_knowledge_component_from_id(kc_ids)
    link_model = get_link_model(knowledge_components)
    domain_graph = DomainGraph(knowledge_components, link_model)
    return domain_graph


def import_domain_graph_from_knowledge_component_ids(knowledge_component_ids):
    """
    Create a DomainGraph covering the given knowledge component ids
    :param knowledge_component_ids: knowledge_component_ids
    :return: DomainGraph
    """
    knowledge_components = import_knowledge_component_from_id(knowledge_component_ids)
    link_model = get_link_model(knowledge_components)
    domain_graph = DomainGraph(knowledge_components, link_model)
    # TODO: import prerequisite links
    return domain_graph


def import_domain_graph_from_exercise_family_ids(exercise_family_ids):
    """
    Create a DomainGraph covering the given exercise family ids
    :param exercise_family_ids: exercise_family_ids
    :return: DomainGraph
    """
    exercise_families = import_exercise_family_from_id(exercise_family_ids)
    knowledge_components = [ex_fam.kc for ex_fam in exercise_families]
    link_model = get_link_model(knowledge_components)
    domain_graph = DomainGraph(knowledge_components, link_model)
    # TODO: import prerequisite links
    return domain_graph


