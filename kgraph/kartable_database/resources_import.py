import numpy as np
import pandas as pd
import math
import tqdm
from kgraph.expert_layer.knowledge_components import KnowledgeComponent, ProceduralKnowledgeComponent, \
    DeclarativeKnowledgeComponent
from kgraph.resources_layer.exercise import Exercise
from kgraph.resources_layer.exercise_family import ExerciseFamily
from kgraph.kartable_database.credentials import db_engine
from kgraph.kartable_database.convert_tools import convert_lhc_str_to_int
from kgraph.helpers.is_str_int import check_int


def get_kd_ids_from_level_course_schoolyear(level, course, schoolyear):
    """
    Gives the ids of KartableDocuments contained in a tuple (level, course, schoolyear).
    :param level: the name or the id of the level
    :param course: the name or the id of the course
    :param schoolyear: the name or the id of the schoolyear
    :return: a list of the KartableDocuments ids
    """
    # TODO: accept ids or name for level, course and schoolyear
    if isinstance(level, int) & isinstance(level, int) & isinstance(level, int):
        level_id, course_id, schoolyear_id = level, course, schoolyear
    elif isinstance(level, str) & isinstance(level, str) & isinstance(level, str):
        level_id, course_id, schoolyear_id = convert_lhc_str_to_int(level, course, schoolyear)
    req = f'SELECT KD.id as kd_id\
    FROM LevelHasCourse AS LHC\
    LEFT JOIN Level AS L ON L.id = LHC.level_id\
    LEFT JOIN Course AS C ON LHC.course_id = C.id\
    LEFT JOIN Category AS chapter ON LHC.category_id = chapter.root\
    LEFT JOIN Category AS theme ON theme.id = chapter.parent_id\
    LEFT JOIN kartabledocument_category AS KC on chapter.id = KC.category_id\
    LEFT JOIN KartableDocument AS KD ON KD.id = KC.kartabledocument_id\
    LEFT JOIN SkillLink AS SL ON SL.document_id = KD.id\
    LEFT JOIN Skill S ON S.id = SL.skill_id\
    WHERE L.id = {level_id} AND C.id = {course_id} AND schoolYear_id = {schoolyear_id} AND S.id <> "Null"\
    AND KD.displayedOnFront = 1'
    df = pd.read_sql(req, db_engine)
    return df.dropna()["kd_id"].to_numpy()


def get_ex_fam_ids_from_chapter_name(chapter_name, level_name=None, course_name=None, schoolyear_name=None):
    """
    Requesting the ids of Kartable Documents contained in a given chapter in database.
    :param chapter_name: the name of the chapter where KD are supposed to be
    :param level_name: additional, the name of the level where the chapter is (e.g. "Première")
    :param course_name: additional, the name of the course where the chapter is (e.g. "Mathématiques")
    :param schoolyear_name: additional, the name of the schoolyear where the chapter is (e.g. "2020-2021")
    :return: a dataframe containing the ids of KartableDocuments contained in the chapter "chapter_name"
    """
    # Checkin if the chapter name is unique: otherwise, we ask for additional infos
    req = f'SELECT chapter.id\
    FROM LevelHasCourse AS LHC\
    LEFT JOIN Category AS chapter ON LHC.category_id = chapter.root\
    WHERE chapter.fullLabel = "{chapter_name}" AND LHC.schoolyear_id = 166'
    df = pd.read_sql(req, db_engine)
    if len(df.index) > 1:
        print(f"There are several chapters named {chapter_name}. Please give additional infos:")
        level_name = input("- What is the level of the chapter?")
        course_name = input("- What is the course of the chapter?")
        schoolyear_name = input("- What is the schoolyear of the chapter?")
    # Requesting the ids of KartableDocuments related to chapter named chapter_name
    req = f'SELECT KD.id as document_id\
    FROM LevelHasCourse AS LHC\
    LEFT JOIN Level AS L ON L.id = LHC.level_id\
    LEFT JOIN Course AS C ON LHC.course_id = C.id\
    LEFT JOIN SchoolYear AS SY ON LHC.schoolYear_id = SY.id\
    LEFT JOIN Category AS chapter ON LHC.category_id = chapter.root\
    LEFT JOIN Category AS theme ON theme.id = chapter.parent_id\
    LEFT JOIN kartabledocument_category AS KC on chapter.id = KC.category_id\
    LEFT JOIN KartableDocument AS KD ON KD.id = KC.kartabledocument_id\
    LEFT JOIN SkillLink AS SL ON SL.document_id = KD.id\
    LEFT JOIN Skill S ON S.id = SL.skill_id\
    WHERE chapter.fullLabel = "{chapter_name}" AND S.id <> "Null" AND KD.displayedOnFront = 1'
    if level_name:
        req += f' AND L.fullLabel = "{level_name}"'
    if course_name:
        req += f' AND C.shortLabel = "{course_name}"'
    if schoolyear_name:
        req += f' AND SY.name = "{schoolyear_name}"'
    else:
        req += f' AND SY.name = "2020-2021"'
    id_df = pd.read_sql(req, db_engine).dropna()[
        "document_id"].to_numpy()  # we keep only the lines with a document name
    return id_df


#  IMPORTING KARTABLE DOCUMENT IN EXERCISE/EXERCISE FAMILY OBJECTS

def import_exercise_family_from_id(kd_ids):
    """
    Create an exercise family from its id in database.
    :param kd_id: the id of the KartableDocument related to the exercise family
    :param db_engine: the database engine
    :return: ExerciseFamily object (from resources_layer)
    """
    if isinstance(kd_ids, (int, np.integer)):
        # Determine the direct parameters of exercise_family
        req = f"SELECT KD.name FROM KartableDocument KD WHERE KD.id = {kd_ids}"
        df = pd.read_sql(req, db_engine)
        kd_name = df.iloc[0]["name"]
        # Determine the skill associated with exercise family
        req = f"SELECT KD.behavior, S.id, S.name FROM KartableDocument KD\
        LEFT JOIN SkillLink SL on KD.id = SL.document_id\
        LEFT JOIN Skill S on SL.skill_id = S.id\
        WHERE KD.id = {kd_ids}"
        df = pd.read_sql(req, db_engine)
        assert df.iloc[0]["behavior"] in ("exercice_technique",
                                          "exercice_de_connaissance",
                                          "probleme",
                                          "unknown"), f"KD behavior {df.iloc[0]['behavior']} not recognized."
        if df.iloc[0]["behavior"] in ("exercice_technique", "unknown"):  # Exercice technique
            # TODO: treat the 'unknown' behavior case
            kc = ProceduralKnowledgeComponent(df.iloc[0]["id"], df.iloc[0]["name"])
        elif df.iloc[0]["behavior"] == "exercice_de_connaissance":  # Exercice de connaissance
            kc = DeclarativeKnowledgeComponent(df.iloc[0]["id"], df.iloc[0]["name"])
        else:  # Problème
            kc = ProceduralKnowledgeComponent(df.iloc[0]["id"], df.iloc[0]["name"])
        # Create the exercise family object
        ex_fam = ExerciseFamily(kd_ids, kd_name, kc)
        kc.declare_associated_ex_fam(ex_fam)
        # Determine the exercises that belong to exercise family
        req = f"SELECT KAE.id, KAE.solution, BT.slug FROM KartableDocument KD\
        LEFT JOIN KartableApplication KA on KD.id = KA.kartableDocument_id\
        LEFT JOIN KartableApplicationExercise KAE on KA.id = KAE.kartableApplication_id\
        LEFT JOIN Block B on KAE.block_id = B.id\
        LEFT JOIN BlockType BT on B.type_id = BT.id\
        WHERE KD.id = {kd_ids} AND isEval=1 AND B.cmsDeletedAt is null AND KA.displayOrder=0"
        df = pd.read_sql(req, db_engine)
        exercise_list = [Exercise(row["id"], row["slug"],
                                                   row["solution"], ex_fam) for index, row in df.iterrows()]
        ex_fam.add_exercises(exercise_list)
        return ex_fam
    elif isinstance(kd_ids, list):
        return [import_exercise_family_from_id(kd_id) for kd_id in list(kd_ids)]
    else:
        print("Type of ids variable is not handled.")


def import_ex_fam_from_knowledge_component(kc):
    """
    Create ExerciseFamily object from a given KnowledgeComponent
    :param kc: int or KnowledgeComponent
    :return: ExerciseFamily
    """
    if isinstance(kc, (KnowledgeComponent,
                       DeclarativeKnowledgeComponent,
                       ProceduralKnowledgeComponent)):
        kc_id = kc.id
    elif isinstance(kc, np.integer):
        kc_id = kc
    else:
        print("Type of kc variable is not handled.")

    req = f"SELECT KD.id, KD.name FROM KartableDocument KD " \
          f"LEFT JOIN SkillLink SL on KD.id = SL.document_id " \
          f"LEFT JOIN Skill S on SL.skill_id = S.id " \
          f"WHERE S.id = {kc_id}"
    df = pd.read_sql(req, db_engine)
    assert len(df.index) == 1, f"More than one ExerciseFamily found for KnowledgeComponent #{kc_id}"

    kd_id = df.iloc[0]["id"]
    kd_name = df.iloc[0]["name"]
    ex_fam = ExerciseFamily(kd_id, kd_name, kc)

    # Determine the exercises that belong to exercise family
    req = f"SELECT KAE.id, KAE.solution, BT.slug FROM KartableDocument KD "\
          f"LEFT JOIN KartableApplication KA on KD.id = KA.kartableDocument_id "\
          f"LEFT JOIN KartableApplicationExercise KAE on KA.id = KAE.kartableApplication_id "\
          f"LEFT JOIN Block B on KAE.block_id = B.id "\
          f"LEFT JOIN BlockType BT on B.type_id = BT.id "\
          f"WHERE KD.id = {kd_id} AND isEval=1 AND B.cmsDeletedAt is null AND KA.displayOrder=0"

    df = pd.read_sql(req, db_engine)
    exercise_list = [Exercise(row["id"], row["slug"], row["solution"], ex_fam)
                     for index, row in df.iterrows()
                     if row["solution"]]
    ex_fam.add_exercises(exercise_list)
    return ex_fam


# GUESS PARAMETER COMPUTING FROM CONTENT IN DATABASE

def compute_guess_param_for_precise_answer_exercise(content):
    """
    Compute guess param for a "precise answer exercise", that is to say "question-a-completer".
    :param content: the content is in the form of {"inputs":["perd"]}
    :return: guess parameter
    """
    if isinstance(content, dict):


        possible_answers = content["inputs"]
        answer_len_mean = np.mean([len(answer) for answer in possible_answers])
    elif isinstance(content, list):
        possible_answers = content
        answer_len_mean = np.mean([len(answer) for answer in possible_answers])
    else:
        print(f'New instance spotted in {content}')
    if check_int(possible_answers[0]):
        return (1 / 10) ** answer_len_mean
    else:
        return (1 / 26) ** answer_len_mean


def compute_guess_param_for_qcm_exercise(content):
    """
    Compute the guess parameter for QCM exercise, that is to say "question-qcm-texte" and "question-qcm-image".
    :param content: the content is in the form {"choices":{"block_id#1":true,"block_id#2":false}}
    :return: guess parameter
    """
    if "choices" in content.keys():
        number_of_propositions = len(content["choices"].keys())
        number_of_answers = sum([1 if content["choices"][key] else 0 for key in content["choices"].keys()])
    else:
        number_of_propositions = len(content.keys())
        number_of_answers = sum([1 if content[key] else 0 for key in content.keys()])
    return 1 / math.comb(number_of_propositions, number_of_answers)


def compute_guess_param_for_select_exercise(content):
    """
    Compute the guess parameter for select exercise, that is to say "question-a-selectionner"
    :param content: the content is in the form {"choices":{"block_id#1":true,"block_id#2":false}}
    :return: guess parameter
    """
    assert 'choices' in content.keys(), f"Content not in the good shape, {content}"
    number_of_proposition = len(content["choices"].keys())
    number_of_answers = sum([1 if content['choices'][key] else 0 for key in content['choices'].keys()])
    return 1 / math.comb(number_of_proposition, number_of_answers)


def compute_guess_param_for_anagram_exercise(content):
    """
    Compute the guess parameter for anagram exercise, that is to say "question-anagramme"
    :param content: the content is in the form {"text":["tu","chaqueta","está","aquí","."]}
    :return: guess parameter
    """
    number_of_words = len(content["text"])
    return 1 / math.factorial(number_of_words)


def compute_guess_param_for_drag_and_drop_exercise(content):
    """
    Compute the guess parameter for drag and drop exercise, that is to say "question-glisser-deposer"
    :param content: the content is in the form {"dropzones":{"cat_id#1":[sol#1,sol#2],"cat_id#2":[sol#1,sol#2]}}
    :return: guess parameter
    """
    if isinstance(content, dict):
        if "dropzones" in content.keys():
            number_of_categories = len(content["dropzones"].keys())
            number_of_answers = sum([len(content["dropzones"][key]) for key in content["dropzones"].keys()])
        else:
            number_of_categories = len(content.keys())
            number_of_answers = sum([len(content[key]) for key in content.keys()])
    else:
        print(f'New instance spotted in {content}')
    return (1 / number_of_categories) ** number_of_answers


def compute_guess_param_for_link_exercise(content):
    """
    Compute the guess parameter for link exercise, that is to say "question-a-relier"
    :param content: the content is in the form {"links":{"3320847":3320848,"3320850":3320851,"3320853":3320854}}
    :return: guess parameter
    """
    if isinstance(content, dict):
        if "links" in content.keys():
            number_of_couples = len(content["links"].keys())
        else:
            number_of_couples = len(content.keys())
    else:
        print(f'New instance spotted in {content}')
    return 1 / math.factorial(number_of_couples)


def compute_guess_param_for_ordered_elements_exercise(content):
    """
    Compute the guess parameter for ordered elements exercise, that is to say "question-a-ordonner"
    :param content: the content is in the form {"elements":[3285855,3285856,3285857,3285884]}
    :return: guess parameter
    """
    number_of_elements = len(content["elements"])
    return 1 / math.factorial(number_of_elements)


def compute_guess_param_for_text_identification_exercise(content):
    """
    Compute the guess parameter for text identification exercise, that is to say "question-a-identifier"
    :param content: the content is in the form {"word_groups":[{"text":"La Princesse de Montpensier","solution":false},
    {"text":"La Princesse de Clèves","solution":false},...]}
    :return: guess parameter
    """
    if isinstance(content, dict):
        number_of_propositions = len(content["word_groups"])
        number_of_answers = sum([1 if el["solution"]
                                 else 0 for el in content["word_groups"]])
    elif isinstance(content, list):
        number_of_propositions = len(content)
        number_of_answers = sum([1 if el["solution"]
                                 else 0 for el in content])
    else:
        print(f'New instance spotted in {content}')
    return 1 / math.comb(number_of_propositions, number_of_answers)


def compute_guess_param_for_nested_exercise(content):
    """
    Compute the guess parameter for nested exercise
    :param content:
    :return: guess parameter
    """
    # TODO: compute guess for any nested exercise
    if "subExercises" not in content.keys():
        return compute_guess_param_for_qcm_exercise(content)
    else:
        exercises = content["subExercises"]
        guess = 1
        for subexercise in exercises:
            if subexercise["behavior"] != 10 : print("Should take attention here")
            guess = guess * compute_guess_param_for_qcm_exercise(subexercise["solution"])
        return guess