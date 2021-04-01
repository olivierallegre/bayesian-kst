import numpy as np
import pandas as pd
from kgraph.kartable_database.credentials import db_engine
from kgraph.kartable_database.convert_tools import convert_lhc_str_to_int, convert_lhc_int_to_str


level_str_to_int = {'CE1': 6,'CE2': 7, 'CM1': 8, 'CM2': 9,
                    '6e': 10, '5e': 11, '4e': 12, '3e': 13,
                    '2nde': 14, '1ère': 15, 'Tle': 21}

level_int_to_str = {value: key for (key, value) in level_str_to_int.items()}

course_str_to_int = {'Mathématiques': 1,
                     'Philosophie': 2,
                     'Français': 3,
                     'Histoire': 4,
                     'Géographie': 5,
                     'SVT': 8,
                     'SES': 9,
                     'Anglais': 10,
                     'Espagnol': 11,
                     'Allemand': 12,
                     'Littérature': 13,
                     'Sciences': 14,
                     'ECJS': 15,
                     'Musique': 16,
                     "Histoire de l'art": 17,
                     "Sciences de l'ingénieur": 18,
                     'PFEG': 19,
                     'Physique-chimie': 21,
                     'EMC': 22,
                     'Enseigement scientifique': 30,
                     'Géopolitique': 31,
                     'Humanités': 32,
                     'LLCA': 33,
                     'LLCE Anglais': 34,
                     'NSI': 35,
                     'Arts': 36,
                     'Questionner le monde': 37,
                     'LLCE Espagnol': 38,
                     'LLCE Allemand': 39,
                     'LLCE Italien': 40}

course_int_to_str = {value: key for (key, value) in course_str_to_int.items()}

schoolyear_str_to_int = {'2019-2020': 162,
                         '2020-2021': 166}

schoolyear_int_to_str = {value: key for (key, value) in schoolyear_str_to_int.items()}


def print_lhc(lhc_id):
    req = f'SELECT C.shortLabel, L.fullLabel, SY.name FROM LevelHasCourse LHC ' \
          f'LEFT JOIN Course C on LHC.course_id = C.id ' \
          f'LEFT JOIN Level L on LHC.level_id = L.id ' \
          f'LEFT JOIN SchoolYear SY on LHC.schoolYear_id = SY.id ' \
          f'WHERE LHC.id = {lhc_id}'
    df = pd.read_sql(req, db_engine)
    print(df.iloc[0]["shortLabel"], df.iloc[0]["fullLabel"], df.iloc[0]["name"])


def process_perimeter(perimeter):
    if isinstance(perimeter, dict):
        req = "WHERE " if any(perimeter.keys()) else ""
        if not req:
            return False
        if 'level' in perimeter.keys():
            req += f"LHC.level_id = {level_str_to_int[perimeter['level']]} "
        if 'course' in perimeter.keys():
            req += f"LHC.course_id = {course_str_to_int[perimeter['course']]} "
        if 'schoolyear' in perimeter.keys():
            req += f"LHC.schoolyear_id = {schoolyear_str_to_int[perimeter['schoolyear']]} "
        if 'chapter' in perimeter.keys():
            req += f"chapter.name = {perimeter['chapter']} "
        print(req)
        return req
    elif isinstance(perimeter, tuple):
        if isinstance(perimeter[0], str):
            level_id, course_id, schoolyear_id = convert_lhc_str_to_int(*perimeter)
        elif isinstance(perimeter[0], np.integer):
            (level_id, course_id, schoolyear_id) = perimeter
        req = f"SELECT LHC.id " \
              f"FROM LevelHasCourse LHC " \
              f"WHERE LHC.level_id = {level_id} " \
              f"AND LHC.course_id = {course_id} " \
              f"AND LHC.schoolyear_id = {schoolyear_id}"
        return pd.read_sql(req, db_engine).iloc[0]["id"]
    elif isinstance(perimeter, np.integer):
        # TODO: check if the given int is really a lhc id
        return perimeter
    elif isinstance(perimeter, str):
        str_el = perimeter.split()
    else:
        # TODO: process any given perimeter
        print("wont work")
        return 0


def get_lhc_id_for_given_schoolyear(schoolyear_name):
    """
    Get the ids of all LevelHasCourse corresponding to a given schoolyear.
    :param schoolyear_name: str, the name of the schoolyear
    :return: a list of the corresponding lhc ids
    """
    req = f'SELECT LHC.id '\
          f'FROM LevelHasCourse LHC ' \
          f'LEFT JOIN SchoolYear SY on LHC.schoolYear_id = SY.id ' \
          f'LEFT JOIN Course C on LHC.course_id = C.id ' \
          f'LEFT JOIN Level L on LHC.level_id = L.id ' \
          f'WHERE SY.name = "{schoolyear_name}" ' \
          f'AND L.shortLabel NOT IN ("6e", "5e", "4e", "3e")'
    lhc_ids = pd.read_sql(req, db_engine).dropna()["id"].to_numpy()
    return lhc_ids


