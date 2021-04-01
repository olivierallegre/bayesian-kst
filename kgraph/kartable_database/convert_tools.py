import pandas as pd
from kgraph.kartable_database.credentials import db_engine


def convert_lhc_str_to_int(level_name, course_name, schoolyear_name):
    """
    Convert the LHC names into LHC ids
    :param level_name: name of the level
    :param course_name: name of the course
    :param schoolyear_name: name of the schoolyear
    :return: level_id, course_id, schoolyear_id
    """
    # COURSE ID determination
    req = f'SELECT L.id FROM Level L WHERE L.fullLabel = "{level_name}"'
    df = pd.read_sql(req, db_engine)
    level_id = df.iloc[0]["id"]

    # LEVEL ID determination
    req = f'SELECT C.id FROM Course C WHERE C.shortLabel = "{course_name}"'
    df = pd.read_sql(req, db_engine)
    course_id = df.iloc[0]["id"]

    # SCHOOLYEAR ID determination
    req = f'SELECT S.id FROM SchoolYear S WHERE S.name = "{schoolyear_name}"'
    df = pd.read_sql(req, db_engine)
    schoolyear_id = df.iloc[0]["id"]

    return level_id, course_id, schoolyear_id


def convert_lhc_int_to_str(level_id, course_id, schoolyear_id):
    """
    Convert the LHC ids into LHC names
    :param level_id: id of the level
    :param course_id: id of the course
    :param schoolyear_id: id of the schoolyear
    :return: level name, course name, schoolyear name
    """
    # COURSE ID determination
    req = f'SELECT L.fullLabel FROM Level L WHERE L.id = "{level_id}"'
    df = pd.read_sql(req, db_engine)
    level_name = df.iloc[0]["fullLabel"]

    # LEVEL ID determination
    req = f'SELECT C.shortLabel FROM Course C WHERE C.id = "{course_id}"'
    df = pd.read_sql(req, db_engine)
    course_name = df.iloc[0]["shortLabel"]

    # SCHOOLYEAR ID determination
    req = f'SELECT S.name FROM SchoolYear S WHERE S.id = "{schoolyear_id}"'
    df = pd.read_sql(req, db_engine)
    schoolyear_name = df.iloc[0]["name"]

    return level_name, course_name, schoolyear_name
