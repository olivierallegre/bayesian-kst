import numpy as np
import pandas as pd
from kgraph.resources_layer.exercise_family import ExerciseFamily
from kgraph.learner_layer.learner import Learner
from kgraph.learner_layer.learner_pool import LearnerPool
from kgraph.learner_layer.evaluation import Evaluation
from kgraph.kartable_database.credentials import db_engine


def import_learner_from_id(learner_id, learner_pool=None):
    pass


def import_learner_pool(learner_pool_specs):
    pass


def import_learner_pool_from_domain_graph(domain_graph, eval_trigger=50, specs=False):
    """
    Import LearnerPool object from a given DomainGraph object
    :param domain_graph: DomainGraph on which the LearnerPool's learners rely
    :param eval_trigger: number of evaluations on Domain that a learner must have done to be part of the LearnerPool
    :param specs: not available now, characteristics of learners belonging to LearnerPool
    :return: LearnerPool
    """
    kd_ids = [kc.exercise_family.id for kc in domain_graph.knowledge_components]
    assert len(kd_ids) > 0, "No exercises nor knowledge component in the domain graph."
    assert isinstance(eval_trigger, (int, np.integer)), f"eval_trigger must be a integer, {type(eval_trigger)} given"
    if len(kd_ids) == 1:
        req = f"SELECT UASA.user_id, COUNT(*) " \
              f"FROM UserApplicationSessionAnswer UASA " \
              f"WHERE UASA.kartableDocument_id = {kd_ids[0]} " \
              f"GROUP BY UASA.user_id " \
              f"HAVING COUNT(UASA.id)>{eval_trigger}"
    else:
        req = f"SELECT UASA.user_id, COUNT(*) " \
              f"FROM UserApplicationSessionAnswer UASA " \
              f"WHERE UASA.kartableDocument_id IN {tuple(kd_ids)} " \
              f"GROUP BY UASA.user_id " \
              f"HAVING COUNT(UASA.id)>{eval_trigger}"
    user_ids = pd.read_sql(req, db_engine).dropna()["user_id"].to_numpy()
    assert len(user_ids) > 0, f"No users having more than {eval_trigger} evals on given domain."
    if specs:
        learner_pool = LearnerPool(domain_graph)
        for user_id in user_ids:
            learner_pool.add_learner(Learner(user_id, learner_pool))
    else:
        learner_pool = LearnerPool(domain_graph)
        for user_id in user_ids:
            learner_pool.add_learner(Learner(user_id, learner_pool))
    return learner_pool


def get_most_valuable_learner_from_learner_pool(learner_pool):
    learner_ids = learner_pool.get_learner_ids()
    ex_fam_ids = learner_pool.get_ex_fam_ids()
    req = f"SELECT UASA.user_id, COUNT(*) as c " \
          f"FROM UserApplicationSessionAnswer UASA " \
          f"WHERE UASA.user_id IN {tuple(learner_ids)} AND UASA.kartableDocument_id IN {tuple(ex_fam_ids)} " \
          f"GROUP BY UASA.user_id " \
          f"ORDER BY c DESC LIMIT 1"
    most_valuable_learner_id = pd.read_sql(req, db_engine).iloc[0]['user_id']
    return next((learner for learner in learner_pool.learners if learner.id == most_valuable_learner_id), None)


def import_evaluations_from_learner_pool(learner_pool):
    evaluations = []
    users = learner_pool.learners
    exercise_families = [kc.exercise_family for kc in learner_pool.domain_graph.knowledge_components]
    for learner in users:
        evaluations = np.concatenate((evaluations, import_evaluations_from_user_and_ex_fams(learner,
                                                                                            exercise_families)))
    return evaluations

def import_evaluations_from_user_and_ex_fams(user, ex_fams, verbose=False):
    """
    Create evaluations done by a given user on given exercise familys
    :param user: the user that has done evaluations
    :param ex_fams: the exercise_fam on which the evaluations are exported
    :return: a list of Evaluation objects
    """
    user_id = user.id
    if isinstance(ex_fams, ExerciseFamily):
        req = f'SELECT * '\
              f'FROM UserApplicationSessionAnswer AS UASA '\
              f'WHERE UASA.kartableDocument_id = {ex_fams.id} AND UASA.user_id = {user_id}'
    elif isinstance(ex_fams, (list, np.ndarray)):
        if len(ex_fams)>1:
            ex_fam_ids = np.array([ex_fam.id for ex_fam in ex_fams])
            req = f'SELECT * '\
                  f'FROM UserApplicationSessionAnswer AS UASA '\
                  f'WHERE UASA.kartableDocument_id IN {tuple(ex_fam_ids)} AND UASA.user_id = {user_id}'
        else:
            req = f'SELECT * '\
                  f'FROM UserApplicationSessionAnswer AS UASA '\
                  f'WHERE UASA.kartableDocument_id = {ex_fams[0].id} AND UASA.user_id = {user_id}'

    df = pd.read_sql(req, db_engine)  # dataframe of evals done by user user_id on kds in kd_ids
    eval_list = []
    for index, row in df.iterrows():
        eval_id = row["id"]
        ex_fam = next((ef for ef in ex_fams if ef.id == row["kartableDocument_id"]), None)
        req = f"SELECT UEA.kartableApplicationExercise_id   AS exercise_id, "\
              f"       UASA.end-UASA.start                  AS length, " \
              f"       UEA.success                          AS success "\
              f"FROM UserApplicationSessionAnswer AS UASA "\
              f"LEFT JOIN UserApplicationAnswer AS UAA ON UAA.userApplicationSessionAnswer_id = UASA.id "\
              f"LEFT JOIN UserExerciseAnswer AS UEA ON UEA.userApplicationAnswer_id = UAA.id "\
              f"WHERE UASA.id = {row['id']}"
        ans_df = pd.read_sql(req, db_engine)
        ans_dict = {next((ex for ex in ex_fam.exercise_list if ex.id == ans_row["exercise_id"]), None):
                        {"success": True if ans_row["success"] == 1 else False,
                         "length": int(ans_row["length"]) if ans_row["length"] else 0}
                    for idx, ans_row in ans_df.iterrows() if ans_row["exercise_id"]}
        if None not in ans_dict.keys():
            eval_list.append(Evaluation(eval_id, ex_fam, user, ans_dict))
        else:
            if verbose: print(f"Evaluation {row['id']} is empty:\n"
                              f"{ans_df}")
    return eval_list


