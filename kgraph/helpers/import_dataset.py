import pandas as pd
import numpy as np
from kgraph.expert_layer.knowledge_components import KnowledgeComponent
from kgraph.expert_layer.domain import Domain
from kgraph.learner_layer.learner import Learner
from kgraph.learner_layer.evaluation import LearnerTrace
from kgraph.resources_layer.exercise import Exercise


def setup_domain_and_resources_from_dataset(dataset, defaults=None):
    assert any((isinstance(dataset, str), isinstance(dataset, pd.DataFrame))), "dataset must be str or dataframe"
    df = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)

    if defaults is None:
        defaults = {'learner_id': 'user_id', 'kc_id': 'kd_id', 'exercise_id': 'kae_id'}
    else:
        assert all([x in defaults.keys() for x in ('learner_id', 'kc_id', 'exercise_id')]), \
            'Missing column equivalence.'
    knowledge_components, exercises = [], []
    for i, row in df.iterrows():
        kc_id = row[defaults['kc_id']]
        if kc_id not in [kc.id for kc in knowledge_components]:
            kc_name = row[defaults['kc_name']] if 'kc_name' in defaults else kc_id
            kc = KnowledgeComponent(kc_id, kc_name)
            knowledge_components.append(kc)
        else:
            kc = [kc for kc in knowledge_components if kc.id == kc_id][0]
        exercise_id = row[defaults['exercise_id']]
        if exercise_id not in [exercise.id for exercise in exercises]:
            exercise = Exercise(exercise_id, kc)
            exercises.append(exercise)
    domain = Domain(knowledge_components)
    return domain, exercises


def deduce_learner_traces_from_dataset(dataset, exercises, learner_pool, defaults=None):
    """
    The traces are sorted by learner id
    """
    assert any((isinstance(dataset, str), isinstance(dataset, pd.DataFrame))), "dataset must be str or dataframe"
    df = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)

    if defaults is None:
        defaults = {'learner_id': 'user_id', 'kc_id': 'kd_id', 'exercise_id': 'kae_id', 'success': 'uae_success'}
    else:
        assert all([x in defaults.keys() for x in ('learner_id', 'kc_id', 'exercise_id', 'success')]), \
            'Missing column equivalence.'

    learners = [Learner(learner_id, learner_pool) for learner_id in np.unique(dataset[defaults['learner_id']])]
    learner_traces = {}
    for learner in learners:
        learner_df = df[df[defaults['learner_id']] == learner.id]
        learner_traces[learner] = []
        for i, row in learner_df.iterrows():
            exercise = next((x for x in exercises if x.id == row[defaults['exercise_id']]), None)
            if exercise is None:
                return Exception('exercise is none')
            success = bool(row[defaults['success']])
            trace = LearnerTrace(learner, exercise, success)
            learner_traces[learner].append(trace)
    return learner_traces

