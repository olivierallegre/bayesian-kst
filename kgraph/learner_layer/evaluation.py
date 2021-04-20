import numpy as np
from kgraph.learner_layer.learner import Learner


class Evaluation(object):

    def __init__(self, evaluation_id, exercise_family, learner: Learner, answers):
        """
        Initialization of the Evaluation object.
        :param exercise_family: ExerciseFamily, exercise family which has been evaluated
        :param answers: dict, answers of the evaluation -- {"exercise" : {"success": bool, "length": x sec}}
        """
        self.id = evaluation_id
        self.exercise_family = exercise_family
        self.learner = learner
        assert isinstance(answers, dict), "answers must be a dict {exercise: {'success': bool, 'length': x (in s)}}"
        if any([isinstance(key, np.integer) for key in answers.keys()]):
            answers = {exercise_family.get_exercise_from_id(key): answers[key] for key in answers.keys()}
        self.answers = answers

    def __str__(self):
        string = f"Evaluation #{self.id} on ExerciseFamily #{self.exercise_family.id}: " \
                 f"{self.get_number_of_right_answers()}/{len(self.answers.keys())}"
        return string

    def __eq__(self, other):
        if self.id == other.id:
            if self.exercise_family == other.exercise_family:
                if self.learner == other.learner:
                    if self.answers == other.answers:
                        return True
        return False

    def get_exercise_ids(self):
        return [exercise.id for exercise in self.exercise_family.exercise_list]

    def get_kc_id(self):
        return self.exercise_family.kc.id

    def get_kc(self):
        return self.exercise_family.kc

    def get_results(self):
        return [self.answers[key]['success'] for key in self.answers.keys()]

    def get_number_of_right_answers(self):
        return len([res for res in self.get_results() if res])

    def process(self):
        self.learner.process_evaluation(self)

    def get_success_probability(self):
        return 0.7
