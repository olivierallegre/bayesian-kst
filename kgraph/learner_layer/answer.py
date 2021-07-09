import numpy as np


class LearnerAnswer(object):

    def __init__(self, answer_id, learner, exercise, success):
        """
        Initialization of the Evaluation object.
        :param answer_id: int, id of the answer
        :param learner: Learner, the learner that answers the exercise
        :param exercise: Exercise, the exercise on which the answer is given
        :param success: bool, boolean success on the answer
        """
        self.id = answer_id
        self.learner = learner
        self.exercise = exercise
        self.success = success

    def get_kc(self):
        return self.exercise.kc

