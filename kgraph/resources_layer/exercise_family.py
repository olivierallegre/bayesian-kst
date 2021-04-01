from kgraph.expert_layer.knowledge_components import KnowledgeComponent
import random


class ExerciseFamily(object):

    def __init__(self, ef_id, ef_name, kc=None, exercise_list=None):
        """
        Initialization of the ExerciseFamily Object
        :param ef_id: id of the exercise family
        :param ef_name: name of the exercise family
        :param kc: optional - knowledge component related to the exercise family
        :param exercise_list: exercises inside the exercise family
        """
        self.id = ef_id
        self.name = ef_name
        self.kc = kc
        if exercise_list is None:
            self.exercise_list = []
        else:
            self.exercise_list = exercise_list

    def add_exercises(self, exercises):
        """
        Method to add exercises in the exercise family.
        :param exercises: exercises to add in ExerciseFamily
        :return: self.exercise_list with exercises in it
        """
        if not isinstance(exercises, list):
            exercises = [exercises]
        if not self.exercise_list:
            self.exercise_list = exercises
        else:
            for exercise in exercises:
                if exercise not in self.exercise_list:
                    self.exercise_list.append(exercise)
        for exercise in exercises:
            exercise.exercise_family = self

    def declare_related_kc(self, kc):
        """
        Declare the knowledge component related to the exercise family
        :param kc: KnowledgeComponent object to the associated
        :return: add the knowledge component if it is a KnowledgeComponent object -- AssertionError otherwise
        """
        if isinstance(kc, KnowledgeComponent):
            self.kc = kc
        else:
            return AssertionError

    def get_number_of_exercises(self):
        """
        Method to get the number of exercises in the ExerciseFamily
        :return: the number of exercises in the ExerciseFamily
        """
        return len(self.exercise_list)

    def get_exercise_from_id(self, id):
        for exercise in self.exercise_list:
            if exercise.id == id: return exercise

    def get_random_evaluation(self, learner):
        from kgraph.learner_layer.evaluation import Evaluation
        return Evaluation(0, self, learner,
                          {ex: {"success": bool(random.getrandbits(1))} for ex in self.exercise_list})


