class LearnerTrace(object):

    def __init__(self, learner, exercise, success):
        """
        Initialization of the Evaluation object.
        :param exercise_family: ExerciseFamily, exercise family which has been evaluated
        :param answers: dict, answers of the evaluation -- {exercise : {"success": bool, "length": x sec}}
        """
        self.learner = learner
        self.exercise = exercise
        assert isinstance(success, bool), "answers must be a dict {exercise: {'success': bool, 'length': x (in s)}}"
        self.success = success
        self.knowledge_component = self.exercise.get_kc()

    def get_kc(self):
        return self.knowledge_component

    def get_success(self):
        return self.success

    def get_exercise(self):
        return self.exercise

    def get_learner(self):
        return self.learner

def df_to_learner_traces(df, learner_pool):
    from kgraph.learner_layer.learner import Learner

    learner_traces = []
    for learner_id in df["user_id"].unique():
        learner = Learner(learner_id, learner_pool)
        learner_df = df[df["user_id"] == learner_id]
        for i, row in learner_df.iterrows():
            learner_traces.append(LearnerTrace(learner, ))

    return learner_traces