import numpy as np


def update_mastering_probability_for_procedural_knowledge_component(initial_m_pba, answer, params):
    """
    Method to update probability to master self from the result of an exercise about self.
    :param initial_m_pba: initial mastering probability of the procedural knowledge component
    :param answer: dict that contains the result and content of an answer to an exercise
    :param params: params of the exercise -- must contain learn, guess and slip params (ref. kgraph content)
    :return: update the value of the mastering probability of self
    """
    learn, guess, slip = params['learn'], params['guess'], params['slip']
    if isinstance(answer, bool):
        success = answer
    else:
        success = answer["success"]
    if success:
        m_pba = (initial_m_pba * (1 - slip) / (initial_m_pba * (1 - slip) + (1 - initial_m_pba) * guess))
    else:
        m_pba = (initial_m_pba * slip) / (initial_m_pba * slip + (1 - initial_m_pba) * (1 - guess))
    return m_pba + (1 - m_pba) * learn


def update_mastering_probability_for_declarative_knowledge_component(initial_m_pba, answer, params):
    """
    Method to update probability to master self from the result of an exercise about self.
    :param initial_m_pba: value of the mastering probability before evaluation
    :param answer: dict that contains the result and content of an answer to an exercise
    :param params: params of the exercise -- must contain guess, delta and gamma params (ref. kgraph content)
    :return: update the value of the mastering probability of self
    """
    guess, delta, gamma = params['guess'], params['delta'], params['gamma']
    if isinstance(answer, bool):
        success = answer
    else:
        success = answer["success"]
    assert isinstance(success, bool), "Success param of answer must be a boolean."

    def logistic_function(x):
        return 1 / (1 + np.exp(-x))

    theta = np.log(initial_m_pba / (1 - initial_m_pba))
    theta += success * gamma * (1 - (guess + (1 - guess) * logistic_function(theta))) \
             + (1 - success) * delta * (guess + (1 - guess) * logistic_function(theta))

    return logistic_function(theta)


class KnowledgeComponent(object):

    def __init__(self, kc_id: int, kc_name: str, ex_fam=None):
        """
        Constructor of the KnowledgeComponent class
        :param kc_id: the id of the KnowledgeComponent
        :param kc_name: the name of the KnowledgeComponent
        :param ex_fam: the ExerciseFamily associated to self
        """
        self.id = kc_id
        self.name = kc_name
        self.declare_associated_ex_fam(ex_fam)
        self.behavior = None

    def declare_associated_ex_fam(self, ex_fam):
        self.exercise_family = ex_fam
        if self.exercise_family.kc is not self:
            self.exercise_family.declare_related_kc(self)

    def update_mastering_probability(self, initial_m_pba, answer, params):
        if self.behavior == 'declarative':
            return update_mastering_probability_for_declarative_knowledge_component(initial_m_pba,
                                                                                    answer,
                                                                                    params)
        elif self.behavior == 'procedural':
            return update_mastering_probability_for_procedural_knowledge_component(initial_m_pba,
                                                                                   answer,
                                                                                   params)
        else:
            behavior = input("No behavior informed: please type 'declarative' or 'procedural'.")
            assert behavior in ('declarative', 'procedural'), "Given behavior unknown."
            if behavior == 'declarative':
                self = DeclarativeKnowledgeComponent()
            else:
                self = ProceduralKnowledgeComponent()
            self.update_mastering_probability()

    def get_exercise_family(self):
        return self.exercise_family


class DeclarativeKnowledgeComponent(KnowledgeComponent):

    def __init__(self, kc_id, kc_name, ex_fam=None):
        """
        Constructor of the KnowledgeComponent class
        :param kc_id: the id of the KnowledgeComponent
        :param kc_name: the name of the KnowledgeComponent
        :param ex_fam: the ExerciseFamily associated to self
        """
        super().__init__(kc_id, kc_name, ex_fam)
        self.behavior = 'declarative'

    def __str__(self):
        string = f"Declarative KnowledgeComponent #{self.id}"
        return string


class ProceduralKnowledgeComponent(KnowledgeComponent):

    def __init__(self, kc_id, kc_name, ex_fam=None):
        """
        Initialization of the KnowledgeComponent class
        Inputs :
        - kc_id : id of the knowledge component
        - kc_name : name of the knowledge component
        :param kc_mastering_pba: the probability to master self
        """
        super().__init__(kc_id, kc_name, ex_fam)
        self.behavior = 'procedural'

    def __str__(self):
        string = f"Procedural KnowledgeComponent #{self.id}"
        return string
