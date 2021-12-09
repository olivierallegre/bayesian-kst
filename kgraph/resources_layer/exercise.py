import json
from ast import literal_eval


class Exercise:

    def __init__(self, ex_id, knowledge_component, ex_type=None, ex_content=None, params=None):
        """
        Initialization of Exercise object
        :param ex_id: int, exercise id
        :param ex_content: str, exercise content
        :param params: dict, exercise parameters -- keys must belong to [learn, guess, slip, delta, gamma]
        """
        self.id = ex_id
        self.set_kc(knowledge_component)

        if not ex_content:
            self.content = "Empty"
        elif ex_content[1] == "'":
            self.content = literal_eval(ex_content)
        else:
            self.content = json.loads(ex_content)
        assert self.content, print(f"Exercise #{self.id} content is empty.")
        self.type = ex_type
        self.params = {}
        #self._initialize_guess_param()
        #self._initialize_slip_param()
        if params is not None:
            if "guess" in params.keys():
                self.params["guess"] = params["guess"]
            if "slip" in params.keys():
                self.params["slip"] = params["slip"]


    def _initialize_guess_param(self, verbose=False):
        """
        Internal method that initialize the guess parameter in function of the self type.
        :param verbose: bool, tells if a print is expected
        :return: None, only changes the value of the guess paramater
        """

        from kgraph.kartable_database.resources_import import (
            compute_guess_param_for_qcm_exercise,
            compute_guess_param_for_anagram_exercise,
            compute_guess_param_for_drag_and_drop_exercise,
            compute_guess_param_for_link_exercise,
            compute_guess_param_for_nested_exercise,
            compute_guess_param_for_ordered_elements_exercise,
            compute_guess_param_for_precise_answer_exercise,
            compute_guess_param_for_select_exercise,
            compute_guess_param_for_text_identification_exercise
        )

        if self.type == 'question-a-completer':
            guess = compute_guess_param_for_precise_answer_exercise(self.content)
        elif self.type in ('question-qcm-texte', 'question-qcm-image', 'question-qcm-quiz'):
            guess = compute_guess_param_for_qcm_exercise(self.content)
        elif self.type == 'question-a-selectionner':
            guess = compute_guess_param_for_select_exercise(self.content)
        elif self.type == 'question-anagramme':
            guess = compute_guess_param_for_anagram_exercise(self.content)
        elif self.type == 'question-glisser-deposer':
            guess = compute_guess_param_for_drag_and_drop_exercise(self.content)
        elif self.type == 'question-a-relier':
            guess = compute_guess_param_for_link_exercise(self.content)
        elif self.type == 'question-a-ordonner':
            guess = compute_guess_param_for_ordered_elements_exercise(self.content)
        elif self.type == 'question-a-identifier':
            guess = compute_guess_param_for_text_identification_exercise(self.content)
        elif self.type in ('question-qcm-texte-avec-sous-questions', 'question-qcm-image-avec-sous-questions'):
            guess = compute_guess_param_for_nested_exercise(self.content)
        else:
            if verbose : print(f"Unknown type {self.type} for KAE {self.id}")
            guess = .25
        self.params["guess"] = max(guess, .1)

    def _initialize_slip_param(self):
        """
        Internal method that initialize the guess parameter in function of the self type.
        :param verbose: bool, tells if a print is expected
        :return: None, only changes the value of the guess paramater
        """
        # TODO: complexify the slip param
        self.params["slip"] = .1

    def set_slip(self, slip):
        """
        Set the value of the slip parameter with a given value.
        :param slip: the wanted value for slip parameter
        :return: None, only sets the value to param value
        """
        self.params['slip'] = slip

    def get_slip(self):
        """
        Get the value of the slip parameter of self Exercise.
        :return: slip parameter of self
        """
        return self.params['slip']

    def set_guess(self, guess):
        """
        Set the value of the guess parameter with a given value.
        :param guess: the wanted value for guess parameter
        :return: None, only sets the value to param value
        """
        self.params['guess'] = guess

    def get_guess(self):
        """
        Get the value of the guess parameter of self Exercise.
        :return: guess parameter of self
        """
        return self.params['guess']

    def set_kc(self, kc):
        from kgraph.expert_layer.knowledge_components import KnowledgeComponent
        assert isinstance(kc, KnowledgeComponent), "KnowledgeComponent object expected."
        self.knowledge_component = kc
        if self not in self.knowledge_component.get_exercises():
            self.knowledge_component.add_associated_exercise(self)

    def get_kc(self):
        return self.knowledge_component
