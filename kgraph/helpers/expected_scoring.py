def expected_declarative_kc_scoring(answers):
    """
    Define expected scoring for DeclarativeKnowledgeComponent for parameters' optimization.
    :param answers: list, the answers on which the scoring has to happen
    :return: the expected score for given answers
    """
    return sum(answers) / len(answers)


def expected_procedural_kc_scoring(answers):
    """
    Define expected scoring for ProceduralKnowledgeComponent for parameters' optimization.
    :param answers: list, the answers on which the scoring has to happen
    :return: the expected score for given answers
    """
    return sum([answers[j] * (.2 + .6 * j / len(answers)) for j in range(len(answers))]) \
           / sum([(.2 + .6 * j / len(answers)) for j in range(len(answers))])
