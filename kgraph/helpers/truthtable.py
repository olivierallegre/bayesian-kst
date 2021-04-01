def truthtable(n):
    """
    Create a boolean truthtable of a given length, that's to say a list of lists of all combination of n booleans
    :param n: the number of elements to be combinated
    :return: a list of boolean lists, the expected truthtable
    """
    if n < 1:
        return [[]]
    subtable = truthtable(n - 1)
    return [row + [v] for row in subtable for v in [False, True]]


def get_representative_answers(ex_fam, mode='simple'):
    n = len(ex_fam.exercise_list)
    assert mode in ('simple', 'complex'), f"Given mode {mode} unknown"
    if mode=='complex':
        answers = truthtable(n)
    else:
        answers = [[False]*n]*(n+1)
        for idx, answer in enumerate(answers):
            answers[idx] = answer[idx:] + [True]*idx
    return answers


def bool_list_to_int(lst):
    return int('0b' + ''.join(['1' if x else '0' for x in lst]), 2) if lst else 0


def int_to_bool_list(num):
    bin_string = format(num, '#010b')
    return [x == '1' for x in bin_string]

