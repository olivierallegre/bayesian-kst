import random
import math

def get_random_integers_list_summing_to_given_integer(n_int, sum_val):
    # n random floats
    result = [0 for _ in range(n_int)]
    for i in range(sum_val):
        rand_idx = random.randint(0, n_int-1)
        result[rand_idx] += 1
    return result
