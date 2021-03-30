# %%

import random
import sys
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn

sys.path.append('../')

import thesis.expert_layer as expert_layer
import thesis.ressources_layer as ressources_layer
import thesis.learner_layer as learner_layer

# Define kc as KC classes
kc_a = expert_layer.ProceduralKnowledgeComponent(1, "A")
kc_b = expert_layer.ProceduralKnowledgeComponent(2, "B")
kc_c = expert_layer.ProceduralKnowledgeComponent(3, "C")
kc_d = expert_layer.ProceduralKnowledgeComponent(4, "D")

domain_graph = expert_layer.DomainGraph([kc_a, kc_b, kc_c, kc_d])

# Define parents of the DomainGraph
kc_b.declare_parents([kc_a])
kc_c.declare_parents([kc_a])
kc_d.declare_parents([kc_b, kc_c])

kc_a.declare_c_vec(vec=[0.4, 0.8, 0.8, 0.95])
kc_b.declare_c_vec(vec=[0.5, 0.9])
kc_c.declare_c_vec(vec=[0.5, 0.9])

kc_b.declare_p_vec(vec=[0.01, 0.3])
kc_c.declare_p_vec(vec=[0.01, 0.3])
kc_d.declare_p_vec(vec=[0.01, 0.02, 0.02, 0.3])

# INITIALIZATION
# Define exercises as Exercises classes
ex_fam_a = ressources_layer.ExerciseFamily(51738, "ExerciseFamily A", kc_a)
ex_fam_b = ressources_layer.ExerciseFamily(52068, "ExerciseFamily B", kc_b)
ex_fam_c = ressources_layer.ExerciseFamily(52068, "ExerciseFamily C", kc_c)
ex_fam_d = ressources_layer.ExerciseFamily(52068, "ExerciseFamily D", kc_d)


def create_default_exercise_family(n):
    ex_fam = []
    for i in range(n):
        ex_fam.append(ressources_layer.Exercise(0, "default", {'slip': 0.1, 'guess': 0.25}))
    return ex_fam


default_ex_fam = create_default_exercise_family(5)

ex_fam_pool = [ex_fam_a, ex_fam_b, ex_fam_c, ex_fam_d]
for ex_fam in ex_fam_pool:
    ex_fam.add_exercises(default_ex_fam)

# Define the student as a Learner
stud_pool = learner_layer.LearnerPool(domain_graph)
stud = learner_layer.Learner(1, stud_pool)


def create_rand_answers_from_exercise_family(ex_fam):
    return {ex: {"success": bool(random.getrandbits(1))} for ex in ex_fam}


# EVALUATION
# Defining evaluations as Evaluation objects
evala = learner_layer.Evaluation(ex_fam_a, create_rand_answers_from_exercise_family(default_ex_fam))
evalb = learner_layer.Evaluation(ex_fam_b, create_rand_answers_from_exercise_family(default_ex_fam))
evalc = learner_layer.Evaluation(ex_fam_c, create_rand_answers_from_exercise_family(default_ex_fam))
evald = learner_layer.Evaluation(ex_fam_d, create_rand_answers_from_exercise_family(default_ex_fam))

print("Evaluation A results: ", evala.get_results())
print("Evaluation B results: ", evalb.get_results())
print("Evaluation C results: ", evalc.get_results())
print("Evaluation D results: ", evald.get_results())


# %%

def print_bn_from_learner(learner):
    bn = get_initial_bn(learner)
    gnb.showInference(bn)


def print_bn_from_learner_with_all_observables(learner):
    bn = get_initial_bn_with_all_observables(learner)
    gnb.showInference(bn)


def get_initial_bn_with_all_observables(learner):
    bn = gum.BayesNet('LearnerGraph')
    kc_list = learner.learner_graph.kc_dict.keys()
    bn_edges = []
    # SETTING BN NODES
    for kc in kc_list:
        bn.add(gum.LabelizedVariable(f"{kc.name}t", '', 2))
        bn.add(gum.LabelizedVariable(f"{kc.name}t-1", '', 2))
        bn.addArc(*(f"{kc.name}t-1", f"{kc.name}t"))
        for child in kc.children:
            bn_edges.append((f"{child.name}t-1", f"{kc.name}t"))
    # SETTING BN ARCS
    for link in bn_edges:
        bn.addArc(*link)
    # SETTING BN ARC VALUES
    for kc in kc_list:
        m_pba = learner.learner_graph.kc_dict[kc]['m_pba']
        bn.cpt(f"{kc.name}t-1").fillWith([1 - m_pba, m_pba])
        if kc.children:
            children = kc.children
            c_vec = kc.c_vec
            truthtable = expert_layer.truthtable(len(children))
            for i in range(len(truthtable)):
                bn.cpt(f"{kc.name}t")[{**{f"{kc.name}t-1": True},
                                       **{f"{children[k].name}t-1": truthtable[i][k] \
                                          for k in range(len(children))}}] = [1 - c_vec[i], c_vec[i]]
                bn.cpt(f"{kc.name}t")[{**{f"{kc.name}t-1": False},
                                       **{f"{children[k].name}t-1": truthtable[i][k] \
                                          for k in range(len(children))}}] = [1, 0]
        else:
            bn.cpt(f"{kc.name}t")[{f"{kc.name}t-1": True}] = [0, 1]
            bn.cpt(f"{kc.name}t")[{f"{kc.name}t-1": False}] = [1, 0]
    return bn


def get_initial_bn(learner):
    bn = gum.BayesNet('LearnerGraph')
    kc_list = learner.learner_graph.kc_dict.keys()
    bn_edges = []
    # SETTING BN NODES
    for kc in kc_list:
        bn.add(gum.LabelizedVariable(f"{kc.name}t", '', 2))
        if learner.learner_graph.kc_dict[kc]['diagnosis']:
            bn.add(gum.LabelizedVariable(f"{kc.name}{0}", '', 2))
            bn.addArc(*(f"{kc.name}{0}", f"{kc.name}t"))
        for child in kc.children:
            bn_edges.append((f"{child.name}t", f"{kc.name}t"))
    # SETTING BN ARCS
    for link in bn_edges:
        bn.addArc(*link)
    # SETTING BN ARC VALUES
    for kc in kc_list:
        m_pba = learner.learner_graph.kc_dict[kc]['m_pba']
        if learner.learner_graph.kc_dict[kc]['diagnosis']:
            bn.cpt(f"{kc.name}{0}").fillWith([1 - m_pba, m_pba])
            if kc.children:
                children = kc.children
                c_vec = kc.c_vec
                truthtable = expert_layer.truthtable(len(children))
                for i in range(len(truthtable)):
                    bn.cpt(f"{kc.name}t")[{**{f"{kc.name}{0}": True},
                                           **{f"{children[k].name}t": truthtable[i][k] \
                                              for k in range(len(children))}}] = [1 - c_vec[i], c_vec[i]]
                    bn.cpt(f"{kc.name}t")[{**{f"{kc.name}{0}": False},
                                           **{f"{children[k].name}t": truthtable[i][k] \
                                              for k in range(len(children))}}] = [1, 0]
            else:
                bn.cpt(f"{kc.name}t")[{f"{kc.name}t-1": True}] = [0, 1]
                bn.cpt(f"{kc.name}t")[{f"{kc.name}t-1": False}] = [1, 0]
        else:
            if kc.children:
                children = kc.children
                c_vec = kc.c_vec
                truthtable = expert_layer.truthtable(len(children))
                for i in range(len(truthtable)):
                    bn.cpt(f"{kc.name}t")[{f"{children[k].name}t": truthtable[i][k] \
                                           for k in range(len(children))}] = [1 - c_vec[i], c_vec[i]]
            else:
                bn.cpt(f"{kc.name}t").fillWith([1 - m_pba, m_pba])
    return bn


# %%

# bn_test = get_initial_bn_with_all_observables(stud)
# for i in bn_test.nodes():
#    gnb.showPotential(bn_test.cpt(i),digits=3)

def dbn_wo_eval(stud):
    bn = gum.BayesNet('LearnerGraph')
    kc_list = learner.learner_graph.kc_dict.keys()
    bn_edges = []
    # SETTING BN NODES
    for kc in kc_list:
        bn.add(gum.LabelizedVariable(f"{kc.name}t", '', 2))
        bn.add(gum.LabelizedVariable(f"{kc.name}0", '', 2))
        bn.addArc(*(f"{kc.name}t-1", f"{kc.name}t"))
        for child in kc.children:
            bn_edges.append((f"{child.name}0", f"{kc.name}t"))
        for parent in kc.parents:
            bn_edges.append((f"{parent.name}0", f"{kc.name}t"))
    # SETTING BN ARCS
    for link in bn_edges:
        bn.addArc(*link)
    # SETTING BN ARC VALUES
    for kc in kc_list:
        m_pba = learner.learner_graph.kc_dict[kc]['m_pba']
        bn.cpt(f"{kc.name}0").fillWith([1 - m_pba, m_pba])
        if kc.children:
            children = kc.children
            c_vec = kc.c_vec
            p_vec = kc.p_vec
            truthtable = expert_layer.truthtable(len(children))
            for i in range(len(truthtable)):
                bn.cpt(f"{kc.name}t")[{**{f"{kc.name}t-1": True},
                                       **{f"{children[k].name}0": truthtable[i][k] \
                                          for k in range(len(children))}}] = [1 - c_vec[i], c_vec[i]]
                bn.cpt(f"{kc.name}t")[{**{f"{kc.name}t-1": False},
                                       **{f"{children[k].name}t-1": truthtable[i][k] \
                                          for k in range(len(children))}}] = [1, 0]
        else:
            bn.cpt(f"{kc.name}t")[{f"{kc.name}t-1": True}] = [0, 1]
            bn.cpt(f"{kc.name}t")[{f"{kc.name}t-1": False}] = [1, 0]
    return bn


print_bn_from_learner(stud)
print_bn_from_learner_with_all_observables(stud)

twodbn = get_initial_bn_with_all_observables(stud)
gdyn.showTimeSlices(twodbn)

# %%

T = 5

dbn = gdyn.unroll2TBN(twodbn, T)
gdyn.showTimeSlices(dbn, size="10")

# %%

T = 5

dbn = gdyn.unroll2TBN(twodbn, T)
gdyn.showTimeSlices(dbn, size="10")

# %%

import numpy as np
import pandas as pd


def get_bn_for_bkt_step_with_learn(learner, evaluated_kc, exercise, verbose=3):
    eps = 0  # forgetting parameter to be implemented
    bn = get_initial_bn_with_all_observables(learner)
    if verbose == 3:
        print("Graph before eval")
        gnb.showInference(bn)
    bn.add(gum.LabelizedVariable(f"E_{evaluated_kc.name}", '', 2))
    learn = learner.learner_graph.kc_dict[evaluated_kc]["params"]["learn"]
    slip, guess = exercise.get_slip(), exercise.get_guess()
    bn.addArc(*(f"{evaluated_kc.name}t-1", f"E_{evaluated_kc.name}"))
    bn.cpt(f"E_{evaluated_kc.name}")[{f"{evaluated_kc.name}t-1": True}] = [1 - ((1 - slip) * (1 - eps) + guess * learn),
                                                                           (1 - slip) * (1 - eps) + guess * learn]
    bn.cpt(f"E_{evaluated_kc.name}")[{f"{evaluated_kc.name}t-1": False}] = [
        1 - ((1 - slip) * eps + guess * (1 - learn)),
        (1 - slip) * eps + guess * (1 - learn)]
    return bn


def get_bn_for_bkt_step(learner, evaluated_kc, exercise, verbose=3):
    eps = 0  # forgetting parameter to be implemented
    bn = get_initial_bn_with_all_observables(learner)
    if verbose == 3:
        print("Graph before eval")
        gnb.showInference(bn)
    bn.add(gum.LabelizedVariable(f"E_{evaluated_kc.name}", '', 2))
    learn = learner.learner_graph.kc_dict[evaluated_kc]["params"]["learn"]
    slip, guess = exercise.get_slip(), exercise.get_guess()
    bn.addArc(*(f"{evaluated_kc.name}t-1", f"E_{evaluated_kc.name}"))
    bn.cpt(f"E_{evaluated_kc.name}")[{f"{evaluated_kc.name}t-1": True}] = [slip, 1 - slip]
    bn.cpt(f"E_{evaluated_kc.name}")[{f"{evaluated_kc.name}t-1": False}] = [1 - guess, guess]
    if evaluated_kc.children:
        children = evaluated_kc.children
        c_vec = evaluated_kc.c_vec
        truthtable = expert_layer.truthtable(len(children))
        for i in range(len(truthtable)):
            bn.cpt(f"{evaluated_kc.name}t")[{**{f"{evaluated_kc.name}t-1": True},
                                             **{f"{children[k].name}t-1": truthtable[i][k] \
                                                for k in range(len(children))}}] = [0, 1]  # [1-c_vec[i], c_vec[i]]
            bn.cpt(f"{evaluated_kc.name}t")[{**{f"{evaluated_kc.name}t-1": False},
                                             **{f"{children[k].name}t-1": truthtable[i][k] \
                                                for k in range(len(children))}}] = [1 - learn - c_vec[i],
                                                                                    learn + c_vec[i]]
    else:
        bn.cpt(f"{evaluated_kc.name}t")[{f"{evaluated_kc.name}t-1": True}] = [0, 1]
        bn.cpt(f"{evaluated_kc.name}t")[{f"{evaluated_kc.name}t-1": False}] = [1 - learn, learn]

    return bn


def new_bkt_process(learner, evaluation, verbose=3):
    kc_list = [key for key in list(learner.learner_graph.kc_dict.keys())]
    evaluated_kc = evaluation.exercise_family.kc
    learner.learner_graph.kc_dict[evaluated_kc]['diagnosis'] = True
    exercise_list = list(evaluation.answers.keys())
    df_content = []
    for exercise in exercise_list:
        print(f"Evaluation of exercice {exercise} :")
        evs = {f"E_{evaluated_kc.name}": evaluation.answers[exercise]["success"]}
        bn = get_bn_for_bkt_step(learner, evaluated_kc, exercise)
        if verbose == 3:
            print("Graph after eval")
            gnb.showInference(bn, evs=evs)
        df_line = []
        for kc in kc_list:
            pot = gum.getPosterior(bn, evs=evs, target=f"{kc.name}t")
            inst = gum.Instantiation(pot)
            m_pba = 1 - pot.get(inst)
            learner.set_mastering_probability(kc, m_pba)
            if verbose == 3: print(f"pba of {kc.name} values {learner.get_mastering_probability(kc)}")
            df_line.append(m_pba)
        df_content.append(df_line)
    return pd.DataFrame(np.array(df_content), columns=[kc.name for kc in kc_list])


# %%

df = new_bkt_process(stud, evalb)

# %%

print(df)

# %%

stud_direct_compute = learner_layer.Learner(2, stud_pool)
stud_direct_compute.has_done_a_new_evaluation(evalb)
