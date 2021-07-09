# Bayesian KST

This work is part of my PhD, done in MOCAH team, LIP6, Sorbonne Université. My PhD is founded by Kartable and all 
algorithms in this repo are tested on their data, which is not freely available. 

This repo consists in a framework for cognitive diagnosis on multiple knowledge components associated with each other 
throught prerequisite links. The diagnosis is done via a dynamic bayesian network from pyAgrum library. It is inspired
from Knowledge Space Theory (Doignon et Falmagne) and from Bayesian Knowledge Tracing to provide dynamic multiskill 
knowledge infering for learners.

## kgraph package 

The package kgraph contains all the architecture of our project, decomposed in three parts:
- the **expert layer** that contains all the objects related to the expert, that is to say the knowledge components 
that composes the domain and the prerequisite links that might exist between them
- the **learner layer** that contains all the objects related to the learner, that is to say the learner itself, the 
learner pool in which she belongs and her answers.
- the **resources layer** that contains all the objects related to resources, such as the exercises and their compounds.

The three layers are highly entangled as they rely on each other. For example, the learner's answer (learner layer)
rely on an exercise (resources layer) that is based on a knowledge component (expert layer).

## notebooks
### comparison_between_bn_structures folder
In this folder, one may find notebooks that compares the different structure we propose for the learner knowledge 
inference. We try to understand how the way bayesian networks are constructed influences the predictions for learners' 
answers. We also compares these multiskill structure with famous BKT to spot the differences. 

### find_prereq_link_strength folder
In this folder, we try to learn the prerequisite links' strength from data from learner traces of a set of learner 
(that are for now supposed to have similar learner profile).
