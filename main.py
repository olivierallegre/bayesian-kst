import sys

sys.path.append("../")
from kgraph.expert_layer.domain_graph import DomainGraph
from kgraph.expert_layer.knowledge_components import KnowledgeComponent
from kgraph.expert_layer.links import LinkModel, LinkFromParents
from kgraph.resources_layer.exercise_family import ExerciseFamily
from kgraph.learner_layer.evaluation import Evaluation
from elements.networks import BayesianBKTNetwork

# initiate the knowledge components and associated exercises
ex_a = ExerciseFamily(1, "Ex A")
kc_a = KnowledgeComponent(1, "A", ex_a)

ex_b = ExerciseFamily(1, "Ex B")
kc_b = KnowledgeComponent(1, "B", ex_b)

ex_c = ExerciseFamily(1, "Ex C")
kc_c = KnowledgeComponent(1, "C", ex_c)

# initiate the link model of the domain graph
c_from_parents_link = LinkFromParents(kc_c, [kc_a, kc_b])
link_model = LinkModel([c_from_parents_link])

# pack infos into domain graph
domain_graph = DomainGraph([kc_a, kc_b, kc_c], link_model)

# first example - no evaluations
evaluations = []
b_network = BayesianBKTNetwork(domain_graph, evaluations)
print(b_network)
