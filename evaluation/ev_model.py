import ecole
import torch
import numpy as np 
from proyecto_toc.data import generate_mwvc as mwvc_m
from proyecto_toc.gnn.gnn import GNNPolicy

NB_EVAL_INSTANCES = 20

scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}
env = ecole.environment.Branching(
    observation_function=ecole.observation.NodeBipartite(),
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    },
    scip_params=scip_parameters,
)
default_env = ecole.environment.Configuring(
    observation_function=None,
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    },
    scip_params=scip_parameters,
)

node_number = 150
mwvc = mwvc_m.mwvc_scip(node_number)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = GNNPolicy().to(DEVICE)

for instance_count, instance in zip(range(NB_EVAL_INSTANCES), mwvc.generate_model()):
    # Run the GNN brancher
    nb_nodes, time = 0, 0
    observation, action_set, _, done, info = env.reset(instance)
    nb_nodes += info["nb_nodes"]
    time += info["time"]
    while not done:
        with torch.no_grad():
            observation = (
                torch.from_numpy(observation.row_features.astype(np.float32)).to(DEVICE),
                torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(DEVICE),
                torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(DEVICE),
                torch.from_numpy(observation.variable_features.astype(np.float32)).to(DEVICE),
            )
            logits = policy(*observation)
            action = action_set[logits[action_set.astype(np.int64)].argmax()]
            observation, action_set, _, done, info = env.step(action)
        nb_nodes += info["nb_nodes"]
        time += info["time"]

    # Run SCIP's default brancher
    default_env.reset(instance)
    _, _, _, _, default_info = default_env.step({})

    print(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
    print(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
    print(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")