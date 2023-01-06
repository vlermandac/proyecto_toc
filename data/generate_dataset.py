import gzip
import pickle
from pathlib import Path

import ecole
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from strong_branching import ExploreThenStrongBranch
import generate_mwvc as mwvc

DATA_MAX_SAMPLES = 1000
LEARNING_RATE = 0.001
NB_EPOCHS = 50
NB_EVAL_INSTANCES = 20

# Generate mwvc model instances
node_number = 10
mwvc = mwvc.mwvc_scip(10)

# We can pass custom SCIP parameters easily
scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}

# Note how we can tuple observation functions to return complex state information
env = ecole.environment.Branching(
    observation_function=(
        ExploreThenStrongBranch(expert_probability=0.05),
        ecole.observation.NodeBipartite(),
    ),
    scip_params=scip_parameters,
)

# This will seed the environment for reproducibility
env.seed(0)

episode_counter, sample_counter = 0, 0
Path("samples/").mkdir(exist_ok=True)

# We will solve problems (run episodes) until we have saved enough samples
while sample_counter < DATA_MAX_SAMPLES:
    episode_counter += 1

    observation, action_set, _, done, _ = env.reset(mwvc.generate_model())
    while not done:
        (scores, scores_are_expert), node_observation = observation
        action = action_set[scores[action_set].argmax()]

        # Only save samples if they are coming from the expert (strong branching)
        if scores_are_expert and (sample_counter < DATA_MAX_SAMPLES):
            sample_counter += 1
            data = [node_observation, action, action_set, scores]
            filename = f"samples/sample_{sample_counter}.pkl"

            with gzip.open(filename, "wb") as f:
                pickle.dump(data, f)

        observation, action_set, _, done, _ = env.step(action)

    print(f"Episode {episode_counter}, {sample_counter} samples collected so far")


