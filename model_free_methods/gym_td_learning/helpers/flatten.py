import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Tuple, Dict, Box
import itertools
from collections import OrderedDict

def get_all_possible_actions(space):
    """
    Returns a list of all possible actions for discrete spaces.
    Returns None for continuous spaces.
    """
    if isinstance(space, Discrete):
        return list(range(space.n))

    elif isinstance(space, MultiDiscrete):
        action_ranges = [range(n) for n in space.nvec]
        return [list(action) for action in itertools.product(*action_ranges)]

    elif isinstance(space, Tuple):
        # Recursively get actions for each subspace in the tuple
        sub_actions = [get_all_possible_actions(s) for s in space.spaces]
        # If any subspace is continuous, the whole space is not listable
        if any(s is None for s in sub_actions):
            return None
        return list(itertools.product(*sub_actions))

    elif isinstance(space, Dict):
        # Recursively get actions for each subspace in the dict
        sub_actions = {key: get_all_possible_actions(s) for key, s in space.spaces.items()}
        # If any subspace is continuous, the whole space is not listable
        if any(v is None for v in sub_actions.values()):
            return None
        
        # Get keys and the lists of their possible values
        keys = sub_actions.keys()
        vals = sub_actions.values()
        
        # Create all combinations of values
        all_value_combinations = list(itertools.product(*vals))
        
        # Reconstruct the dictionaries
        return [dict(zip(keys, combo)) for combo in all_value_combinations]

    elif isinstance(space, Box):
        # Continuous space, cannot list all actions
        print(f"Warning: Action space is Box (continuous). Cannot list all actions.")
        return None
        
    else:
        raise NotImplementedError(f"Action space of type {type(space)} is not supported.")

# --- DEMONSTRATION ---
if __name__ == '__main__':
    print("--- 1. Discrete ---")
    env1 = gym.make("FrozenLake-v1")
    print(f"FrozenLake actions: {get_all_possible_actions(env1.action_space)}\n")
    
    print("--- 2. MultiDiscrete ---")
    space2 = MultiDiscrete([2, 3])
    print(f"MultiDiscrete([2, 3]) actions: {get_all_possible_actions(space2)}\n")
    
    print("--- 3. Tuple of Discrete ---")
    space3 = Tuple((Discrete(2), Discrete(3)))
    print(f"Tuple(Discrete(2), Discrete(3)) actions: {get_all_possible_actions(space3)}\n")
    
    print("--- 4. Dict of Discrete ---")
    space4 = Dict({"position": Discrete(3), "velocity": Discrete(2)})
    print(f"Dict actions: {get_all_possible_actions(space4)}\n")

    print("--- 5. Continuous (Box) ---")
    env5 = gym.make("BipedalWalker-v3")
    print(f"BipedalWalker actions: {get_all_possible_actions(env5.action_space)}\n")
    
    print("--- 6. Composite with Continuous ---")
    space6 = Tuple((Discrete(2), Box(low=0, high=1, shape=(1,))))
    print(f"Tuple with Box actions: {get_all_possible_actions(space6)}")