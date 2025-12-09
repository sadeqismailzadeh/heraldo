"""Interactive evaluation script for a single trained agent.

This script loads a pre-trained PPO model and runs it in the QuantumCircuitEnv
for a few episodes. It provides a detailed, step-by-step log of the agent's
actions, the resulting measurements, and the rewards received.

After each episode, it calculates the final state's fidelity against the
target states and generates a Wigner function plot to visualize the
quantum state in phase space.

Usage:
    1. Edit the `MODEL_PATH` variable below to point to your trained `.zip` file.
    2. Run the script from the command line: `python evaluate_quantum.py`
    
"""



# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")


import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import *


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


from stable_baselines3 import PPO
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state
from stable_baselines3.common.vec_env import VecNormalize
from quantum_cubic_env import CubicPhaseEnv
from stable_baselines3.common.env_util import make_vec_env
# from quantum_circuit_env_mixed import QuantumCircuitEnv

import strawberryfields as sf 


def main():
    """Runs the main evaluation loop."""
    # --- Configuration ---
    # IMPORTANT: Environment parameters MUST match those used during training.
    CUTOFF_DIM = 31
    MAX_STEPS = 50
    REWARD_POWER = 2
    NUM_EPISODES = 10 # Number of episodes to run

    # --- EDIT THIS: Path to the trained model ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

    # --- Environment and Model Setup ---
    # env = CubicPhaseEnv(cutoff_dim=CUTOFF_DIM,
    #                         max_steps=MAX_STEPS,
    #                         reward_power=REWARD_POWER,
    #                         tunable_r=True,
    #                         is_loss_channel=False,
    #                         loss_channel=1)

    try:
        stats_path = MODEL_PATH.replace('.zip', '_vecnormalize.pkl')
        # vec_env = make_vec_env(env, n_envs=1)
        env = make_vec_env(
            CubicPhaseEnv,
            n_envs=1,
            env_kwargs=dict(
                cutoff_dim=CUTOFF_DIM,
                max_steps=MAX_STEPS,
                reward_power=REWARD_POWER,
                is_loss_channel=False, 
                loss_channel=1
            ),
        )
        env = VecNormalize.load(stats_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False

        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{MODEL_PATH}'")
        print("Please edit the MODEL_PATH variable in this script.")
        exit()

    # --- Run Evaluation Episodes ---
    for episode in range(NUM_EPISODES):
        print(f"\n{'='*20} Starting Evaluation Episode {episode + 1} {'='*20}")

        obs = env.reset()
        done = False
        total_reward = 0
        step  = 1
        while not done:
            # Use the deterministic policy for evaluation
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            done = dones[0]

            denorm_action = env.env_method('_denormalize_action', actions[0], indices=[0])[0]

            measured_n = infos[0].get('detected_photons', 'N/A')
            photon_loss = infos[0].get('photon_loss', 'N/A')
            fidelity = infos[0].get('fidelity', 'N/A')
            print(
                f"Step {step:2d}: "
                f"Action=[squeezing_r={denorm_action[0]:.4f},tau_1={np.cos(denorm_action[1]):.4f}], "
                f"Measured_n={measured_n}, "
                f"photon_loss={photon_loss}, "
                f"fidelity={fidelity:.4f}"
            )
            step += 1
            total_reward += rewards[0]

        # --- Post-Episode Analysis and Visualization ---
        print(f"\n--- Episode {episode + 1} Finished ---")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.4f}")

        # Get final fidelity from the last info
        final_fidelity = infos[0].get('fidelity', 0.0)
        print(f"Final Fidelity: {final_fidelity:.4f}")

        # Generate and display the Wigner function plot
        final_ket = infos[0].get('final_ket', 'N/A')

        prog = sf.Program(1)
        with prog.context as q:
            ops.Ket(final_ket) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": CUTOFF_DIM})
        result = eng.run(prog)
        state = result.state
        xvec = np.linspace(-6, 6, 200)
        W = state.wigner(mode=0, xvec=xvec, pvec=xvec)

        scale = np.max(np.abs(W))
        nrm = mpl.colors.Normalize(-scale, scale)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal")
        ax.contourf(xvec, xvec, W, 60, cmap=cm.RdBu, norm=nrm)

        fig.colorbar(cm.ScalarMappable(norm=nrm, cmap=cm.RdBu), ax=ax)
        plt.title(f"Episode {episode+1} - Final State Wigner Function\nFinal Fidelity: {final_fidelity:.4f}", fontsize=14)
        plt.xlabel("q", fontsize=12)
        plt.ylabel("p", fontsize=12)
        plt.grid(True, linestyle='--')
        plt.show()

    env.close()

if __name__ == '__main__':
    main()