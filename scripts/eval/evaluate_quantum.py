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
from pathlib import Path

from quantum_agent.envs.modular_env import ModularQuantumEnv

# 2. The Specific Modules to assemble the environment
from quantum_agent.components.circuits import *
from quantum_agent.components.targets import *
from quantum_agent.components.rewards import *
from quantum_agent.envs.modular_env import fidelity_max_rotation, decode_measurement_result, fidelity_pure_state

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# from quantum_circuit_env_mixed import QuantumCircuitEnv

import strawberryfields as sf 


def plot_target_wigner(target_instance, name, cutoff_dim=35, grid_size=200, x_limit=5):
    """
    Generates the target state and plots its Wigner function and Fock probabilities.
    Adheres to the style of demo_cubic.py.
    """

    # 2. Load into Strawberry Fields
    # We use a temporary engine to utilize the built-in Wigner calculator
    prog = sf.Program(1)
    with prog.context as q:
        ops.Ket(target_instance) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    # 3. Calculate Wigner Function
    print("Calculating Wigner function...")
    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    # 4. Calculate Fock Probabilities
    probs = state.all_fock_probs(cutoff=cutoff_dim)

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Wigner Function
    # Using RdBu colormap centered at 0
    X, P = np.meshgrid(xvec, pvec)
    lim = np.max(np.abs(W))
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-lim, vmax=lim)
    fig.colorbar(c, ax=ax1, label='W(x, p)')
    ax1.set_title(f"Wigner Function ({name})")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("p (Momentum)")
    ax1.set_aspect('equal')
    
    # Add grid lines
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Fock Distribution
    # Show only up to cutoff_dim (or limit to 25 for clarity if cutoff is huge)
    display_cutoff = min(cutoff_dim, 60)
    ax2.bar(range(display_cutoff), probs[:display_cutoff], color='teal', alpha=0.7, edgecolor='black')
    ax2.set_title("Fock State Probabilities")
    ax2.set_xlabel("Fock Number |n>")
    ax2.set_ylabel("Probability")
    ax2.set_xticks(range(display_cutoff))

    plt.tight_layout()
    plt.show()


def main():
    """Runs the main evaluation loop."""
    # --- Configuration ---
    # IMPORTANT: Environment parameters MUST match those used during training.
    CUTOFF_DIM = 20
    MAX_STEPS = 50
    NUM_EPISODES = 10 # Number of episodes to run

    # --- EDIT THIS: Path to the trained model ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_modular_cat.zip")

    # 1. Circuit Context: The physical loop setup
    circuit_context = SqueezeOnlyCircuit(
        tunable_phases=True, 
        max_squeezing=1.38
    )

    circuit_context2 = ThreeModeGadgetCircuit(
        max_sq_r=1, 
        max_disp=1,
        tunable_bs_phase=False
    )

    circuit_context3 = CubicSpecificCircuit(max_sq_r=1.38)

    circuit_context4 = ThreeModeSqueezeOnlyCircuit(max_sq_r=1)
    
    circuit_2mode_gadget = GadgetCircuit(max_squeezing=1.38, max_disp=1)

    # 2. Target Generator: The state we want to reach (Squeezed Cat)
    target1 = SqueezedCatTarget(
        alpha=3, 
        r=1.38,
        p=0
    )

    target2 = SqueezedCatTarget(
        alpha=3, 
        r=1.38,
        p=1
    )
    
    gkp_targets = [CoreGKPTarget(csv_path=Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv", 
                                 n_max=n, delta_db=10.4, mu=m)
                   for n in [4, 6, 8, 10, 12] for m in [0, 1]]

    csv_path =  Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    target3=CoreGKPTarget(csv_path=csv_path, 
                          n_max=4, 
                          delta_db=10.4, 
                          mu=0)
    
    # 3. Reward Mechanism: How we calculate success
    reward_mech = LogFidelityReward()

    # add target1 and 2 to the current gkp_targets list
    # gkp_targets.append(target1)
    # gkp_targets.append(target2)

    # Generate all Binomial Codes with max Fock state <= 12
    binomial_targets = []
    max_fock_n = 12
    for S in range(1, max_fock_n):
        for N in range(1, max_fock_n):
            if (N + 1) * (S + 1) <= max_fock_n:
                binomial_targets.append(BinomialCodeTarget(N=N, S=S, mu=0))
                binomial_targets.append(BinomialCodeTarget(N=N, S=S, mu=1))
    
    gkp_targets.extend(binomial_targets)
    

    # --- Setup Parallel Environments ---
    
    # We pass the instances via env_kwargs. SubprocVecEnv will pickle them 
    # and send them to the worker processes.
    env_kwargs = dict(
        cutoff_dim=CUTOFF_DIM,
        max_steps=MAX_STEPS,
        loss_channel=1.0, # Assumes no loss for now, consistent with basic setup
        initial_target_fidelity=0.94,
        # INJECT MODULES HERE:
        circuit_context=circuit_context2,
        target_gens=gkp_targets,
        reward_mech=reward_mech
    )


    try:
        stats_path = MODEL_PATH.replace('.zip', '_vecnormalize.pkl')
        # vec_env = make_vec_env(env, n_envs=1)
        env = make_vec_env(
            ModularQuantumEnv,
            n_envs=1,
            env_kwargs=env_kwargs,
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

            # denorm_action = env.env_method('_denormalize_action', actions[0], indices=[0])[0]

            measured_n = infos[0].get('detected_photons', 'N/A')
            photon_loss = infos[0].get('photon_loss', 'N/A')
            fidelity = infos[0].get('fidelity', 'N/A')
            print(
                f"Step {step:2d}: "
                f"Action=[{', '.join([f'{k}={v:.4f} \n' for k, v in env.env_method('_denormalize_to_dict', actions[0], indices=[0])[0].items()])}], "
                # f"Action=[r={denorm_action[0]:.4f},  phi_sq={denorm_action[1]:.4f}, tau={np.cos(denorm_action[2]):.4f}, phi={(denorm_action[3]):.4f}, alpha_mag={denorm_action[4]:.4f}, alpha_phi={denorm_action[5]:.4f}]"

                # f"Action=[squeezing_r={denorm_action[0]:.4f},tau_1={np.cos(denorm_action[1]):.4f}] "
                # f"Action=[squeezing_r={denorm_action[0]:.4f},tau_1={np.cos(denorm_action[1]):.4f},alpha={(denorm_action[2]):.4f}] "
                f"Measured_n={measured_n}, "
                f"photon_loss={photon_loss}, "
                f"fidelity={fidelity:.4f} \n\n"
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
        
   
        plot_target_wigner(state.ket(), "target", cutoff_dim=CUTOFF_DIM)

    env.close()

if __name__ == '__main__':
    main()