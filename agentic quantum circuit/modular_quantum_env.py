class ModularQuantumEnv(gym.Env):
    def __init__(
        self, 
        target_gen: TargetGenerator,
        circuit_context: CircuitContext,
        reward_mech: RewardMechanism,
        cutoff_dim=25,
        max_steps=10
    ):
        super().__init__()
        self.cutoff_dim = cutoff_dim
        self.max_steps = max_steps
        
        # Composition
        self.target_gen = target_gen
        self.circuit_context = circuit_context
        self.reward_mech = reward_mech
        
        # Initialize
        self.action_space = self.circuit_context.get_action_space()
        
        # Observation space (Standardized)
        obs_size = 2 * self.cutoff_dim
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)

        # Lazy load target (in case it's expensive)
        self.target_ket = self.target_gen.get_target_ket(self.cutoff_dim)

        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Delegate to Circuit Strategy
        prog = self.circuit_context.build_reset_program()
        result = self.eng.run(prog)
        self.current_ket = result.state.ket()[:, 0] # Assuming mode 0 is the memory
        
        return self._ket_to_observation(self.current_ket), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Delegate Circuit Execution
        prog = self.circuit_context.build_step_program(action)
        result = self.eng.run(prog)
        self.current_ket = result.state.ket()[:, 0]
        
        # 2. Delegate Reward Calculation
        # Pass context like step number, max steps, etc.
        step_info = {
            "step": self.current_step, 
            "max_steps": self.max_steps,
            "samples": result.samples
        }
        
        reward, terminated, info = self.reward_mech.compute(
            self.current_ket, 
            self.target_ket, 
            step_info
        )
        
        truncated = self.current_step >= self.max_steps
        obs = self._ket_to_observation(self.current_ket)
        
        return obs, reward, terminated, truncated, info

    def _ket_to_observation(self, ket):
        # ... (Same implementation as your base class) ...
        # Can be moved to a utility function
        pass