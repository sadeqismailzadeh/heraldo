import quantum_agent
import numpy as np
import strawberryfields.backends.fockbackend.ops as ops
from strawberryfields.backends.fockbackend.circuit import Circuit

def _prepare_multimode_patched(self, state, modes):
    r"""
    Prepares a given mode or list of modes in the given state.
    
    This patched version includes an Unentangled Detection Optimization:
    If the system is in a pure state, and we are preparing a subset of modes with a 
    pure state, and the resulting cut leaves the system unentangled (separable), 
    we update the state vector directly without converting to a mixed state (density matrix).
    """
    if isinstance(modes, int):
        modes = [modes]

    n_modes = len(modes)
    pure_shape = tuple([self._trunc] * n_modes)
    mixed_shape = tuple([self._trunc] * (2 * n_modes))
    pure_shape_as_vector = tuple([self._trunc**n_modes])
    mixed_shape_as_matrix = tuple([self._trunc**n_modes] * 2)

    # Do consistency checks
    # pylint: disable=consider-using-in
    if self._checks:
        if (
            state.shape != pure_shape
            and state.shape != mixed_shape
            and state.shape != pure_shape_as_vector
            and state.shape != mixed_shape_as_matrix
        ):
            raise ValueError("Incorrect shape for state preparation")
        if len(modes) != len(set(modes)):
            raise ValueError("The specified modes cannot appear multiple times.")

    # reshape to support input both as tensor and vector/matrix
    if state.shape == pure_shape_as_vector:
        state = state.reshape(pure_shape)
    elif state.shape == mixed_shape_as_matrix:
        state = state.reshape(mixed_shape)

    # -------------------------------------------------------------------------
    # UNENTANGLED DETECTION PATCH (OPTIMIZATION)
    # -------------------------------------------------------------------------
    # Attempt to keep state pure if:
    # 1. Current state is pure
    # 2. Input state is pure
    # 3. We are not replacing all modes (which is trivial anyway)
    if self._pure and state.shape == pure_shape and self._num_modes > n_modes:
        
        # Identify kept modes and replaced modes
        mode_list = list(modes)
        all_modes = np.arange(self._num_modes)
        kept_mask = np.isin(all_modes, mode_list, invert=True)
        kept_modes = all_modes[kept_mask] # these are sorted
        
        # Heuristic: Extract candidate vector for kept modes from max amplitude slice
        flat_idx = np.argmax(np.abs(self._state))
        multi_idx = np.unravel_index(flat_idx, self._state.shape)
        
        slicer = [slice(None)] * self._num_modes
        for m in mode_list:
            slicer[m] = multi_idx[m]
            
        candidate_vec = self._state[tuple(slicer)].copy()
        norm_candidate = np.linalg.norm(candidate_vec)
        
        # If the slice we picked is effectively zero (unlikely given argmax), skip
        if norm_candidate > 1e-12:
            candidate_vec /= norm_candidate
            
            # Check separability: Project full state onto candidate (conjugated)
            # Contract kept_modes indices of state with candidate indices
            projection = np.tensordot(self._state, candidate_vec.conj(),
                                      axes=(list(kept_modes), list(range(len(kept_modes)))))
            
            # Robust Check: Compare projection norm to the actual state norm.
            # Truncation errors can cause state norm to drop below 1.0.
            # If separable: |psi_total> = c * |psi_kept> (x) |psi_replaced>
            # |projection> = c * |psi_replaced>
            # norm(projection) should equal norm(total_state)
            
            state_norm = np.linalg.norm(self._state)
            proj_norm = np.linalg.norm(projection)
            
            # Use a relative tolerance based on the state norm
            if abs(proj_norm - state_norm) < (1e-5 * state_norm + 1e-9):
                # OPTIMIZATION SUCCESS: Update pure state directly
                
                # New state = candidate_vec (kept) (x) state (replaced)
                new_full_state = np.tensordot(candidate_vec, state, axes=0)
                
                # Tensor axes are now: [kept_modes...] + [mode_list...]
                # Need to permute back to natural order 0, 1, ... N-1
                current_ordering = list(kept_modes) + mode_list
                inv_perm = np.argsort(current_ordering)
                
                self._state = new_full_state.transpose(inv_perm)
                
                # Re-normalize to prevent drift accumulation
                new_norm = np.linalg.norm(self._state)
                if new_norm > 1e-12:
                    self._state /= new_norm
                    
                # self._pure remains True
                return

    # -------------------------------------------------------------------------
    # FALLBACK: ORIGINAL LOGIC
    # -------------------------------------------------------------------------

    if self._num_modes == n_modes:
        # Hack for marginally faster state preparation
        self._state = state.astype(ops.def_type)
        self._pure = bool(state.shape == pure_shape)
    else:
        if self._pure:
            self._state = ops.mix(self._state, self._num_modes)
            self._pure = False

        if state.shape == pure_shape:
            state = ops.mix(state, len(modes))

        # Take the partial trace
        # todo: For performance the partial trace could be done directly from the pure state. This would of course require a better partial trace function...
        reduced_state = ops.partial_trace(self._state, self._num_modes, modes)

        # Insert state at the end (I know there is also tensor() from ops but it has extra aguments wich only confuse here)
        self._state = np.tensordot(reduced_state, state, axes=0)

        # unless the preparation was meant to go into the last modes in the standard order, we need to swap indices around
    if modes != list(range(self._num_modes - len(modes), self._num_modes)):
        mode_permutation = [x for x in range(self._num_modes) if x not in modes] + modes
        if self._pure:
            scale = 1
            index_permutation = mode_permutation
        else:
            scale = 2
            index_permutation = [
                scale * x + i for x in mode_permutation for i in (0, 1)
            ]  # two indices per mode if we have pure states
        index_permutation = np.argsort(index_permutation)

        self._state = np.transpose(self._state, index_permutation)


def patch_prepare_multimode():
    if not hasattr(Circuit, 'prepare_multimode_original'):
        Circuit.prepare_multimode_original = Circuit.prepare_multimode
    Circuit.prepare_multimode = _prepare_multimode_patched
    print("Circuit.prepare_multimode patched with unentangled pure state optimization.")

def revert_prepare_multimode_patch():
    if hasattr(Circuit, 'prepare_multimode_original'):
        Circuit.prepare_multimode = Circuit.prepare_multimode_original
        print("Circuit.prepare_multimode reverted to original.")

# -----------------------------------------------------------------------------
# VERIFICATION
# -----------------------------------------------------------------------------

def verify_correctness():
    print("--- Running prepare_multimode correctness verification ---")
    import strawberryfields as sf
    from strawberryfields.ops import Dgate, Vacuum

    revert_prepare_multimode_patch()
    
    # 1. Test case where state IS unentangled
    prog = sf.Program(2)
    with prog.context as q:
        Dgate(0.5) | q[0]
        Dgate(0.5) | q[1]
        # Prepare q[0] in Vacuum again (should trigger opt)
        Vacuum() | q[0]

    eng_orig = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    state_orig = eng_orig.run(prog).state
    
    patch_prepare_multimode()
    eng_opt = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    state_opt = eng_opt.run(prog).state
    
    # Original logic always mixes when subset is prepared
    print(f"Original state pure? {state_orig.is_pure}") # Expected: False
    print(f"Optimized state pure? {state_opt.is_pure}") # Expected: True

    diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
    print(f"Max difference (Unentangled case): {diff:.2e}")
    assert diff < 1e-10

    # 2. Test case where state IS entangled (Optimization should fallback)
    revert_prepare_multimode_patch()
    prog_ent = sf.Program(2)
    with prog_ent.context as q:
        sf.ops.S2gate(1.0) | (q[0], q[1])
        Vacuum() | q[0]

    eng_orig = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    state_orig = eng_orig.run(prog_ent).state

    patch_prepare_multimode()
    eng_opt = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    state_opt = eng_opt.run(prog_ent).state
    
    print(f"Entangled case - Optimized state pure? {state_opt.is_pure}") 
    # Must be False; remaining q[1] is mixed
    
    diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
    print(f"Max difference (Entangled case): {diff:.2e}")
    assert diff < 1e-10
    
    revert_prepare_multimode_patch()
    print("Verification passed.")

if __name__ == "__main__":
    verify_correctness()
