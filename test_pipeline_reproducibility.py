"""Deterministic toy calculations; these are not model or security tests."""

import numpy as np


def _synthetic_case(seed=2026):
    rng = np.random.default_rng(seed)

    time_steps = 100
    fast_rate = 0.85
    slow_rate = 0.05
    noise_sd_fast = 0.01
    noise_sd_slow = 0.001

    h_fast = np.zeros(time_steps)
    h_slow = np.zeros(time_steps)
    h_fast[0] = 1.0
    h_slow[0] = 1.0

    for t in range(1, time_steps):
        h_fast[t] = h_fast[t - 1] * np.exp(-fast_rate) + noise_sd_fast * rng.standard_normal()
        h_slow[t] = h_slow[t - 1] * np.exp(-slow_rate) + noise_sd_slow * rng.standard_normal()

    energy_fast = np.linalg.norm(h_fast[-10:])
    energy_slow = np.linalg.norm(h_slow[-10:])

    dim = 8
    projection = np.zeros((dim, dim))
    projection[:4, :4] = np.eye(4)
    projection[4:, 4:] = np.eye(4)
    coupling = 1e-4 * rng.standard_normal((dim, dim))
    matrix = projection + coupling
    off_block_energy = np.linalg.norm(matrix[:4, 4:]) + np.linalg.norm(matrix[4:, :4])
    block_energy = np.linalg.norm(matrix[:4, :4]) + np.linalg.norm(matrix[4:, 4:])
    coupling_ratio = off_block_energy / block_energy

    return h_fast, h_slow, energy_fast, energy_slow, coupling_ratio


def run_reproducibility_suite():
    print("=" * 78)
    print("RUNNING SYNTHETIC TOY CALCULATIONS (NOT A MODEL OR SECURITY VALIDATION)")
    print("=" * 78)

    first = _synthetic_case()
    replay = _synthetic_case()
    for first_array, replay_array in zip(first[:2], replay[:2]):
        np.testing.assert_array_equal(first_array, replay_array)

    h_fast, h_slow, energy_fast, energy_slow, coupling_ratio = first
    assert np.isfinite(h_fast).all() and np.isfinite(h_slow).all()
    assert np.isfinite([energy_fast, energy_slow, coupling_ratio]).all()

    print(f"Fast-channel residual L2 energy (t=90..99): {energy_fast:.8f}")
    print(f"Slow-channel residual L2 energy (t=90..99): {energy_slow:.8f}")
    print("Residual-energy ordering is descriptive only: the fast channel receives 10x noise.")
    print("  -> Same-seed toy replay matched exactly in this environment.")
    print("-" * 78)
    print(f"Synthetic off-block coupling ratio: {coupling_ratio:.8e}")
    assert coupling_ratio < 1e-3
    print("  -> Toy threshold passed; this is not a gradient-leakage measurement.")
    print("=" * 78)


if __name__ == "__main__":
    run_reproducibility_suite()

