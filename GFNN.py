
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class DivergenceResult:
    """Results of Synaptic Divergence evaluation."""

    neuron_divergence: np.ndarray
    mean_activation: np.ndarray
    region_divergence: Dict[str, float]
    selected_spot: Optional[str]
    maximum_divergence: float
    growth_required: bool


def compute_synaptic_divergence(
    activations: np.ndarray,
    threshold: float,
    spots: Optional[Dict[str, List[int]]] = None,
) -> DivergenceResult:
    """
    Compute the Synaptic Divergence metric.

    Parameters
    ----------
    activations : np.ndarray
        Hidden-layer activation matrix with shape (n_samples, n_neurons).
        Each row corresponds to one training sample.
        Each column corresponds to one hidden neuron.

    threshold : float
        User-defined divergence threshold.

    spots : dict, optional
        Dictionary defining admissible Topological Spots.
        Example:
        {
            "spot_1": [0, 1],
            "spot_2": [2, 3],
            "spot_3": [4, 5]
        }

        If omitted, each neuron is treated as an individual spot.

    Returns
    -------
    DivergenceResult
        Neuron-level divergence, regional divergence,
        selected Topological Spot, and growth decision.
    """

    # --------------------------------------------------
    # 1. Input validation
    # --------------------------------------------------

    A = np.asarray(activations, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError(
            "activations must be a 2D matrix "
            "(n_samples, n_neurons)."
        )

    n_samples, n_neurons = A.shape

    if n_samples == 0 or n_neurons == 0:
        raise ValueError("Activation matrix cannot be empty.")

    if not np.all(np.isfinite(A)):
        raise ValueError(
            "Activation matrix contains NaN or infinite values."
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            "threshold must be a finite non-negative number."
        )

    # --------------------------------------------------
    # 2. Empirical mean activation of each neuron
    # --------------------------------------------------

    mean_activation = np.mean(A, axis=0)

    # --------------------------------------------------
    # 3. Neuron-level Synaptic Divergence
    #
    # D_j = (1/N) * sum_i |a_ij - mean_j|
    # --------------------------------------------------

    neuron_divergence = np.mean(
        np.abs(A - mean_activation[None, :]),
        axis=0
    )

    # --------------------------------------------------
    # 4. Define admissible Topological Spots
    # --------------------------------------------------

    if spots is None:
        spots = {
            f"neuron_{j}": [j]
            for j in range(n_neurons)
        }

    if not spots:
        raise ValueError("At least one Topological Spot is required.")

    # --------------------------------------------------
    # 5. Regional Synaptic Divergence
    #
    # D_R = (1 / |R|) * sum_{j in R} D_j
    # --------------------------------------------------

    region_divergence = {}

    for spot_name, neuron_indices in spots.items():

        indices = np.asarray(neuron_indices, dtype=int)

        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(
                f"{spot_name} must contain at least one neuron."
            )

        if np.any(indices < 0) or np.any(indices >= n_neurons):
            raise ValueError(
                f"{spot_name} contains an invalid neuron index."
            )

        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                f"{spot_name} contains duplicate neuron indices."
            )

        region_divergence[spot_name] = float(
            np.mean(neuron_divergence[indices])
        )

    # --------------------------------------------------
    # 6. Select the highest-divergence Topological Spot
    # --------------------------------------------------

    selected_spot = max(
        region_divergence,
        key=region_divergence.get
    )

    maximum_divergence = region_divergence[selected_spot]

    # --------------------------------------------------
    # 7. Growth decision
    # --------------------------------------------------

    growth_required = maximum_divergence > threshold

    return DivergenceResult(
        neuron_divergence=neuron_divergence,
        mean_activation=mean_activation,
        region_divergence=region_divergence,
        selected_spot=selected_spot,
        maximum_divergence=float(maximum_divergence),
        growth_required=bool(growth_required)
    )


# ======================================================
# Example: GFNN Synaptic Divergence Evaluation
# ======================================================

if __name__ == "__main__":

    # Reproducible example
    rng = np.random.default_rng(42)

    n_samples = 100
    n_neurons = 6

    # Simulated hidden-layer activations
    activations = np.tanh(
        rng.normal(
            loc=0.0,
            scale=1.0,
            size=(n_samples, n_neurons)
        )
    )

    # Define admissible Topological Spots
    spots = {
        "spot_1": [0, 1],
        "spot_2": [2, 3],
        "spot_3": [4, 5]
    }

    # Divergence threshold
    threshold = 0.30

    # Evaluate divergence
    result = compute_synaptic_divergence(
        activations=activations,
        threshold=threshold,
        spots=spots
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\nGFNN SYNAPTIC DIVERGENCE ANALYSIS")
    print("=" * 55)

    print("\nMean activation:")
    print(np.round(result.mean_activation, 6))

    print("\nNeuron-level Synaptic Divergence:")

    for j, value in enumerate(result.neuron_divergence):
        print(f"Neuron {j}: {value:.6f}")

    print("\nRegional Synaptic Divergence:")

    for name, value in result.region_divergence.items():
        print(f"{name}: {value:.6f}")

    print("\nSelected Topological Spot:")
    print(result.selected_spot)

    print("\nMaximum Divergence:")
    print(f"{result.maximum_divergence:.6f}")

    print("\nGrowth Decision:")

    if result.growth_required:
        print("GROWTH REQUIRED")
    else:
        print("NO GROWTH REQUIRED")

    print("=" * 55)
