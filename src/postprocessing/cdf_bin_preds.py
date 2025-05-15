import numpy as np
import time


def get_cdf_from_binned_probabilities_loop(
    y_value: float, probabilities: list[float], bin_edges: list[float]
) -> float:
    """
    Calculates the Cumulative Distribution Function (CDF) value F(y) = P(Y <= y_value)
    from a binned probability distribution using an iterative approach.

    Assumes probability mass within each bin is uniformly distributed for interpolation.
    The probability probabilities[i] is associated with the interval (bin_edges[i], bin_edges[i+1]].

    Args:
        y_value: The value at which to evaluate the CDF.
        probabilities: A list or NumPy array of probabilities for each bin.
        bin_edges: A list or NumPy array of N+1 bin edges, defining N bins. Must be sorted non-decreasingly.

    Returns:
        The calculated CDF value F(y_value).

    Raises:
        TypeError: If probabilities or bin_edges are not list-like or NumPy arrays.
        ValueError: If inputs are invalid.
    """
    # --- Input Validation ---
    if not isinstance(probabilities, (list, np.ndarray)):
        raise TypeError("probabilities must be a list or NumPy array.")
    if not isinstance(bin_edges, (list, np.ndarray)):
        raise TypeError("bin_edges must be a list or NumPy array.")

    probs_arr = np.asarray(probabilities, dtype=float)
    edges_arr = np.asarray(bin_edges, dtype=float)

    if not (probs_arr.ndim == 1 and edges_arr.ndim == 1):
        raise ValueError("probabilities and bin_edges must be 1-dimensional.")
    if len(edges_arr) != len(probs_arr) + 1:
        raise ValueError(
            f"Length of bin_edges ({len(edges_arr)}) must be length of "
            f"probabilities ({len(probs_arr)}) + 1."
        )
    if np.any(np.diff(edges_arr) < 0):
        raise ValueError("bin_edges must be sorted in non-decreasing order.")
    if np.any(probs_arr < 0):
        raise ValueError("Probabilities cannot be negative.")
    # --- End Validation ---

    # --- Handle boundary cases ---
    if y_value <= edges_arr[0]:
        return 0.0
    # Note: y_value >= edges_arr[-1] is handled correctly by the loop logic below

    # --- Iterative Calculation ---
    cdf_value = 0.0
    for i in range(len(probs_arr)):
        lower_bound = edges_arr[i]
        upper_bound = edges_arr[i + 1]
        prob_in_bin = probs_arr[i]

        if y_value >= upper_bound:
            # y is beyond or exactly at the end of this bin, include full probability
            cdf_value += prob_in_bin
        elif y_value > lower_bound:
            # y is strictly within this bin (lower_bound < y_value < upper_bound)
            # Include the interpolated fraction of this bin's probability
            bin_width = upper_bound - lower_bound
            if bin_width > 0:
                fraction = (y_value - lower_bound) / bin_width
                cdf_value += prob_in_bin * fraction
            # If bin_width is 0, this block adds no probability (correct).
            break  # Found the bin containing y_value, calculation is complete.
        else:  # y_value <= lower_bound
            # y is before or exactly at the start of this bin.
            # No contribution from this bin or subsequent ones needed for P(Y <= y).
            break

    return cdf_value


def get_cdf_from_binned_probabilities_numpy(
    y_value: float, probabilities: list[float], bin_edges: list[float]
) -> float:
    """
    Calculates the Cumulative Distribution Function (CDF) value F(y) = P(Y <= y_value)
    from a binned probability distribution using NumPy for potentially optimized computation.

    Assumes probability mass within each bin is uniformly distributed for interpolation.
    The probability probabilities[i] is associated with the interval (bin_edges[i], bin_edges[i+1]].

    Args:
        y_value: The value at which to evaluate the CDF.
        probabilities: A list or NumPy array of probabilities for each bin.
        bin_edges: A list or NumPy array of N+1 bin edges, defining N bins. Must be sorted non-decreasingly.

    Returns:
        The calculated CDF value F(y_value).

    Raises:
        TypeError: If probabilities or bin_edges are not list-like or NumPy arrays.
        ValueError: If inputs are invalid.
    """
    # --- Input Validation ---
    if not isinstance(probabilities, (list, np.ndarray)):
        raise TypeError("probabilities must be a list or NumPy array.")
    if not isinstance(bin_edges, (list, np.ndarray)):
        raise TypeError("bin_edges must be a list or NumPy array.")

    probs_arr = np.asarray(probabilities, dtype=float)
    edges_arr = np.asarray(bin_edges, dtype=float)

    if not (probs_arr.ndim == 1 and edges_arr.ndim == 1):
        raise ValueError("probabilities and bin_edges must be 1-dimensional.")
    if len(edges_arr) != len(probs_arr) + 1:
        raise ValueError(
            f"Length of bin_edges ({len(edges_arr)}) must be length of "
            f"probabilities ({len(probs_arr)}) + 1."
        )
    if np.any(np.diff(edges_arr) < 0):
        raise ValueError("bin_edges must be sorted in non-decreasing order.")
    if np.any(probs_arr < 0):
        raise ValueError("Probabilities cannot be negative.")
    # --- End Validation ---

    # --- Handle boundary cases ---
    if y_value <= edges_arr[0]:
        return 0.0
    if y_value >= edges_arr[-1]:
        # If y is beyond or at the last edge, CDF is the total probability sum
        return np.sum(probs_arr)

    # --- Vectorized Calculation ---

    # 1. Identify bins fully completed by y_value (where y_value >= upper_bound)
    # We compare y_value with the *upper* edge of each bin (edges_arr[1:])
    mask_completed = y_value >= edges_arr[1:]
    # Sum the probabilities of these fully completed bins
    cdf_value = np.sum(probs_arr[mask_completed])

    # 2. Identify the single bin (if any) where interpolation is needed
    # This bin 'idx' must satisfy: edges_arr[idx] < y_value < edges_arr[idx+1]
    mask_interp = (y_value > edges_arr[:-1]) & (y_value < edges_arr[1:])

    # Find the index where interpolation should occur
    idx_interp = np.flatnonzero(mask_interp)

    if len(idx_interp) > 0:
        # Normally, there should only be one such index
        idx = idx_interp[0]
        lower_bound = edges_arr[idx]
        upper_bound = edges_arr[idx + 1]
        prob_in_bin = probs_arr[idx]

        bin_width = upper_bound - lower_bound
        # Check bin_width > 0 to avoid division by zero and handle zero-width bins
        if bin_width > 0:
            fraction = (y_value - lower_bound) / bin_width
            # Add the interpolated probability from this bin
            cdf_value += prob_in_bin * fraction
        # If bin_width is 0, no contribution is added here (correct, as the
        # full probability was added by the mask_completed sum if y_value >= upper_bound).

    return cdf_value


def get_cdf_from_binned_probabilities_multidim(
    target_values: np.ndarray,  # Shape (H, W)
    probabilities: np.ndarray,  # Shape (NUM_BINS, H, W)
    bin_edges: np.ndarray,  # Shape (NUM_BINS + 1), 1D
) -> np.ndarray:  # Shape (H, W)
    """
    Calculates the Cumulative Distribution Function (CDF) values F(Y <= y) pixel-wise
    for multi-dimensional inputs using NumPy for optimized computation.

    Assumes probability mass within each bin is uniformly distributed for interpolation.
    The probability probabilities[b, h, w] is associated with the interval
    (bin_edges[b], bin_edges[b+1]] for pixel (h, w).

    Args:
        target_values: NumPy array of shape (H, W) with the y-values for which
                       to compute the CDF at each pixel.
        probabilities: NumPy array of shape (NUM_BINS, H, W) where NUM_BINS is the
                       number of probability bins. probabilities[b, h, w] is the
                       probability for the b-th bin at pixel (h, w).
        bin_edges: 1D NumPy array of shape (NUM_BINS + 1) defining the bin edges,
                   sorted in non-decreasing order. This is common for all pixels.

    Returns:
        NumPy array of shape (H, W) containing the calculated CDF values for each pixel.

    Raises:
        TypeError: If inputs are not NumPy arrays.
        ValueError: If input shapes, dimensions, or properties (e.g., sorted edges)
                    are invalid.
    """
    # --- Input Validation ---
    if not isinstance(target_values, np.ndarray):
        raise TypeError("target_values must be a NumPy array.")
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array.")
    if not isinstance(bin_edges, np.ndarray):
        raise TypeError("bin_edges must be a NumPy array.")

    if target_values.ndim != 2:
        raise ValueError(
            f"target_values must be 2-dimensional (H, W), got {target_values.ndim} dimensions."
        )
    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must be 3-dimensional (NUM_BINS, H, W), got {probabilities.ndim} dimensions."
        )
    if bin_edges.ndim != 1:
        raise ValueError(
            f"bin_edges must be 1-dimensional, got {bin_edges.ndim} dimensions."
        )

    num_bins_from_probs = probabilities.shape[0]
    h_probs, w_probs = probabilities.shape[1], probabilities.shape[2]
    h_targets, w_targets = target_values.shape

    if bin_edges.shape[0] != num_bins_from_probs + 1:
        raise ValueError(
            f"Length of bin_edges ({bin_edges.shape[0]}) must be "
            f"NUM_BINS from probabilities ({num_bins_from_probs}) + 1."
        )
    if not (h_probs == h_targets and w_probs == w_targets):
        raise ValueError(
            f"Spatial dimensions of probabilities ({h_probs},{w_probs}) must match "
            f"target_values dimensions ({h_targets},{w_targets})."
        )

    if np.any(np.diff(bin_edges) < 0):  # Checks for non-decreasing order
        raise ValueError("bin_edges must be sorted in non-decreasing order.")
    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")
    # --- End Validation ---

    H, W = target_values.shape
    NUM_BINS = probabilities.shape[0]

    # Initialize output array with zeros
    final_output = np.zeros_like(target_values, dtype=float)

    # --- Handle boundary conditions directly on the full output array ---

    # Condition 1: target_values <= bin_edges[0] (y is less than or at the very first edge)
    # CDF is 0 for these pixels. (Already initialized to 0, but can be explicit)
    mask_le_first_edge = target_values <= bin_edges[0]
    final_output[mask_le_first_edge] = 0.0

    # Condition 2: target_values >= bin_edges[-1] (y is greater than or at the very last edge)
    # CDF is the sum of all probabilities for each respective pixel.
    mask_ge_last_edge = target_values >= bin_edges[-1]
    # Sum probabilities along the NUM_BINS axis (axis 0)
    sum_all_probs_per_pixel = np.sum(probabilities, axis=0)  # Shape (H, W)
    final_output[mask_ge_last_edge] = sum_all_probs_per_pixel[mask_ge_last_edge]

    # --- Core logic for pixels not handled by the above boundary conditions ---
    # Create a mask for pixels that require the main CDF calculation logic
    core_processing_mask = ~(mask_le_first_edge | mask_ge_last_edge)  # Shape (H,W)

    # If all pixels are handled by boundary conditions, we can return early
    if not np.any(core_processing_mask):
        return final_output

    # Filter inputs to only include "core" pixels that need detailed processing.
    # This flattens the H,W dimensions for these core pixels temporarily.
    Y_core = target_values[core_processing_mask]  # Shape (num_core_pixels,)
    P_core = probabilities[:, core_processing_mask]  # Shape (NUM_BINS, num_core_pixels)

    # Reshape y_values and bin_edges for broadcasting:
    # Y_core_bc will be (1, num_core_pixels)
    # E_lower_bc, E_upper_bc will be (NUM_BINS, 1)
    Y_core_bc = Y_core[np.newaxis, :]
    E_lower = bin_edges[:-1]  # Lower edges of all bins
    E_upper = bin_edges[1:]  # Upper edges of all bins
    E_lower_bc = E_lower[:, np.newaxis]
    E_upper_bc = E_upper[:, np.newaxis]

    # 1. Sum probabilities from bins fully completed by Y_core values
    # mask_completed_core[b, p] is True if Y_core[p] >= E_upper[b]
    mask_completed_core = Y_core_bc >= E_upper_bc  # Shape (NUM_BINS, num_core_pixels)
    # cdf_base_core[p] sums P_core[b,p] for completed bins 'b' for pixel 'p'
    cdf_base_core = np.sum(
        P_core * mask_completed_core, axis=0
    )  # Shape (num_core_pixels,)

    # 2. Calculate and add the interpolated part for the bin containing Y_core
    # mask_interp_region_core[b, p] is True if E_lower[b] < Y_core[p] < E_upper[b]
    mask_interp_region_core = (Y_core_bc > E_lower_bc) & (
        Y_core_bc < E_upper_bc
    )  # Shape (NUM_BINS, num_core_pixels)

    # For each core pixel, determine if any bin is marked for interpolation
    # active_interp_pixels_mask_in_core[p] is True if pixel 'p' has an interpolation bin
    active_interp_pixels_mask_in_core = np.any(
        mask_interp_region_core, axis=0
    )  # Shape (num_core_pixels,)
    interp_cdf_part_core = np.zeros_like(
        Y_core, dtype=float
    )  # Initialize for all core pixels

    # Proceed only if there are any pixels that actually need interpolation
    if np.any(active_interp_pixels_mask_in_core):
        # Get the indices (within the Y_core array) of pixels that need interpolation
        indices_of_interp_pixels_in_core = np.flatnonzero(
            active_interp_pixels_mask_in_core
        )

        # Filter data for only these specific pixels requiring interpolation
        y_for_interp = Y_core[
            indices_of_interp_pixels_in_core
        ]  # Values of Y for interpolation

        # Probabilities and masks for these specific pixels
        # P_for_interp_pixels shape: (NUM_BINS, num_actual_interp_pixels)
        P_for_interp_pixels = P_core[:, indices_of_interp_pixels_in_core]
        # mask_bins_for_interp_pixels shape: (NUM_BINS, num_actual_interp_pixels)
        mask_bins_for_interp_pixels = mask_interp_region_core[
            :, indices_of_interp_pixels_in_core
        ]

        # For each pixel needing interpolation, find the index of its specific interpolation bin
        # (np.argmax returns the index of the first True along axis 0)
        # interp_bin_indices shape: (num_actual_interp_pixels,)
        interp_bin_indices = np.argmax(mask_bins_for_interp_pixels, axis=0)

        # Extract the probability, lower bound, and upper bound for the identified interpolation bin
        # for each of these pixels using advanced indexing.
        num_actual_interp_pixels = len(y_for_interp)
        prob_in_interp_bin = P_for_interp_pixels[
            interp_bin_indices, np.arange(num_actual_interp_pixels)
        ]
        lower_b_interp = E_lower[interp_bin_indices]
        upper_b_interp = E_upper[interp_bin_indices]

        bin_width_interp = upper_b_interp - lower_b_interp

        # Calculate interpolation fraction (handle division by zero for zero-width bins)
        fraction_values = np.zeros_like(y_for_interp, dtype=float)
        # Create a mask for valid bin widths (avoid division by zero)
        valid_bin_width_mask = bin_width_interp > 0

        # Perform calculation only for valid bin widths
        numerator = (
            y_for_interp[valid_bin_width_mask] - lower_b_interp[valid_bin_width_mask]
        )
        denominator = bin_width_interp[valid_bin_width_mask]
        fraction_values[valid_bin_width_mask] = numerator / denominator

        # Add the calculated interpolated probability to the corresponding elements
        # in interp_cdf_part_core (which was initialized for all core_pixels)
        interp_cdf_part_core[indices_of_interp_pixels_in_core] = (
            prob_in_interp_bin * fraction_values
        )

    # Combine the CDF from completed bins with the interpolated part for core pixels
    core_output_values = cdf_base_core + interp_cdf_part_core

    # Assign the calculated CDF values for core_processing_mask pixels back into the final output array
    final_output[core_processing_mask] = core_output_values

    return final_output


if __name__ == "__main__":
    # --- Testing both versions ---
    print("Comparing Loop and NumPy versions:")

    probs_example = [0.1, 0.1, 0.3, 0.4, 0.1]
    edges_example = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("\nStandard Example:")
    print(f"Probs: {probs_example}, Edges: {edges_example}")
    test_y_vals = [-0.1, 0.0, 0.1, 0.2, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0, 1.1]
    loop_times = []
    numpy_times = []
    for y in test_y_vals:
        start_loop = time.perf_counter()
        cdf_loop = get_cdf_from_binned_probabilities_loop(
            y, probs_example, edges_example
        )
        loop_times.append(time.perf_counter() - start_loop)

        start_numpy = time.perf_counter()
        cdf_np = get_cdf_from_binned_probabilities_numpy(
            y, probs_example, edges_example
        )
        numpy_times.append(time.perf_counter() - start_numpy)

        print(
            f"y={y:.2f} -> Loop={cdf_loop:.4f}, NumPy={cdf_np:.4f}{' <-- Differs!' if not np.isclose(cdf_loop, cdf_np) else ''}"
        )
        print(
            f"y={y:.2f} -> Loop={cdf_loop:.4f}, NumPy={cdf_np:.4f}{' <-- Differs!' if not np.isclose(cdf_loop, cdf_np) else ''}"
        )

    print(f"\nAverage Loop Time: {np.mean(loop_times):.6f}s")
    print(f"Average NumPy Time: {np.mean(numpy_times):.6f}s")
    probs_zw = [0.2, 0.3, 0.5]  # Sum = 1.0
    edges_zw = [0.0, 0.2, 0.2, 0.4]  # Includes zero-width bin for point mass at 0.2
    print("\nZero-Width Bin Example:")
    print(f"Probs: {probs_zw}, Edges: {edges_zw}")
    test_y_zw = [0.0, 0.1, 0.19, 0.2, 0.21, 0.3, 0.4, 0.5]
    for y in test_y_zw:
        cdf_loop = get_cdf_from_binned_probabilities_loop(y, probs_zw, edges_zw)
        cdf_np = get_cdf_from_binned_probabilities_numpy(y, probs_zw, edges_zw)
        print(
            f"y={y:.2f} -> Loop={cdf_loop:.4f}, NumPy={cdf_np:.4f}{' <-- Differs!' if not np.isclose(cdf_loop, cdf_np) else ''}"
        )

    # Example Usage:
    NUM_BINS_EX = 5
    H_EX, W_EX = 3, 4  # Small spatial dimensions for testing

    # Common bin edges for all pixels
    bin_edges_ex = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # NUM_BINS_EX + 1 edges

    # Probabilities: Randomly generate, ensure sum to 1 for each pixel along bin axis
    np.random.seed(0)
    probabilities_ex_raw = np.random.rand(NUM_BINS_EX, H_EX, W_EX)
    probabilities_ex = probabilities_ex_raw / np.sum(
        probabilities_ex_raw, axis=0, keepdims=True
    )

    # Target values for each pixel
    target_values_ex = np.random.rand(H_EX, W_EX)
    # Add some edge cases for target_values
    target_values_ex[0, 0] = -0.1  # Below first edge
    target_values_ex[0, 1] = 0.0  # At first edge
    target_values_ex[0, 2] = 1.0  # At last edge
    target_values_ex[0, 3] = 1.1  # Above last edge
    target_values_ex[1, 0] = 0.2  # On an internal edge
    target_values_ex[1, 1] = 0.3  # Mid-bin

    print("--- Multi-dimensional CDF Calculation Example ---")
    print(f"Target Values (Y) shape: {target_values_ex.shape}")
    # print("Target Values (Y):\n", target_values_ex)
    print(f"Probabilities (P) shape: {probabilities_ex.shape}")
    print(f"Bin Edges (E) shape: {bin_edges_ex.shape}")
    # print("Bin Edges (E):\n", bin_edges_ex)

    cdf_output_multidim = get_cdf_from_binned_probabilities_multidim(
        target_values_ex, probabilities_ex, bin_edges_ex
    )
    print(f"\nOutput CDF Values shape: {cdf_output_multidim.shape}")
    print("Output CDF Values:\n", cdf_output_multidim)

    def get_cdf_from_binned_probabilities_loop_ref(
        y_value: float, probabilities_1d: np.ndarray, bin_edges_1d: np.ndarray
    ) -> float:
        if y_value <= bin_edges_1d[0]:
            return 0.0
        cdf_value = 0.0
        for i in range(len(probabilities_1d)):
            lower_bound, upper_bound, prob_in_bin = (
                bin_edges_1d[i],
                bin_edges_1d[i + 1],
                probabilities_1d[i],
            )
            if y_value >= upper_bound:
                cdf_value += prob_in_bin
            elif y_value > lower_bound:
                bin_width = upper_bound - lower_bound
                if bin_width > 0:
                    cdf_value += prob_in_bin * ((y_value - lower_bound) / bin_width)
                break
            else:
                break
        return cdf_value

    print("\nVerification for a few pixels against 1D reference function:")
    test_pixels = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (H_EX - 1, W_EX - 1)]
    for r, c in test_pixels:
        y_pixel = target_values_ex[r, c]
        probs_pixel = probabilities_ex[:, r, c]
        cdf_ref = get_cdf_from_binned_probabilities_loop_ref(
            y_pixel, probs_pixel, bin_edges_ex
        )
        cdf_multi_val = cdf_output_multidim[r, c]
        print(
            f"Pixel ({r},{c}): Y={y_pixel:.3f}, CDF_MultiDim={cdf_multi_val:.4f}, CDF_Ref={cdf_ref:.4f} "
            f"{'(OK)' if np.isclose(cdf_multi_val, cdf_ref) else '<-- MISMATCH'}"
        )
