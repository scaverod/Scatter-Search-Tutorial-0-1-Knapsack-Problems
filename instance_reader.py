import os
from typing import List, Tuple


def validate_instance_file(instance_path: str) -> None:
    """Validate that the instance file exists and is accessible."""
    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    if not os.path.isfile(instance_path):
        raise ValueError(f"Path is not a file: {instance_path}")


def read_instance_lines(instance_path: str) -> List[str]:
    """Read and clean lines from the instance file."""
    with open(instance_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 1:
        raise ValueError("Empty or invalid instance file")
    return lines


def parse_instance_header(header_line: str) -> Tuple[int, int]:
    """Parse the header line to get number of items and capacity."""
    try:
        n, capacity = map(int, header_line.split())
        if n <= 0:
            raise ValueError(f"Number of items must be positive, got {n}")
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        return n, capacity
    except ValueError as e:
        raise ValueError(f"Invalid header line format: {header_line}. Error: {e}")


def parse_item_data(lines: List[str], n: int) -> Tuple[List[int], List[int]]:
    """Parse item data (profit and weight) from instance lines."""
    if len(lines) < n + 1:
        raise ValueError(f"Insufficient lines in file: expected {n+1}, got {len(lines)}")

    profits, weights = [], []
    for i in range(1, n + 1):
        try:
            profit, weight = map(int, lines[i].split())
            if profit < 0:
                raise ValueError(f"Item {i-1}: profit cannot be negative, got {profit}")
            if weight <= 0:
                raise ValueError(f"Item {i-1}: weight must be positive, got {weight}")
            profits.append(profit)
            weights.append(weight)
        except ValueError as e:
            raise ValueError(f"Invalid item data at line {i+1}: {lines[i]}. Error: {e}")

    return profits, weights


def load_instance_from_file(instance_path: str) -> Tuple[List[int], List[int], int, int]:
    """
    Load a knapsack instance from file.

    File format:
    - First line: n capacity (number of items and knapsack capacity)
    - Next n lines: profit weight (for each item)

    Returns: (profits, weights, capacity, n)
    """
    # Step 1: Validate file
    validate_instance_file(instance_path)

    # Step 2: Read lines
    lines = read_instance_lines(instance_path)

    # Step 3: Parse header
    n, capacity = parse_instance_header(lines[0])

    # Step 4: Parse item data
    profits, weights = parse_item_data(lines, n)

    return profits, weights, capacity, n


def create_example_instance() -> Tuple[List[int], List[int], int, int]:
    """Create a small example instance for testing."""
    profits = [10, 20, 30]
    weights = [5, 10, 15]
    capacity = 20
    n = len(profits)
    return profits, weights, capacity, n


if __name__ == "__main__":
    """Example usage: read a specific instance and solve it."""
    from scatter_search_core import KnapsackProblem, run_scatter_search, SSParams, is_feasible, total_weight
    import numpy as np
    import random

    def set_reproducible_seeds():
        """Set seeds for reproducible results."""
        random.seed(1234)
        np.random.seed(4321)

    def solve_instance(instance_path: str):
        """Load and solve a knapsack instance."""
        print(f"Loading instance: {instance_path}")

        try:
            profits, weights, capacity, n = load_instance_from_file(instance_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading instance: {e}")
            print("Using example instance instead...")
            profits, weights, capacity, n = create_example_instance()

        pb = KnapsackProblem(profits=profits, weights=weights, capacity=capacity)

        print(f"Problem size: {n} items, capacity: {capacity}")

        # Configure and run algorithm
        params = SSParams(PSize=15, b=6, b1=3, max_iter=30, patience=8, rebuild_patience=3, elite_preservation=True)
        results = run_scatter_search(pb, params, verbose=True)

        # Display results
        best_sol = results['best_solution']
        best_val = results['best_value']
        best_weight = total_weight(pb, best_sol)
        feasible = is_feasible(pb, best_sol)

        print("\n" + "="*40)
        print("SOLUTION RESULTS")
        print("="*40)
        print(f"Best value: {best_val}")
        print(f"Total weight: {best_weight} / {capacity}")
        print(f"Feasible: {feasible}")
        print(f"Selected items: {[i for i, x in enumerate(best_sol) if x == 1]}")
        print("="*40)

    # Main execution
    set_reproducible_seeds()
    instance_path = "instances_01_KP/large_scale/knapPI_3_100_1000_1"
    solve_instance(instance_path)

