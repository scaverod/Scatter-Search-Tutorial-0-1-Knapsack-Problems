import random
import numpy as np
import sys
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class KnapsackProblem:
    profits: List[int]
    weights: List[int]
    capacity: int

    @property
    def n(self) -> int:
        return len(self.profits)

    @property
    def efficiency_ratios(self) -> np.ndarray:
        return np.array(self.profits) / np.array(self.weights)

    @property
    def total_weight(self) -> int:
        return sum(self.weights)


def objective(pb: KnapsackProblem, x: List[int]) -> int:
    return int(sum(p * s for p, s in zip(pb.profits, x)))


def total_weight(pb: KnapsackProblem, x: List[int]) -> int:
    return int(sum(w * s for w, s in zip(pb.weights, x)))


def is_feasible(pb: KnapsackProblem, x: List[int]) -> bool:
    return total_weight(pb, x) <= pb.capacity


def ratio_order(pb: KnapsackProblem, ascending: bool = False) -> List[int]:
    indices = list(range(pb.n))
    ratios = pb.efficiency_ratios
    indices.sort(key=lambda i: ratios[i], reverse=not ascending)
    return indices


def estimate_capacity_items(pb: KnapsackProblem):
    avg_weight = pb.total_weight / pb.n
    return max(1, int(pb.capacity / avg_weight))


def create_systematic_solution(seed, h, q):
    n = len(seed)
    solution = seed[:]
    for k in range((n - q + h) // h):
        idx = q - 1 + k * h
        if idx < n:
            solution[idx] = 1 - solution[idx]
    return solution


def create_random_solution(pb: KnapsackProblem, target_items):
    n = pb.n
    num_selected = min(target_items, n)
    selected_positions = random.sample(range(n), num_selected)
    return [1 if i in selected_positions else 0 for i in range(n)]


def fit_to_target_items(solutions, target_items):
    for solution in solutions:
        current_items = sum(solution)
        if current_items > target_items:
            ones = [i for i, val in enumerate(solution) if val == 1]
            to_remove = min(current_items - target_items, len(ones))
            for i in random.sample(ones, to_remove):
                solution[i] = 0


def generate_diverse_solutions(pb: KnapsackProblem, target):
    n = pb.n
    hmax = min(max(2, n - 1), 10)


    pool, seen = [], set()
    seed = [0] * n

    # Phase 1: Systematic generation
    for h in range(2, min(hmax + 1, 5)):  # Limit iterations
        for q in range(1, h + 1):
            if len(pool) >= target:
                break
            solution = create_systematic_solution(seed, h, q)
            complement = [1 - x for x in solution]
            target_items = compute_target_items(pb, n)
            fit_to_target_items([solution, complement], target_items)

            for solution in [solution, complement]:
                if len(pool) >= target:
                    break
                if tuple(solution) not in seen:
                    seen.add(tuple(solution))
                    pool.append(solution)
        if len(pool) >= target:
            break

    # Phase 2: Random generation if needed
    attempts = 0
    while len(pool) < target and attempts < 100:
        attempts += 1
        num_selected = random.randint(1, min(n, compute_target_items(pb, n)))
        random_solution = create_random_solution(pb, num_selected)
        if tuple(random_solution) not in seen:
            seen.add(tuple(random_solution))
            pool.append(random_solution)

    return pool[:target]


def compute_target_items(pb: KnapsackProblem, n: int) -> int:
    max_items_estimate = estimate_capacity_items(pb)
    max_deviation = max(1, max_items_estimate // 3)
    alpha = random.randint(-max_deviation, max_deviation)
    target_items = max(1, min(n, max_items_estimate + alpha))
    return target_items


def remove_worst_items(pb: KnapsackProblem, x):
    repaired_solution = x[:]
    current_value = objective(pb, repaired_solution)
    efficiency_order_asc = ratio_order(pb, ascending=True)

    for item_idx in efficiency_order_asc:
        if is_feasible(pb, repaired_solution):
            break
        if repaired_solution[item_idx] == 1:
            repaired_solution[item_idx] = 0
            current_value -= pb.profits[item_idx]

    return repaired_solution, current_value


def repair_solution(pb: KnapsackProblem, x):
    repaired_solution, current_value = remove_worst_items(pb, x)
    return repaired_solution, current_value


def add_best_items(pb: KnapsackProblem, x, current_value):
    improved_solution = x[:]
    efficiency_order_desc = ratio_order(pb, ascending=False)

    for item_idx in efficiency_order_desc:
        if improved_solution[item_idx] == 1:
            continue
        new_weight = total_weight(pb, improved_solution) + pb.weights[item_idx]
        if new_weight <= pb.capacity:
            improved_solution[item_idx] = 1
            current_value += pb.profits[item_idx]

    return improved_solution, current_value


def improve_solution(pb: KnapsackProblem, x):
    improved_solution, current_value = repair_solution(pb, x)
    improved_solution, current_value = add_best_items(pb, improved_solution, current_value)
    return improved_solution, current_value


def hamming_distance(x, y):
    return sum(abs(xi - yi) for xi, yi in zip(x, y))

def min_hamming_dist_to_refset(s, rs_solutions):
    return min(hamming_distance(s, r) for r in rs_solutions)

def create_refset(solutions_with_values, b, b1):
    P_sorted = sorted(solutions_with_values, key=lambda t: t[1], reverse=True)

    # Select b1 best solutions by quality
    rs_solutions = [P_sorted[i][0][:] for i in range(min(b1, len(P_sorted)))]
    rs_values = [P_sorted[i][1] for i in range(min(b1, len(P_sorted)))]



    candidates = [t for t in P_sorted if tuple(t[0]) not in {tuple(r) for r in rs_solutions}]

    # Select remaining solutions to maximize diversity
    while len(rs_solutions) < b and candidates:
        best = max(candidates, key=lambda t: min_hamming_dist_to_refset(t[0], rs_solutions))
        rs_solutions.append(best[0][:])
        rs_values.append(best[1])
        candidates = [t for t in candidates if tuple(t[0]) != tuple(best[0])]

    return rs_solutions, rs_values


def combine_multiple(pb: KnapsackProblem, solutions, values):
    total_value = sum(max(0, v) for v in values)
    if total_value == 0:
        weights = [1.0 / len(values)] * len(values)
    else:
        weights = [max(0, v) / total_value for v in values]

    # Calculate weighted scores for each variable position
    scores = []
    for i in range(pb.n):
        weighted_sum = sum(solutions[j][i] * weights[j] for j in range(len(solutions)))
        score = max(0.0, min(1.0, weighted_sum))
        scores.append(score)

    # Generate trial solutions using different thresholds
    trials = []
    thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
    for threshold in thresholds:
        trial = [1 if scores[i] >= threshold else 0 for i in range(pb.n)]

        # Handle edge cases
        if sum(trial) == 0:
            high_indices = [i for i, s in enumerate(scores) if s > 0.1]
            if high_indices:
                trial[random.choice(high_indices)] = 1

        trials.append(trial)

    # Add one more trial with random threshold
    random_threshold = random.uniform(0.1, 0.9)
    random_trial = [1 if scores[i] >= random_threshold else 0 for i in range(pb.n)]
    if sum(random_trial) == 0:
        high_indices = [i for i, s in enumerate(scores) if s > 0.0]
        if high_indices:
            random_trial[random.choice(high_indices)] = 1
    trials.append(random_trial)

    return trials


def run_scatter_search(pb: KnapsackProblem, max_iter, population_size, refset_size):
    print(f"Iniciando Scatter Search: {max_iter} iteraciones, población: {population_size}, RefSet: {refset_size}")
    print("=" * 80)

    # 1. Diversification Generation
    print("Fase 1: Generación de diversidad...")
    initial_pool = generate_diverse_solutions(pb, target=population_size)

    # 2. Improvement
    print("Fase 2: Mejorando soluciones iniciales...")
    improved_pool = []
    for sol in initial_pool:
        improved_sol, value = improve_solution(pb, sol)
        improved_pool.append((improved_sol, value))

    # 3. Create initial RefSet
    print("Fase 3: Creando RefSet inicial...")
    rs_solutions, rs_values = create_refset(improved_pool, b=refset_size, b1=refset_size//2)

    best_value = max(rs_values)
    best_solution = rs_solutions[rs_values.index(best_value)]
    print(f"Mejor solución inicial: {best_value}")
    print("=" * 80)

    iterations_without_improvement = 0
    rebuild_threshold = max(10, max_iter // 20)  # Rebuild after 10 iterations or 5% of total
    total_rebuilds = 0

    # 4. Main iterative loop
    for iteration in range(max_iter):
        # Progress bar
        progress = (iteration + 1) / max_iter
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # Print progress line
        print(f"\r[{bar}] {iteration+1:4d}/{max_iter} | Mejor: {best_value:6d} | Sin mejora: {iterations_without_improvement:3d} | Rebuilds: {total_rebuilds}", end='')
        sys.stdout.flush()

        # Check if RefSet needs rebuild
        if iterations_without_improvement >= rebuild_threshold:
            # Generate new diverse solutions
            new_pool = generate_diverse_solutions(pb, target=population_size)
            new_improved_pool = []
            for sol in new_pool:
                improved_sol, value = improve_solution(pb, sol)
                new_improved_pool.append((improved_sol, value))

            # Keep best solution and rebuild RefSet with new diverse solutions
            new_improved_pool.append((best_solution, best_value))
            rs_solutions, rs_values = create_refset(new_improved_pool, b=refset_size, b1=refset_size//2)

            iterations_without_improvement = 0
            total_rebuilds += 1
            continue

        refset_changed = False

        # 5. Generate subsets and combine solutions
        for i in range(len(rs_solutions)):
            for j in range(i+1, len(rs_solutions)):
                subset_sols = [rs_solutions[i], rs_solutions[j]]
                subset_vals = [rs_values[i], rs_values[j]]

                # 6. Combination and improvement
                trials = combine_multiple(pb, subset_sols, subset_vals)

                for trial in trials:
                    improved_trial, trial_value = improve_solution(pb, trial)

                    # 7. RefSet update
                    # Check if solution is not already in RefSet
                    trial_tuple = tuple(improved_trial)
                    already_exists = any(tuple(sol) == trial_tuple for sol in rs_solutions)

                    if not already_exists and trial_value > min(rs_values):
                        worst_idx = rs_values.index(min(rs_values))
                        rs_solutions[worst_idx] = improved_trial
                        rs_values[worst_idx] = trial_value
                        refset_changed = True

        # Update best solution
        current_best = max(rs_values)
        if current_best > best_value:
            best_value = current_best
            best_solution = rs_solutions[rs_values.index(best_value)]
            iterations_without_improvement = 0
            # Print new best solution found
            print(f"\n🎯 NUEVA MEJOR SOLUCIÓN en iteración {iteration+1}: {best_value}")
            weight_used = total_weight(pb, best_solution)
            utilization = (weight_used / pb.capacity) * 100
            print(f"   Peso: {weight_used}/{pb.capacity} ({utilization:.1f}%) | Items: {sum(best_solution)}")
        elif not refset_changed:
            # No improvement in best solution AND no RefSet changes
            iterations_without_improvement += 1
        else:
            # RefSet changed but no global improvement - reset partial counter
            if iterations_without_improvement > 0:
                iterations_without_improvement = max(0, iterations_without_improvement - 1)

    print("\n" + "=" * 80)
    return best_solution, best_value