from scatter_search_tutorial import KnapsackProblem, run_scatter_search, is_feasible, total_weight
import random
import numpy as np
from instance_reader import load_instance_from_file


def load_problem_instance(instance_path: str) -> KnapsackProblem:
    """Load and create a KnapsackProblem from file."""
    profits, weights, capacity, n = load_instance_from_file(instance_path)
    return KnapsackProblem(profits=profits, weights=weights, capacity=capacity)

def main():
    """Main execution function for the tutorial."""
    # Set reproducible seeds
    random.seed(42)
    np.random.seed(42)

    instance_path = "instances_01_KP/large_scale/knapPI_3_200_1000_1"
    pb = load_problem_instance(instance_path)


    print(f"🎒 SCATTER SEARCH TUTORIAL 🎒")
    print("Iniciando búsqueda...\n")

    best_solution, best_value = run_scatter_search(pb, max_iter=1000, population_size=30, refset_size=10)

    # Display results
    print("\n" + "="*50)
    print("RESULTADOS FINALES DEL TUTORIAL")
    print("="*50)
    print(f"Best solution: {best_solution}")
    print(f"Best objective value: {best_value}")
    print(f"Total weight used: {total_weight(pb, best_solution)} / {pb.capacity}")
    print(f"Capacity utilization: {(total_weight(pb, best_solution)/pb.capacity)*100:.1f}%")
    print(f"Solution feasible: {is_feasible(pb, best_solution)}")
    selected_items = [i for i, x in enumerate(best_solution) if x == 1]
    print(f"Selected items: {selected_items}")
    print("="*40)


if __name__ == "__main__":
    main()