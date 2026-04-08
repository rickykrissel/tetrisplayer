import numpy as np
from game import Game
from genetic import Genetic_AI, NUM_FEATURES
import random
import pandas as pd


# Crossover
def cross(a1, a2):
    """
    Arithmetic (blend) crossover: new gene = α·g1 + (1-α)·g2
    where α is drawn uniformly from [0, 1] per gene.
    This produces offspring anywhere in the convex hull of the two parents,
    which is strictly better than the broken proportionality crossover.
    Mutation is applied automatically.
    """
    alpha = np.random.uniform(0, 1, size=len(a1.genotype))
    new_genotype = alpha * a1.genotype + (1 - alpha) * a2.genotype
    return Genetic_AI(genotype=new_genotype, mutate=True)


# Fitness
def compute_fitness(agent, num_trials):
    """
    Fitness = mean lines cleared over num_trials games.
    Lines cleared is a better signal than pieces dropped because it
    directly measures the objective the game rewards.
    """
    scores = []
    for t in range(num_trials):
        game = Game('genetic', agent=agent)
        _, rows_cleared = game.run_no_visual()
        scores.append(rows_cleared)
        print(f'    trial {t + 1}/{num_trials}: {rows_cleared} lines')
    return float(np.mean(scores))


# Selection

def rank_select(sorted_pop, num_parents):
    """
    Rank-based selection: agent ranked i (0 = worst) gets selection
    probability proportional to (i + 1).  This avoids the problem where
    one super-fit agent dominates early generations.
    """
    n = len(sorted_pop)
    ranks = np.arange(1, n + 1, dtype=float)   # worst=1, best=n
    probs = ranks / ranks.sum()
    indices = np.random.choice(n, size=num_parents, replace=False, p=probs)
    return [sorted_pop[i] for i in indices]


# Main training loop

def run_X_epochs(
    num_epochs=15,
    num_trials=5,
    pop_size=50,
    num_elite=5,
    survival_rate=0.35,
    initial_noise_sd=0.3,
    noise_decay=0.97,          # multiply noise_sd by this each epoch
    logging_file='training',
):
    """
    Evolutionary loop with:
      - rank-based parent selection
      - arithmetic blend crossover
      - adaptive (decaying) mutation
      - fitness = lines cleared
    """
    headers = ['epoch', 'avg_lines', 'top_lines', 'elite_lines',
               'avg_gene', 'top_gene', 'noise_sd']
    rows = []

    noise_sd = initial_noise_sd
    population = [Genetic_AI() for _ in range(pop_size)]

    for epoch in range(num_epochs):
        print(f'\n=== Epoch {epoch + 1}/{num_epochs}  (noise_sd={noise_sd:.4f}) ===')

        # --- Fitness evaluation ---
        for n, agent in enumerate(population):
            print(f'  Agent {n + 1}/{pop_size}')
            agent.fit_score = compute_fitness(agent, num_trials)

        total_fitness = sum(a.fit_score for a in population)
        for agent in population:
            agent.fit_rel = agent.fit_score / total_fitness if total_fitness > 0 else 0.0

        # --- Sort descending by fitness ---
        sorted_pop = sorted(population, reverse=True)

        # --- Stats ---
        avg_lines   = total_fitness / pop_size
        top_agent   = sorted_pop[0]
        elite_lines = np.mean([a.fit_score for a in sorted_pop[:num_elite]])

        print(f'  avg={avg_lines:.1f}  top={top_agent.fit_score:.1f}  elite={elite_lines:.1f}')
        rows.append([
            epoch + 1, avg_lines, top_agent.fit_score, elite_lines,
            np.mean([a.genotype for a in population], axis=0).tolist(),
            top_agent.genotype.tolist(),
            noise_sd,
        ])

        pd.DataFrame(rows, columns=headers).to_csv(
            f'data/{logging_file}.csv', index=False
        )

        # --- Build next generation ---
        next_gen = []

        # Elite: copy top agents unchanged
        for i in range(num_elite):
            next_gen.append(Genetic_AI(genotype=sorted_pop[i].genotype, mutate=False))

        # Select parent pool via rank-based selection
        num_parents = round(pop_size * survival_rate)
        parents = rank_select(sorted_pop, num_parents)

        # Fill remainder via crossover + mutation
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = cross(p1, p2)
            # Override noise_sd with current adaptive value
            child.genotype = (
                child.genotype
                * np.random.normal(1.0, noise_sd, size=NUM_FEATURES)
            )
            next_gen.append(child)

        # --- Adaptive mutation decay ---
        noise_sd = max(0.05, noise_sd * noise_decay)

        population = next_gen

    print('\nTraining complete.')
    return rows


if __name__ == '__main__':
    run_X_epochs(num_epochs=15, num_trials=5, pop_size=50, num_elite=5)

        


