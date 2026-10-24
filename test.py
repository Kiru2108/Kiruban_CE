import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Genetic Algorithm - TV Scheduling", layout="wide"
)

st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.write(
    "Adjust parameters on the sidebar and run the Genetic Algorithm to optimize the TV schedule."
)

# -----------------------------------------------------------------------------
# 1. SIDEBAR PARAMETERS (Assignment Requirements)
# -----------------------------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

# Crossover Rate: Default 0.8, Range 0.0 to 0.95
co_r = st.sidebar.slider(
    label="Crossover Rate (CO_R)",
    min_value=0.0,
    max_value=0.95,
    value=0.80,
    step=0.05,
)

# Mutation Rate: Default 0.02 (UI bounds 0.01 to 0.05 per assignment spec)
mut_r = st.sidebar.slider(
    label="Mutation Rate (MUT_R)",
    min_value=0.01,
    max_value=0.05,
    value=0.02,
    step=0.01,
)

pop_size = st.sidebar.number_input("Population Size", value=50, step=10)
generations = st.sidebar.number_input("Generations", value=100, step=10)


# -----------------------------------------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("modified_ratings.csv")


try:
    df_ratings = load_data()
    st.subheader("📊 Modified Rating Matrix")
    st.dataframe(df_ratings, use_container_width=True)
except Exception as e:
    st.error(
        f"Error loading 'modified_ratings.csv'. Ensure the file is pushed to your GitHub root folder. Details: {e}"
    )
    st.stop()

# Extract matrix and program list
program_names = df_ratings["Type of Program"].tolist()
rating_matrix = df_ratings.drop(columns=["Type of Program"]).values
num_programs, num_hours = rating_matrix.shape
hours_list = df_ratings.columns[1:].tolist()

# -----------------------------------------------------------------------------
# 3. GENETIC ALGORITHM CORE FUNCTIONS
# -----------------------------------------------------------------------------


def calculate_fitness(chromosome):
    """Fitness = total rating score achieved across all 18 hourly slots."""
    total_score = 0
    for hour_idx, prog_idx in enumerate(chromosome):
        total_score += rating_matrix[prog_idx, hour_idx]
    return total_score


def create_initial_population(size):
    return [
        np.random.randint(0, num_programs, size=num_hours) for _ in range(size)
    ]


def selection(population, fitnesses):
    """Tournament Selection."""
    idx1, idx2 = np.random.choice(len(population), 2, replace=False)
    return (
        population[idx1].copy()
        if fitnesses[idx1] >= fitnesses[idx2]
        else population[idx2].copy()
    )


def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, num_hours)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = np.random.randint(0, num_programs)
    return chromosome


def run_genetic_algorithm(pop_s, gens, co_rate, mut_rate):
    population = create_initial_population(pop_s)
    best_chromosome = None
    best_fitness = -1

    for g in range(gens):
        fitnesses = [calculate_fitness(ind) for ind in population]

        max_fit = max(fitnesses)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_chromosome = population[np.argmax(fitnesses)].copy()

        new_population = []
        while len(new_population) < pop_s:
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2, co_rate)
            c1 = mutate(c1, mut_rate)
            c2 = mutate(c2, mut_rate)
            new_population.extend([c1, c2])

        population = new_population[:pop_s]

    return best_chromosome, best_fitness


# -----------------------------------------------------------------------------
# 4. EXECUTION & DISPLAY
# -----------------------------------------------------------------------------
if st.button("🚀 Run Genetic Algorithm", type="primary"):
    with st.spinner("Optimizing schedule..."):
        best_schedule_indices, best_score = run_genetic_algorithm(
            pop_size, generations, co_r, mut_r
        )

    st.success(
        f"Optimization Complete! Best Total Rating Score: **{best_score:.2f}**"
    )

    scheduled_programs = [
        program_names[idx] for idx in best_schedule_indices
    ]
    ratings_achieved = [
        rating_matrix[idx, h] for h, idx in enumerate(best_schedule_indices)
    ]

    df_result = pd.DataFrame(
        {
            "Time Slot": hours_list,
            "Scheduled Program": scheduled_programs,
            "Rating Score": ratings_achieved,
        }
    )

    st.subheader("📅 Generated Optimal TV Schedule")
    st.table(df_result)
