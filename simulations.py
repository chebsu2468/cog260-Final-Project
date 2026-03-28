
"""
SIMULATION — ACS Integration & Experiment
=========================================================

COGNITIVE CONCEPT:
    This implements the ACTION-CENTERED SUBSYSTEM (ACS) of Clarion.
    The ACS is where behavior actually happens — it takes the outputs from 
    BOTH levels and combines them to select an action.

    The processing pipeline:
        Stimulus → Bottom Level (automatic) → activations
        Stimulus → Top Level (controlled)   → activations
                                               ↓
                                    ACS INTEGRATION
                            (weighted sum of both levels)
                                               ↓
                                    BOLTZMANN SELECTION
                            (convert activations → probabilities → pick one)
                                               ↓
                                          RESPONSE

    Key parameters:
        W_bottom: weight of bottom-level contribution (default 1.0)
            → automatic processing is effortless, always full strength
        W_top: weight of top-level contribution (default 0.3)
            → controlled processing is EFFORTFUL, reduced influence
            → this asymmetry is WHY the automatic pathway can win
        temperature: Boltzmann selection noise (default 0.5)
            → low = deterministic, high = random

INTERFACE CONTRACT:
    Main functions for the notebook:
        - run_single_trial(ink_color, word_content, ...) -> dict
        - run_experiment(n_trials, n_runs, ...) -> list of dicts
        - plot_results(results) -> matplotlib figures
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from bottom_level import get_bottom_activations, COLORS, WORDS, RESPONSES
from top_level import get_top_activations


# ============================================================
# ACS INTEGRATION
# Combine bottom-level and top-level activations
# ============================================================

def integrate_activations(bottom_acts, top_acts, w_bottom=1.0, w_top=0.3):
    """
    Combine activations from both levels (ACS integration).

    Cognitive meaning:
        Behavior is determined by BOTH automatic and controlled processes.
        W_bottom and W_top control how much each level influences the final response.

        W_bottom = 1.0: automatic processing runs at full strength (effortless)
        W_top = 0.3: controlled processing has limited influence (effortful)

        This asymmetry is the core of the Stroop effect:
        - Automatic word reading is strong but sometimes WRONG (for the task)
        - Controlled color naming is correct but WEAK
        - On incongruent trials, the strong wrong signal competes with the weak right signal

    Formula:
        combined(response) = W_bottom * bottom(response) + W_top * top(response)

    Args:
        bottom_acts: dict from bottom_level.get_bottom_activations()
        top_acts: dict from top_level.get_top_activations()
        w_bottom: float — weight of implicit/automatic level
        w_top: float — weight of explicit/controlled level

    Returns:
        dict: combined activation for each response
    """
    combined = {}
    for response in RESPONSES:
        combined[response] = (
            w_bottom * bottom_acts.get(response, 0.0)
            + w_top * top_acts.get(response, 0.0)
        )
    return combined


# ============================================================
# BOLTZMANN ACTION SELECTION
# Convert activations to probabilities and sample a response
# ============================================================

def boltzmann_select(activations, temperature=0.5):
    """
    Select a response using Boltzmann (softmax) distribution.

    Cognitive meaning:
        Even when one response has higher activation, selection isn't 
        perfectly deterministic. Noise in the system means the "wrong" 
        response can sometimes win — especially when activations are close.

        Temperature controls how noisy selection is:
        - Low T (→ 0): almost always picks the highest activation (decisive)
        - High T (→ ∞): approaches random selection (confused/distracted)

    Formula:
        P(response) = exp(activation / T) / sum(exp(all_activations / T))

    Args:
        activations: dict of {response: activation_value}
        temperature: float — selection noise parameter

    Returns:
        str: the selected response (e.g., "say_blue")
    """
    responses = list(activations.keys())
    acts = np.array([activations[r] for r in responses])

    # Softmax with temperature (subtract max for numerical stability)
    scaled = acts / temperature
    scaled -= np.max(scaled)
    exp_acts = np.exp(scaled)
    probabilities = exp_acts / np.sum(exp_acts)

    # Sample from the probability distribution
    selected_idx = np.random.choice(len(responses), p=probabilities)
    return responses[selected_idx]


# ============================================================
# TRIAL GENERATOR
# Create Stroop stimuli
# ============================================================

def generate_trial(trial_type, rng=None):
    """
    Generate a single Stroop trial stimulus.

    Args:
        trial_type: "congruent", "incongruent", or "neutral"
        rng: numpy random generator (for reproducibility)

    Returns:
        dict: {"ink_color": str, "word_content": str, "correct_response": str, "trial_type": str}
    """
    if rng is None:
        rng = np.random.default_rng()

    ink_color = rng.choice(COLORS)
    correct_response = f"say_{ink_color}"

    if trial_type == "congruent":
        word_content = ink_color  # word matches ink
    elif trial_type == "incongruent":
        other_colors = [c for c in COLORS if c != ink_color]
        word_content = rng.choice(other_colors)  # word conflicts with ink
    elif trial_type == "neutral":
        word_content = "neutral"  # no color word
    else:
        raise ValueError(f"Unknown trial type: {trial_type}")

    return {
        "ink_color": ink_color,
        "word_content": word_content,
        "correct_response": correct_response,
        "trial_type": trial_type,
    }


# ============================================================
# RUN A SINGLE TRIAL
# Full pipeline: stimulus → both levels → integrate → select → measure
# ============================================================

def run_single_trial(ink_color, word_content, correct_response, trial_type,
                     word_weight=1.0, ink_weight=0.6,
                     w_bottom=1.0, w_top=0.3, temperature=0.5):
    """
    Run one complete Stroop trial through the full architecture.

    Processing pipeline:
        1. Bottom level: automatic feature → response activation
        2. Top level: goal + ink → rule-based response activation
        3. ACS integration: combine both levels
        4. Boltzmann selection: pick a response
        5. Measure: accuracy and response conflict

    Args:
        ink_color: str
        word_content: str
        correct_response: str (e.g., "say_blue")
        trial_type: str
        word_weight, ink_weight: bottom-level pathway strengths
        w_bottom, w_top: integration weights
        temperature: Boltzmann selection noise

    Returns:
        dict with all trial data
    """
    # Step 1: Bottom level (implicit, automatic)
    bottom_acts = get_bottom_activations(ink_color, word_content, word_weight, ink_weight)

    # Step 2: Top level (explicit, controlled)
    top_acts = get_top_activations(ink_color)

    # Step 3: ACS integration
    combined_acts = integrate_activations(bottom_acts, top_acts, w_bottom, w_top)

    # Step 4: Boltzmann selection
    selected_response = boltzmann_select(combined_acts, temperature)

    # Step 5: Measures
    is_correct = (selected_response == correct_response)

    # Response conflict (RT proxy) from the paper:
    #   "the inverse of the difference between the two highest response activations"
    # Small gap → high conflict → slow RT; large gap → low conflict → fast RT
    sorted_acts = sorted(combined_acts.values(), reverse=True)
    activation_gap = sorted_acts[0] - sorted_acts[1]
    # Avoid division by zero; if gap is 0, activations are tied = maximum conflict
    response_conflict = 1.0 / activation_gap if activation_gap > 1e-9 else 1e6

    return {
        "trial_type": trial_type,
        "ink_color": ink_color,
        "word_content": word_content,
        "correct_response": correct_response,
        "selected_response": selected_response,
        "is_correct": is_correct,
        "bottom_activations": bottom_acts,
        "top_activations": top_acts,
        "combined_activations": combined_acts,
        "activation_gap": activation_gap,
        "response_conflict": response_conflict,
    }


# ============================================================
# RUN FULL EXPERIMENT
# Multiple runs × multiple trials × multiple conditions
# ============================================================

def run_experiment(n_trials_per_condition=50, n_runs=30,
                   word_weight=1.0, ink_weight=0.6,
                   w_bottom=1.0, w_top=0.3, temperature=0.5,
                   seed=42):
    """
    Run a complete Stroop experiment.

    Args:
        n_trials_per_condition: int — trials per condition per run
        n_runs: int — number of simulated participants
        word_weight, ink_weight: pathway strengths
        w_bottom, w_top: level integration weights
        temperature: Boltzmann noise
        seed: random seed for reproducibility

    Returns:
        list of dicts — one dict per trial with all data
    """
    rng = np.random.default_rng(seed)
    all_results = []

    trial_types = ["congruent", "incongruent", "neutral"]

    for run in range(n_runs):
        for trial_type in trial_types:
            for trial_num in range(n_trials_per_condition):
                # Generate stimulus
                trial = generate_trial(trial_type, rng)

                # Run through full architecture
                result = run_single_trial(
                    trial["ink_color"],
                    trial["word_content"],
                    trial["correct_response"],
                    trial["trial_type"],
                    word_weight=word_weight,
                    ink_weight=ink_weight,
                    w_bottom=w_bottom,
                    w_top=w_top,
                    temperature=temperature,
                )
                result["run"] = run
                result["trial_num"] = trial_num
                all_results.append(result)

    return all_results


# ============================================================
# PARAMETER SWEEP
# Vary parameters to explore the model's behavior space
# ============================================================

def parameter_sweep():
    """
    Run the experiment across multiple parameter configurations.

    This tests:
    1. Weight ratio: how pathway asymmetry affects the Stroop effect
    2. Temperature: how selection noise affects error rates
    3. W_top: how cognitive control strength affects performance

    Returns:
        dict: mapping parameter config to summary results
    """
    configs = {}

    # Vary weight ratio (word vs ink)
    for ink_w in [0.4, 0.6, 0.8]:
        results = run_experiment(word_weight=1.0, ink_weight=ink_w, n_runs=30)
        summary = summarize_results(results)
        configs[f"ink_weight={ink_w}"] = summary

    # Vary temperature
    for temp in [0.1, 0.5, 1.0]:
        results = run_experiment(temperature=temp, n_runs=30)
        summary = summarize_results(results)
        configs[f"temperature={temp}"] = summary

    # Vary W_top (cognitive control strength)
    for wt in [0.1, 0.3, 0.5, 0.7]:
        results = run_experiment(w_top=wt, n_runs=30)
        summary = summarize_results(results)
        configs[f"w_top={wt}"] = summary

    return configs


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def summarize_results(results):
    """Compute accuracy and mean conflict per condition."""
    summary = {}
    for trial_type in ["congruent", "incongruent", "neutral"]:
        trials = [r for r in results if r["trial_type"] == trial_type]
        accuracy = np.mean([r["is_correct"] for r in trials])
        mean_conflict = np.mean([r["response_conflict"] for r in trials])
        mean_gap = np.mean([r["activation_gap"] for r in trials])
        summary[trial_type] = {
            "accuracy": accuracy,
            "mean_conflict": mean_conflict,
            "mean_gap": mean_gap,
            "n_trials": len(trials),
        }

    # Stroop measures — accuracy
    summary["interference"] = (
        summary["neutral"]["accuracy"] - summary["incongruent"]["accuracy"]
    )
    summary["facilitation"] = (
        summary["congruent"]["accuracy"] - summary["neutral"]["accuracy"]
    )

    # Stroop measures — conflict (higher conflict = more competition)
    summary["conflict_interference"] = (
        summary["incongruent"]["mean_conflict"] - summary["neutral"]["mean_conflict"]
    )
    summary["conflict_facilitation"] = (
        summary["neutral"]["mean_conflict"] - summary["congruent"]["mean_conflict"]
    )

    return summary


# ============================================================
# PLOTTING
# ============================================================

def plot_accuracy(summary, title="Stroop Task: Accuracy by Condition"):
    """Bar chart of accuracy across conditions."""
    conditions = ["congruent", "neutral", "incongruent"]
    accuracies = [summary[c]["accuracy"] for c in conditions]
    colors_plot = ["#2ecc71", "#95a5a6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(conditions, accuracies, color=colors_plot, edgecolor="black")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    return fig


def plot_conflict(summary, title="Stroop Task: Response Conflict by Condition"):
    """Bar chart of response conflict (RT proxy). Higher = more competition."""
    conditions = ["congruent", "neutral", "incongruent"]
    conflicts = [summary[c]["mean_conflict"] for c in conditions]
    colors_plot = ["#2ecc71", "#95a5a6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(conditions, conflicts, color=colors_plot, edgecolor="black")
    ax.set_ylabel("Response Conflict (1 / activation gap)")
    ax.set_title(title)

    for bar, conf in zip(bars, conflicts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    return fig


def plot_parameter_effect(configs, param_name, param_values):
    """Plot how a parameter affects accuracy across conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    conditions = ["congruent", "neutral", "incongruent"]
    markers = {"congruent": "o", "neutral": "s", "incongruent": "^"}
    colors_plot = {"congruent": "#2ecc71", "neutral": "#95a5a6", "incongruent": "#e74c3c"}

    # Left panel: accuracy
    for condition in conditions:
        accs = []
        for val in param_values:
            key = f"{param_name}={val}"
            accs.append(configs[key][condition]["accuracy"])
        axes[0].plot(param_values, accs, marker=markers[condition],
                color=colors_plot[condition], label=condition, linewidth=2, markersize=8)

    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Accuracy vs {param_name}")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # Right panel: response conflict
    for condition in conditions:
        confs = []
        for val in param_values:
            key = f"{param_name}={val}"
            confs.append(configs[key][condition]["mean_conflict"])
        axes[1].plot(param_values, confs, marker=markers[condition],
                color=colors_plot[condition], label=condition, linewidth=2, markersize=8)

    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Response Conflict (1 / gap)")
    axes[1].set_title(f"Conflict vs {param_name}")
    axes[1].legend()

    plt.tight_layout()
    return fig


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if condition:
            print(f"  PASS  {test_name}")
            passed += 1
        else:
            print(f"  FAIL  {test_name}")
            if detail:
                print(f"        {detail}")
            failed += 1

    # ==========================================================
    # TEST 1: Single trial pipeline — smoke test
    # ==========================================================
    print("=== Test 1: Single trial runs without error ===\n")

    trial = run_single_trial("blue", "blue", "say_blue", "congruent")
    check("congruent trial returns dict",
        isinstance(trial, dict) and "is_correct" in trial)

    trial = run_single_trial("blue", "red", "say_blue", "incongruent")
    check("incongruent trial returns dict",
        isinstance(trial, dict) and "is_correct" in trial)

    trial = run_single_trial("blue", "neutral", "say_blue", "neutral")
    check("neutral trial returns dict",
        isinstance(trial, dict) and "is_correct" in trial)

    # ==========================================================
    # TEST 2: ACS integration math
    # ==========================================================
    print("\n=== Test 2: ACS integration produces correct weighted sum ===\n")

    bottom = {"say_red": 0.5, "say_blue": 0.3, "say_green": 0.0}
    top    = {"say_red": 0.0, "say_blue": 1.0, "say_green": 0.0}
    combined = integrate_activations(bottom, top, w_bottom=1.0, w_top=0.3)

    check("say_red = 1.0*0.5 + 0.3*0.0 = 0.50",
        abs(combined["say_red"] - 0.50) < 1e-6,
        f"got {combined['say_red']}")
    check("say_blue = 1.0*0.3 + 0.3*1.0 = 0.60",
        abs(combined["say_blue"] - 0.60) < 1e-6,
        f"got {combined['say_blue']}")
    check("say_green = 1.0*0.0 + 0.3*0.0 = 0.00",
        abs(combined["say_green"] - 0.00) < 1e-6,
        f"got {combined['say_green']}")

    # ==========================================================
    # TEST 3: Response conflict measure
    # ==========================================================
    print("\n=== Test 3: Response conflict = 1/gap (higher = more conflict) ===\n")

    # Large gap -> low conflict
    trial_easy = run_single_trial("blue", "blue", "say_blue", "congruent")
    # Small gap -> high conflict
    trial_hard = run_single_trial("blue", "red", "say_blue", "incongruent")

    check("incongruent conflict > congruent conflict",
        trial_hard["response_conflict"] > trial_easy["response_conflict"],
        f"incongruent={trial_hard['response_conflict']:.3f}, "
        f"congruent={trial_easy['response_conflict']:.3f}")

    # ==========================================================
    # TEST 4: Full experiment — Stroop predictions
    # ==========================================================
    print("\n=== Test 4: Full experiment (50 trials * 30 runs) ===\n")

    results = run_experiment(n_trials_per_condition=50, n_runs=30, seed=42)
    summary = summarize_results(results)

    for condition in ["congruent", "neutral", "incongruent"]:
        s = summary[condition]
        print(f"  {condition:12s}: accuracy={s['accuracy']:.2%}, "
              f"conflict={s['mean_conflict']:.2f}, gap={s['mean_gap']:.3f}")

    # Prediction 1: congruent accuracy > neutral > incongruent
    check("accuracy: congruent > neutral",
        summary["congruent"]["accuracy"] > summary["neutral"]["accuracy"],
        f"{summary['congruent']['accuracy']:.2%} vs {summary['neutral']['accuracy']:.2%}")
    check("accuracy: neutral > incongruent",
        summary["neutral"]["accuracy"] > summary["incongruent"]["accuracy"],
        f"{summary['neutral']['accuracy']:.2%} vs {summary['incongruent']['accuracy']:.2%}")

    # Prediction 2: conflict: incongruent > neutral > congruent
    check("conflict: incongruent > neutral",
        summary["incongruent"]["mean_conflict"] > summary["neutral"]["mean_conflict"],
        f"{summary['incongruent']['mean_conflict']:.2f} vs "
        f"{summary['neutral']['mean_conflict']:.2f}")
    check("conflict: neutral > congruent",
        summary["neutral"]["mean_conflict"] > summary["congruent"]["mean_conflict"],
        f"{summary['neutral']['mean_conflict']:.2f} vs "
        f"{summary['congruent']['mean_conflict']:.2f}")

    # Prediction 3: interference > facilitation
    print(f"\n  Interference (accuracy): {summary['interference']:.2%}")
    print(f"  Facilitation (accuracy): {summary['facilitation']:.2%}")
    check("interference > facilitation (accuracy)",
        summary["interference"] > summary["facilitation"],
        f"{summary['interference']:.4f} vs {summary['facilitation']:.4f}")

    print(f"\n  Interference (conflict): {summary['conflict_interference']:.2f}")
    print(f"  Facilitation (conflict): {summary['conflict_facilitation']:.2f}")
    check("interference > facilitation (conflict)",
        summary["conflict_interference"] > summary["conflict_facilitation"],
        f"{summary['conflict_interference']:.4f} vs "
        f"{summary['conflict_facilitation']:.4f}")

    # ==========================================================
    # TEST 5: W_top parameter sweep — Stroop shrinks with more control
    # ==========================================================
    print("\n=== Test 5: Increasing W_top reduces Stroop effect ===\n")

    interference_by_wtop = {}
    for wt in [0.1, 0.3, 0.5, 0.7]:
        res = run_experiment(n_trials_per_condition=50, n_runs=30,
                             w_top=wt, seed=42)
        s = summarize_results(res)
        interference_by_wtop[wt] = s["interference"]
        print(f"  W_top={wt}: interference={s['interference']:.2%}")

    check("interference at W_top=0.1 > W_top=0.7",
        interference_by_wtop[0.1] > interference_by_wtop[0.7],
        f"{interference_by_wtop[0.1]:.4f} vs {interference_by_wtop[0.7]:.4f}")

    # ==========================================================
    # Generate plots
    # ==========================================================
    print("\n=== Generating plots ===\n")

    fig1 = plot_accuracy(summary)
    fig1.savefig("results/accuracy_by_condition.png", dpi=150)
    print("  Saved results/accuracy_by_condition.png")

    fig2 = plot_conflict(summary)
    fig2.savefig("results/conflict_by_condition.png", dpi=150)
    print("  Saved results/conflict_by_condition.png")

    plt.close("all")

    # ==========================================================
    # SUMMARY
    # ==========================================================
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed == 0:
        print("All Stroop predictions confirmed.")
    print(f"{'='*50}")