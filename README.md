# COVID-19 Bayesian Network
 
**Authors:** Aditya Elayavalli
**Date:** April 2026

---

## Overview

This project models the spread and diagnosis of COVID-19 using a **Bayesian Network** — a probabilistic graphical model that represents variables and their conditional dependencies. The core question the network answers is:

> *How certain can we be that a person has COVID-19, given their test result, symptoms, and exposure history? And who in their contact network is at risk?*

The project is composed of three Python scripts:

| File | Purpose |
|------|---------|
| `bayesian.py` | Core Bayesian Network — inference engine and CLI demo |
| `Slow_Dijkstra.py` | Simulated contact graph (200 nodes) with animated spread visualization |
| `visualize.py` | Static multi-panel PNG report of all query results |

---

## Background

### What is a Bayesian Network?

A Bayesian Network is a **directed acyclic graph (DAG)** where:
- Each **node** represents a random variable (e.g., "Has COVID", "Has Fever")
- Each **directed edge** represents a conditional dependency between variables
- Each node stores a **conditional probability table (CPT)** describing how its probability depends on its parent nodes

This mirrors the classic *Sprinkler → Rain → Wet Grass* example from [Wikipedia's Bayesian network article](https://en.wikipedia.org/wiki/Bayesian_network#Example).

### Network Structure

```
  Close Contact ──→ COVID-19 ←── (prior infection rate)
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           Fever     Cough    Positive Test
```

- **Close Contact** is a root node — it has no parents and uses a fixed prior probability.
- **COVID-19** is the hidden (latent) variable we want to infer. It is conditioned on whether the person had close contact with a confirmed case.
- **Fever**, **Cough**, and **Positive Test** are observable evidence nodes whose probabilities are conditioned on whether the person has COVID-19.

### Probability Parameters

All values are approximate, drawn from CDC/WHO published data:

| Parameter | Value | Source/Notes |
|-----------|-------|--------------|
| P(Close Contact) | 0.17 | Prior — 17% chance of recent exposure |
| P(COVID \| Contact=True) | 0.25 | 25% infection rate given exposure |
| P(COVID \| Contact=False) | 0.02 | 2% background community rate |
| P(Fever \| COVID=True) | 0.78 | ~78% of COVID patients report fever |
| P(Fever \| COVID=False) | 0.04 | ~4% base-rate fever from other causes |
| P(Cough \| COVID=True) | 0.65 | ~65% of COVID patients report cough |
| P(Cough \| COVID=False) | 0.08 | ~8% base-rate cough (allergies, cold, etc.) |
| P(Test+ \| COVID=True) | 0.92 | Test sensitivity (true-positive rate) |
| P(Test+ \| COVID=False) | 0.05 | False-positive rate |

### Inference Method

The project uses **exact inference by enumeration**: every combination of all five Boolean variables is enumerated (2⁵ = 32 combinations), joint probabilities are computed, and observed evidence is used to condition (filter) the distribution. The remaining entries are normalized and marginalised to produce posterior probabilities for each node.

---

## Project Files

### `bayesian.py` — Core Inference Engine

Defines the network, implements the inference engine, and runs six demonstration queries:

1. **Baseline** — no evidence; shows prior probabilities
2. **Positive test only** — how much does a single test raise confidence?
3. **Positive test + fever + cough** — strong multi-signal evidence
4. **Positive test, no symptoms** — likely false positive?
5. **Close contact + fever, no test** — awaiting test result
6. **Symptoms, negative test** — possible false negative?

Also includes a **Contact-Network Risk Assessment** that estimates COVID probability for five named contacts (Alice, Bob, Charlie, Diana, Eve) given the index person's evidence and each contact's own symptoms.

### `Slow_Dijkstra.py` — Simulated Epidemic Spread

Builds a randomized contact graph of 200 nodes grouped into clusters of 10–30, with:
- Intra-cluster connections weighted heavily (mimicking workplaces, towns)
- Occasional inter-cluster "bridge" edges (mimicking travel or rare contacts)
- Bell-curve degree distribution (1–10 connections per node)

The five named contacts from `bayesian.py` are assigned to random nodes. Their COVID status is determined by rolling against their Bayesian posterior probabilities. The infection then spreads through the network via a **BFS (Breadth-First Search)** wave propagation that mirrors Dijkstra's algorithm logic. An **animated matplotlib visualization** shows the spread wave by wave.

Results (cluster membership, infection stage, symptoms) are saved to a timestamped `.txt` file.

### `visualize.py` — Static Report

Generates a single high-resolution PNG (`bayesian_report.png`) containing four panels:
- **Scenario bar chart** — P(COVID) across all six scenarios, colour-coded by risk level
- **DAG diagram** — visual representation of the Bayesian network structure
- **Contact-network risk chart** — bar chart comparing COVID probability for each contact
- **Key Takeaways** — text summary of the most important results

---

## Requirements

### Python Version
Python 3.10 or later (uses `list[dict]` and `tuple[str, str]` type hints).

### Dependencies

Install all dependencies with:

```bash
pip install matplotlib numpy
```

`bayesian.py` has **no external dependencies** — it uses only the Python standard library.

`contact_network.py` and `visualize.py` require:
- `matplotlib` — for graph visualization and plotting
- `numpy` — used indirectly by matplotlib

---

## How to Run

### 1. Run the Bayesian inference demo

```bash
python bayesian.py
```

This prints all six query scenarios with coloured probability bars directly to the terminal, followed by the contact-network risk table. No files are saved.

**Example output (abbreviated):**
```
Q2) Person tests POSITIVE — what is P(COVID)?
── Positive test only ──
    Close Contact          P(Yes)=0.1700   P(No)=0.8300
  ▶ COVID-19               ████████░░░░░░░░░░░░░░░░░░░░░░  53.7%
    Fever                  P(Yes)=0.4438   P(No)=0.5562
    Cough                  P(Yes)=0.3798   P(No)=0.6202
    Positive Test    = Yes  (observed)
```

### 2. Run the contact network simulation

```bash
python Slow_Dijkstra.py
```

This opens an animated matplotlib window showing the infection spreading across the 200-node graph. When the animation completes, a text report is saved:

```
covid_simulation_YYYYMMDD_HHMMSS.txt
```

**Interactive features:** Click any node in the visualization to zoom to it and highlight its connections. Click a cluster circle to zoom to that cluster. Click empty space to zoom out.

> **Note:** The simulation uses `random` without a fixed seed, so results vary each run.

### 3. Generate the static PNG report

```bash
python visualize.py
```

This saves `bayesian_report.png` in the current directory. No window is displayed (uses the `Agg` non-interactive backend).

---

## Key Results

| Scenario | P(COVID) | Interpretation |
|----------|----------|----------------|
| No evidence | ~3.4% | Background community rate |
| Positive test only | ~53.7% | Not certain — false positives matter |
| Positive test + fever + cough | ~99.5% | Near certain |
| Positive test, no symptoms | ~9% | Likely false positive |
| Close contact + fever, no test | ~87% | High risk — get tested |
| Symptoms, negative test | ~46% | Test may have missed it |

**Contact risk** (index person: test+, fever, cough):

| Contact | Evidence | P(COVID) | Risk |
|---------|----------|----------|------|
| Charlie | fever=Y, cough=Y | ~98% | HIGH |
| Alice | fever=Y, cough=N | ~76% | HIGH |
| Eve | none | ~26% | MEDIUM |
| Bob | fever=N, cough=N | ~26% | MEDIUM |
| Diana | test=N | ~2.7% | LOW |

---

## Design Decisions

- **Exact inference** was chosen over approximate (MCMC/sampling) methods because the network is small (5 binary variables) and enumeration over 32 combinations is trivially fast.
- **Cluster-weighted edges** in the contact graph reflect real-world social structure where most interactions are local.
- **BFS for spread** is a simplification — it models the *minimum contact-distance* path rather than a stochastic transmission process, but it effectively illustrates how an index case can reach all nodes.
- **Node roles** (bridge, connector, normal) are inferred from each node's ratio of external-cluster connections, highlighting superspreader candidates.

---

## File Structure

```
.
├── bayesian.py            # Bayesian network inference engine + CLI demo
├── Slow_Dijkstra.py     # Contact graph simulation + animated spread
├── visualize.py           # Static multi-panel PNG report
├── bayesian_report.png    # Generated by visualize.py
└── covid_simulation_*.txt # Generated by contact_network.py
```

---

## References

- [Bayesian network — Wikipedia](https://en.wikipedia.org/wiki/Bayesian_network)
- CDC COVID-19 symptom prevalence data
- WHO diagnostic test sensitivity/specificity guidelines
