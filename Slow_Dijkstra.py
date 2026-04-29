import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime

'''
Contact Network
Authors: Aditya

Purpose:
This network is meant to imitate a small sample size of people (200) and group
them into clusters. Clusters can be randomly assigned from 10-30 nodes, and
each node will have anywhere from 1-12 connections.

To better mimic real-world scenarios, connections will be heavily weighted to
keep to each cluster, however on occasion connections may form outside of
clusters; this mirrors people who would stay in a relatively close grouping
(work places, towns, states, countries, etc...) but may occasionally have
interactions outside their main group

We then effectively apply Dijkstra's algorithm to map the spread of covid
across the network
'''

# ============================================================
#  BAYESIAN NETWORK PROBABILITIES (from bayesian.py)
# ============================================================
P_CONTACT = 0.17
covid_given_contact = {True: 0.25, False: 0.02}
fever_given_covid   = {True: 0.78, False: 0.04}
cough_given_covid   = {True: 0.65, False: 0.08}
test_given_covid    = {True: 0.92, False: 0.05}

P_SYMPTOM_GIVEN_COVID = 0.50

# ============================================================
#  DUAL OUTPUT — console + text file
# ============================================================
output_lines = []

def out(line=""):
    print(line)
    output_lines.append(line)

def compute_joint(contact_obs=None, covid_obs=None,
                  fever_obs=None, cough_obs=None, test_obs=None):
    joint = {}
    for c in [True, False]:
        if contact_obs is not None and c != contact_obs: continue
        p_c = P_CONTACT if c else (1 - P_CONTACT)
        for v in [True, False]:
            if covid_obs is not None and v != covid_obs: continue
            p_v = covid_given_contact[c] if v else (1 - covid_given_contact[c])
            for f in [True, False]:
                if fever_obs is not None and f != fever_obs: continue
                p_f = fever_given_covid[v] if f else (1 - fever_given_covid[v])
                for k in [True, False]:
                    if cough_obs is not None and k != cough_obs: continue
                    p_k = cough_given_covid[v] if k else (1 - cough_given_covid[v])
                    for t in [True, False]:
                        if test_obs is not None and t != test_obs: continue
                        p_t = test_given_covid[v] if t else (1 - test_given_covid[v])
                        joint[(c, v, f, k, t)] = p_c * p_v * p_f * p_k * p_t
    return joint

def infer(evidence: dict) -> dict:
    joint = compute_joint(
        contact_obs=evidence.get('contact'),
        covid_obs=evidence.get('covid'),
        fever_obs=evidence.get('fever'),
        cough_obs=evidence.get('cough'),
        test_obs=evidence.get('test'),
    )
    total = sum(joint.values())
    post = {'contact': 0.0, 'covid': 0.0, 'fever': 0.0, 'cough': 0.0, 'test': 0.0}
    for (c, v, f, k, t), prob in joint.items():
        normed = prob / total
        post['contact'] += normed if c else 0
        post['covid']   += normed if v else 0
        post['fever']   += normed if f else 0
        post['cough']   += normed if k else 0
        post['test']    += normed if t else 0
    return post

def p_covid_for_contact(index_p_covid: float, ct_evidence: dict) -> float:
    ev_exposed = {**ct_evidence, 'contact': True}
    ev_safe    = {**ct_evidence, 'contact': False}
    post_exposed = infer(ev_exposed)
    post_safe    = infer(ev_safe)
    return (index_p_covid * post_exposed['covid']
            + (1 - index_p_covid) * post_safe['covid'])

# ============================================================
#  CONTACT DATA (from bayesian.py)
# ============================================================
index_evidence = {'test': True, 'fever': True, 'cough': True}
index_post = infer(index_evidence)
index_p_covid = index_post['covid']

contacts_data = [
    {'name': 'Alice',   'fever': True,  'cough': False},
    {'name': 'Bob',     'fever': False, 'cough': False},
    {'name': 'Charlie', 'fever': True,  'cough': True},
    {'name': 'Diana',   'test': False},
    {'name': 'Eve'},
]

contact_probs = {}
for ct in contacts_data:
    name = ct['name']
    ev = {k: v for k, v in ct.items() if k != 'name'}
    contact_probs[name] = p_covid_for_contact(index_p_covid, ev)

# ============================================================
#  CONTACT GRAPH
# ============================================================
n = 200
nodes = list(range(1, n + 1))
random.shuffle(nodes)

clusters = []
i = 0
while i < n:
    size = random.randint(10, 30)
    clusters.append(nodes[i:i+size])
    i += size

node_cluster = {v: ci for ci, c in enumerate(clusters) for v in c}
num_clusters = len(clusters)

cluster_centers = {}
R = 30
for i in range(num_clusters):
    angle = (math.pi / 2) - (2 * math.pi * i / num_clusters)
    cluster_centers[i] = (R * math.cos(angle), R * math.sin(angle))

palette = [
    "#1f77ff", "#ff4b4b", "#2ecc71", "#9b59b6", "#f39c12",
    "#00c2c7", "#e84393", "#16a085", "#34495e", "#d35400",
]
colors = {v: palette[node_cluster[v] % len(palette)] for v in nodes}

ROLE_COLORS = {"connector": "#FFD166", "bridge": "#C2185B"}

graph = {v: set() for v in nodes}
edges = []

def bell_degree():
    return max(1, min(10, int(random.gauss(5.5, 1.8))))

def is_adjacent(a, b):
    return abs(a - b) == 1 or abs(a - b) == num_clusters - 1

for c in clusters:
    for v in c:
        target = bell_degree()
        attempts = 0
        while len(graph[v]) < target and attempts < 15:
            attempts += 1
            u = random.choice(c)
            if u != v and u not in graph[v]:
                graph[v].add(u)
                graph[u].add(v)
                edges.append((v, u))

cross_edges = []
max_cross = max(3, int(len(edges) * 0.03))
attempts = 0
while len(cross_edges) < max_cross and attempts < 3000:
    attempts += 1
    v, u = random.choice(nodes), random.choice(nodes)
    if v == u or node_cluster[v] == node_cluster[u]:
        continue
    if u in graph[v]:
        continue
    prob = 0.6 if is_adjacent(node_cluster[v], node_cluster[u]) else 0.1
    if random.random() > prob:
        continue
    graph[v].add(u)
    graph[u].add(v)
    edges.append((v, u))
    cross_edges.append((v, u))

pos = {}
for ci, cluster in enumerate(clusters):
    cx, cy = cluster_centers[ci]
    for v in cluster:
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(6, 12)
        pos[v] = [cx + r * math.cos(angle), cy + r * math.sin(angle)]

degree = {v: len(graph[v]) for v in nodes}

def external_neighbors(v):
    return [u for u in graph[v] if node_cluster[u] != node_cluster[v]]

def external_ratio(v):
    d = degree[v]
    return len(external_neighbors(v)) / d if d > 0 else 0

def external_clusters(v):
    return len({node_cluster[u] for u in external_neighbors(v)})

def node_type(v):
    r = external_ratio(v)
    c = external_clusters(v)
    if r > 0.5 or c >= 3: return "bridge"
    elif r > 0.2:        return "connector"
    else:                return "normal"

def node_size(v):
    return 15 + (degree[v] ** 1.4)

def relax(iterations=100):
    for _ in range(iterations):
        forces = {v: [0, 0] for v in pos}
        for v in pos:
            for u in pos:
                if v == u: continue
                dx = pos[v][0] - pos[u][0]
                dy = pos[v][1] - pos[u][1]
                dist = math.hypot(dx, dy) + 1e-5
                strength = 0.05 if node_cluster[v] == node_cluster[u] else 0.01
                f = strength / dist
                forces[v][0] += dx * f
                forces[v][1] += dy * f
        for v, u in edges:
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            f = 0.02 if node_cluster[v] == node_cluster[u] else 0.006
            forces[v][0] += dx * f
            forces[v][1] += dy * f
            forces[u][0] -= dx * f
            forces[u][1] -= dy * f
        for v in pos:
            cx, cy = cluster_centers[node_cluster[v]]
            dx = cx - pos[v][0]
            dy = cy - pos[v][1]
            forces[v][0] += dx * 0.01
            forces[v][1] += dy * 0.01
        for v in pos:
            pos[v][0] += forces[v][0]
            pos[v][1] += forces[v][1]

relax()

# ============================================================
#  ASSIGN NAMES TO RANDOM NODES
# ============================================================
named_node_ids = random.sample(nodes, 5)
node_name = {}
name_node = {}
for name, nid in zip([c['name'] for c in contacts_data], named_node_ids):
    node_name[nid] = name
    name_node[name] = nid

# ============================================================
#  ROLL FOR POSITIVES
# ============================================================
node_infected = {v: False for v in nodes}
node_step = {}
node_source = {}
node_symptoms = {}

out("=" * 60)
out("  INITIAL ROLL")
out("=" * 60)
initial_positives = []
for name in ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']:
    nid = name_node[name]
    roll = round(random.random(), 5)
    p = contact_probs[name]
    positive = roll <= p
    status = "POSITIVE" if positive else "negative"
    out(f"  {name:<8s}  P(COVID)={p:.5f}  roll={roll:.5f}  -> {status}")
    if positive:
        node_infected[nid] = True
        node_step[nid] = 0
        node_source[nid] = name
        node_symptoms[nid] = True
        initial_positives.append(name)

if not initial_positives:
    out("\n  No initial positives — summoning MIGHTY SORCERER TIM")
    available = [v for v in nodes if v not in node_name]
    tim_node = random.choice(available)
    node_name[tim_node] = "MIGHTY SORCERER TIM"
    name_node["MIGHTY SORCERER TIM"] = tim_node
    node_infected[tim_node] = True
    node_step[tim_node] = 0
    node_source[tim_node] = "MIGHTY SORCERER TIM"
    node_symptoms[tim_node] = True
    initial_positives.append("MIGHTY SORCERER TIM")

out(f"\n  Initial positives: {initial_positives}")
out("=" * 60)

# ============================================================
#  PLAN THE SPREAD — per-index BFS, saved as animation waves
# ============================================================
# For each index (alphabetical), do a full BFS against the entire graph.
# Build an animation plan that preserves per-index ordering: all of Alice's
# stage-1 wave, then her stage-2, ..., then Bob's stage-1, stage-2, ...
#
# Within each wave, classify nodes as "new" (first infection) or "override"
# (already infected, but this index reaches them with a lower stage number).
# The global node_step is updated as we go so later indexes see the minimum.

spread_plan = []     # list of dicts: {index, stage, new_nodes, override_nodes}
stage_updates = []   # (index_name, node_id, old_stage, new_stage) for overrides

for index_name in sorted(initial_positives):
    index_nid = name_node[index_name]

    # BFS from this index
    bfs_stage = {index_nid: 0}
    frontier = [index_nid]
    while frontier:
        next_frontier = []
        for v in frontier:
            for u in graph[v]:
                if u not in bfs_stage:
                    bfs_stage[u] = bfs_stage[v] + 1
                    next_frontier.append(u)
        frontier = next_frontier

    # Organize this index's reach by stage
    by_stage = {}
    for v, s in bfs_stage.items():
        if s == 0:
            continue
        by_stage.setdefault(s, []).append(v)

    # Walk each stage in order, split into new vs. override, update state
    for s in sorted(by_stage):
        new_nodes = []
        override_nodes = []
        for v in by_stage[s]:
            if not node_infected[v]:
                node_infected[v] = True
                node_step[v] = s
                node_source[v] = index_name
                node_symptoms[v] = random.random() < P_SYMPTOM_GIVEN_COVID
                new_nodes.append(v)
            elif s < node_step[v]:
                old = node_step[v]
                node_step[v] = s
                node_source[v] = index_name
                override_nodes.append((v, old))
                stage_updates.append((index_name, v, old, s))
            # else: already infected at same or lower stage -> no change

        if new_nodes or override_nodes:
            spread_plan.append({
                "index": index_name,
                "stage": s,
                "new": new_nodes,
                "override": override_nodes,
            })

# ============================================================
#  VISUALIZATION
# ============================================================
fig, ax = plt.subplots(figsize=(22, 12), dpi=100)
ax.set_autoscale_on(False)

revealed = set()
for name in initial_positives:
    revealed.add(name_node[name])

ZOOM_PAD = 40
selected_node = [None]
selected_cluster = [None]
DARK_RED = "#8B0000"

def apply_zoom():
    if selected_node[0] is not None:
        v = selected_node[0]
        cx, cy = pos[v]
        ax.set_xlim(cx - ZOOM_PAD, cx + ZOOM_PAD)
        ax.set_ylim(cy - ZOOM_PAD, cy + ZOOM_PAD)
    elif selected_cluster[0] is not None:
        ci = selected_cluster[0]
        cx, cy = cluster_centers[ci]
        ax.set_xlim(cx - ZOOM_PAD, cx + ZOOM_PAD)
        ax.set_ylim(cy - ZOOM_PAD, cy + ZOOM_PAD)
    else:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad = 10
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

def on_click(event):
    if not event.inaxes:
        selected_node[0] = None
        selected_cluster[0] = None
        draw()
        return
    x, y = event.xdata, event.ydata
    for v, (vx, vy) in pos.items():
        if (x - vx)**2 + (y - vy)**2 < 0.5:
            selected_node[0] = v
            selected_cluster[0] = node_cluster[v]
            draw()
            return
    for ci, (cx, cy) in cluster_centers.items():
        if (x - cx)**2 + (y - cy)**2 <= 15**2:
            selected_node[0] = None
            selected_cluster[0] = ci
            draw()
            return
    selected_node[0] = None
    selected_cluster[0] = None
    draw()

fig.canvas.mpl_connect("button_press_event", on_click)

# Currently-animating index's stage numbers to display (for animation realism)
# Maps node_id -> stage number to render at this point in the animation.
# Starts empty, gets filled as waves play.
displayed_step = {}
current_title = [""]

def draw():
    ax.clear()
    ax.axis("off")

    if current_title[0]:
        ax.set_title(current_title[0], fontsize=13, fontweight='bold')

    for ci, (cx, cy) in cluster_centers.items():
        ax.text(cx, cy + 16, f"C{ci}", ha='center', fontsize=10, alpha=0.7)
        ax.add_patch(Circle((cx, cy), 15, fill=False, alpha=0.2))

    for v, u in edges:
        x1, y1 = pos[v]
        x2, y2 = pos[u]
        is_cross = node_cluster[v] != node_cluster[u]
        alpha = 0.6
        lw = 0.7
        if selected_node[0] is not None:
            if v == selected_node[0] or u == selected_node[0]:
                alpha = 1
                lw = 2
            else:
                alpha = 0.05
        ax.plot([x1, x2], [y1, y2],
                color=("green" if is_cross else "gray"),
                linewidth=lw, alpha=alpha)

    for v, (x, y) in pos.items():
        size = node_size(v)
        t = node_type(v)
        is_positive = v in revealed and node_infected[v]
        face_color = DARK_RED if is_positive else colors[v]

        if t == "bridge":
            ax.scatter(x, y, s=size * 1.3, color=face_color,
                       edgecolors=ROLE_COLORS["bridge"], linewidths=2.5, alpha=1)
        elif t == "connector":
            ax.scatter(x, y, s=size, color=face_color,
                       edgecolors=ROLE_COLORS["connector"], linewidths=2, alpha=1)
        else:
            ax.scatter(x, y, s=size, color=face_color, alpha=0.95)

        if v in node_name:
            name = node_name[v]
            label = name + (" +" if is_positive else "")
            ax.annotate(label, (x, y),
                        xytext=(0, 10), textcoords="offset points",
                        ha='center', fontsize=9, fontweight='bold',
                        color=(DARK_RED if is_positive else "black"))

        if is_positive and displayed_step.get(v, 0) >= 1:
            ax.annotate(str(displayed_step[v]), (x, y),
                        xytext=(0, -3), textcoords="offset points",
                        ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')

    apply_zoom()
    plt.draw()

# ---- Initial draw with just the index positives ----
current_title[0] = "Initial positives"
draw()
plt.pause(1.5)

# ---- Animate waves in per-index order ----
current_index = None
for wave in spread_plan:
    if wave["index"] != current_index:
        current_index = wave["index"]
        current_title[0] = f"Spreading from {current_index}"
        # brief pause before the next person's spread begins
        draw()
        plt.pause(0.8)

    # Reveal new nodes at this stage
    for v in wave["new"]:
        revealed.add(v)
        displayed_step[v] = wave["stage"]

    # Update displayed step for override nodes (number drops visibly)
    for v, old in wave["override"]:
        displayed_step[v] = wave["stage"]

    draw()
    plt.pause(0.6)

current_title[0] = "Spread complete"
draw()

out("\n" + "=" * 60)
out("  SPREAD COMPLETE")
out("=" * 60)
total_infected = sum(1 for v in nodes if node_infected[v])
total_symptomatic = sum(1 for v in nodes if node_infected[v] and node_symptoms.get(v, False))
out(f"  Total infected: {total_infected} / {n}")
out(f"  Symptomatic:    {total_symptomatic} / {total_infected}")

# ============================================================
#  STAGE-UPDATE LOG
# ============================================================
if stage_updates:
    out("\n" + "=" * 60)
    out("  STAGE OVERRIDES (later index reached node faster)")
    out("=" * 60)
    for idx_name, v, old, new in stage_updates:
        name_tag = f" [{node_name[v]}]" if v in node_name else ""
        out(f"  Node {v}{name_tag}: stage {old} -> {new} (via {idx_name})")

# ============================================================
#  TEXT REPORT — CLUSTERS, CONNECTIONS, INFECTION STATUS
# ============================================================
def print_cluster_report():
    out("\n" + "=" * 70)
    out("  CLUSTER REPORT")
    out("=" * 70)

    for ci in range(num_clusters):
        cluster_nodes = sorted(clusters[ci])
        infected_in_cluster = sum(1 for v in cluster_nodes if node_infected[v])
        out(f"\nCluster C{ci}  ({len(cluster_nodes)} nodes, "
            f"{infected_in_cluster} infected)")
        out("-" * 70)

        for v in cluster_nodes:
            name_tag = f" [{node_name[v]}]" if v in node_name else ""

            if node_infected[v]:
                step = node_step.get(v, 0)
                src = node_source.get(v, "?")
                if step == 0:
                    stage = "index case"
                else:
                    stage = f"stage {step} via {src}"
                sym = "symptomatic" if node_symptoms.get(v, False) else "asymptomatic"
                status = f"  [+ INFECTED — {stage}, {sym}]"
            else:
                status = ""

            out(f"  Node {v}{name_tag}{status}")

            internal = sorted(u for u in graph[v] if node_cluster[u] == ci)
            external = sorted(
                (u for u in graph[v] if node_cluster[u] != ci),
                key=lambda u: (node_cluster[u], u)
            )

            if internal:
                conns = []
                for u in internal:
                    mark = " +" if node_infected[u] else ""
                    nm = f" [{node_name[u]}]" if u in node_name else ""
                    conns.append(f"{u}{nm}{mark}")
                out(f"      within  C{ci}: {', '.join(conns)}")

            if external:
                conns = []
                for u in external:
                    mark = " +" if node_infected[u] else ""
                    nm = f" [{node_name[u]}]" if u in node_name else ""
                    conns.append(f"{u}{nm}{mark} (C{node_cluster[u]})")
                out(f"      cross-cluster: {', '.join(conns)}")

            if not internal and not external:
                out(f"      (no connections)")

    out("\n" + "=" * 70)
    out("  SUMMARY BY STAGE")
    out("=" * 70)
    stage_counts = {}
    for v in nodes:
        if node_infected[v]:
            s = node_step.get(v, 0)
            stage_counts[s] = stage_counts.get(s, 0) + 1
    for s in sorted(stage_counts):
        label = "Index cases (stage 0)" if s == 0 else f"Stage {s}"
        out(f"  {label:<25s} {stage_counts[s]} nodes")
    out(f"  {'TOTAL INFECTED':<25s} {sum(stage_counts.values())} / {n}")

print_cluster_report()

# ============================================================
#  WRITE TO TEXT FILE
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"covid_simulation_{timestamp}.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\n  Results written to: {filename}")

plt.show()
