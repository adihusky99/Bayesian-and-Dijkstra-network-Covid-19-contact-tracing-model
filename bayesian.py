"""
COVID-19 Bayesian Network — Final Project (CS 5002)
Authors: Aditya, James, Yevgeniy

Purpose:
  How certain can we be that a person has COVID-19 given their test result,
  symptoms, and exposure history?  And who in their contact network is at risk?

Network structure (directed acyclic graph):

  Close Contact ──→ COVID-19 ←── (prior infection rate)
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
          Fever     Cough    Positive Test

  • Close Contact with a confirmed case increases the prior probability of COVID.
  • COVID-19 is the hidden variable we want to infer.
  • Fever, Cough, and a Positive Test are observable evidence nodes
    whose probabilities are conditioned on whether the person has COVID.

This mirrors the classic Sprinkler → Rain → Wet-Grass Bayesian Network
from Wikipedia (https://en.wikipedia.org/wiki/Bayesian_network#Example).

All probabilities below are approximate and drawn from published CDC / WHO
data and the project proposal.  They can be tuned as needed.
"""

# ============================================================
#  1.  DEFINE THE NETWORK PROBABILITIES
# ============================================================

# --- Root / prior nodes ---
P_CONTACT = 0.17          # P(Contact=T) — had close contact with confirmed case
P_COVID_BASE = 0.02       # P(COVID=T | Contact=F) — background infection rate (2026)

# --- P(COVID | Contact) ---
#   Contact=T  →  P(COVID)=0.25   (significant exposure)
#   Contact=F  →  P(COVID)=0.02   (background community rate)
covid_given_contact = {
    True:  0.25,
    False: 0.02,
}

# --- P(Fever | COVID) ---
fever_given_covid = {
    True:  0.78,   # ~78 % of COVID patients report fever
    False: 0.04,   # ~4 % base-rate fever from other causes
}

# --- P(Cough | COVID) ---
cough_given_covid = {
    True:  0.65,   # ~65 % of COVID patients report cough
    False: 0.08,   # ~8 % base-rate cough (allergies, cold, etc.)
}

# --- P(Positive Test | COVID) ---
#   Sensitivity (true-positive rate):  P(Test+ | COVID=T) = 0.92
#   False-positive rate:               P(Test+ | COVID=F) = 0.05
test_given_covid = {
    True:  0.92,   # sensitivity
    False: 0.05,   # false-positive rate
}


# ============================================================
#  2.  EXACT INFERENCE BY ENUMERATION
# ============================================================

def compute_joint(contact_obs=None, covid_obs=None,
                  fever_obs=None, cough_obs=None, test_obs=None):
    """
    Enumerate every combination of the 5 Boolean variables and return
    a dict  { (contact, covid, fever, cough, test): probability }.
    Any variable fixed to True/False is treated as observed evidence.
    """
    joint = {}
    for c in [True, False]:
        if contact_obs is not None and c != contact_obs:
            continue
        p_c = P_CONTACT if c else (1 - P_CONTACT)

        for v in [True, False]:
            if covid_obs is not None and v != covid_obs:
                continue
            p_v = covid_given_contact[c] if v else (1 - covid_given_contact[c])

            for f in [True, False]:
                if fever_obs is not None and f != fever_obs:
                    continue
                p_f = fever_given_covid[v] if f else (1 - fever_given_covid[v])

                for k in [True, False]:          # k = cough
                    if cough_obs is not None and k != cough_obs:
                        continue
                    p_k = cough_given_covid[v] if k else (1 - cough_given_covid[v])

                    for t in [True, False]:
                        if test_obs is not None and t != test_obs:
                            continue
                        p_t = test_given_covid[v] if t else (1 - test_given_covid[v])

                        joint[(c, v, f, k, t)] = p_c * p_v * p_f * p_k * p_t

    return joint


def infer(evidence: dict) -> dict:
    """
    Given a dict of evidence  { 'contact': T/F, 'fever': T/F, ... }
    return posterior P(True) for every node.
    Keys not in *evidence* are treated as unobserved.
    """
    joint = compute_joint(
        contact_obs=evidence.get('contact'),
        covid_obs=evidence.get('covid'),
        fever_obs=evidence.get('fever'),
        cough_obs=evidence.get('cough'),
        test_obs=evidence.get('test'),
    )
    total = sum(joint.values())

    # Marginalise
    post = {
        'contact': 0.0, 'covid': 0.0,
        'fever': 0.0, 'cough': 0.0, 'test': 0.0,
    }
    for (c, v, f, k, t), prob in joint.items():
        normed = prob / total
        post['contact'] += normed if c else 0
        post['covid']   += normed if v else 0
        post['fever']   += normed if f else 0
        post['cough']   += normed if k else 0
        post['test']    += normed if t else 0

    return post


# ============================================================
#  3.  PRETTY-PRINT QUERY RESULTS
# ============================================================

# --- ANSI colour helpers ---
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"

def _covid_bar(p: float, width: int = 30) -> str:
    """Visual probability bar for COVID probability."""
    filled = round(p * width)
    bar = "█" * filled + "░" * (width - filled)
    if p >= 0.75:
        colour = C.RED
    elif p >= 0.40:
        colour = C.YELLOW
    else:
        colour = C.GREEN
    return f"{colour}{bar}{C.RESET} {p*100:5.1f}%"

def _covid_bar_short(p: float, width: int = 20) -> str:
    """Compact bar for contact-network table."""
    filled = round(p * width)
    bar = "█" * filled + "░" * (width - filled)
    if p >= 0.75:
        colour = C.RED
    elif p >= 0.40:
        colour = C.YELLOW
    else:
        colour = C.GREEN
    return f"{colour}{bar}{C.RESET} {p*100:5.1f}%"

NODE_LABELS = {
    'contact': 'Close Contact',
    'covid':   'COVID-19',
    'fever':   'Fever',
    'cough':   'Cough',
    'test':    'Positive Test',
}

def query(evidence: dict, title: str = ""):
    """Run inference, print results nicely."""
    if title:
        print(f"  {C.CYAN}{C.BOLD}── {title} ──{C.RESET}")
    post = infer(evidence)
    for key in ['contact', 'covid', 'fever', 'cough', 'test']:
        label = NODE_LABELS[key]
        if key in evidence:
            val_str = f"{C.BOLD}{'Yes' if evidence[key] else 'No':3s}{C.RESET}  {C.DIM}(observed){C.RESET}"
            print(f"    {label:20s} = {val_str}")
        else:
            p = post[key]
            if key == 'covid':
                # Highlight the key result prominently
                print(f"    {C.BOLD}{C.WHITE}{'▶ ' + label:20s}{C.RESET}   {_covid_bar(p)}")
            else:
                print(f"    {label:20s}   P(Yes)={C.DIM}{p:.4f}{C.RESET}   P(No)={C.DIM}{1-p:.4f}{C.RESET}")
    print()
    return post


# ============================================================
#  4.  CONTACT-NETWORK RISK ASSESSMENT
# ============================================================

def _risk_label(p: float) -> tuple[str, str]:
    """Returns (plain label, coloured label) for alignment-safe printing."""
    if p >= 0.75:
        return "HIGH  ", f"{C.RED}{C.BOLD}HIGH  {C.RESET}"
    elif p >= 0.40:
        return "MEDIUM", f"{C.YELLOW}{C.BOLD}MEDIUM{C.RESET}"
    else:
        return "LOW   ", f"{C.GREEN}{C.BOLD}LOW   {C.RESET}"

def contact_risk(person_evidence: dict, contacts: list[dict]):
    """
    Given evidence about one person, estimate the risk for each
    person in their contact network.

    *contacts* is a list of dicts, each with at least a 'name' key
    and optional symptom/test evidence of their own.
    """
    person_post = infer(person_evidence)
    p_infected = person_post['covid']

    print(f"  {C.BOLD}Index person P(COVID){C.RESET} = {_covid_bar(p_infected)}\n")
    print(f"  {C.BOLD}{'Contact':<12s}  {'Own evidence':<22s}  {'Risk':<6s}  P(COVID){C.RESET}")
    print(f"  {C.DIM}{'-'*12}  {'-'*22}  {'-'*6}  {'-'*27}{C.RESET}")

    for ct in contacts:
        name = ct['name']
        ct_evidence = {k: v for k, v in ct.items() if k != 'name'}

        ct_evidence_exposed = {**ct_evidence, 'contact': True}
        ct_evidence_safe    = {**ct_evidence, 'contact': False}

        post_exposed = infer(ct_evidence_exposed)
        post_safe    = infer(ct_evidence_safe)

        p_covid_contact = (p_infected * post_exposed['covid']
                         + (1 - p_infected) * post_safe['covid'])

        evidence_str = ', '.join(f"{k}={'Y' if v else 'N'}"
                                 for k, v in ct_evidence.items()) or 'none'
        bar = _covid_bar_short(p_covid_contact)
        _, coloured_risk = _risk_label(p_covid_contact)
        print(f"  {name:<12s}  {evidence_str:<22s}  {coloured_risk}  {bar}")
    print()


# ============================================================
#  5.  DEMO — ANSWERING THE PROJECT QUESTIONS
# ============================================================

if __name__ == "__main__":
    print(C.CYAN + C.BOLD + "=" * 60 + C.RESET)
    print(C.CYAN + C.BOLD + "  COVID-19 Bayesian Network — Proof of Concept" + C.RESET)
    print(C.CYAN + C.BOLD + "=" * 60 + C.RESET)
    print()

    # ---- Q1: Marginal (no evidence) ----
    print(C.BOLD + "Q1) No evidence — baseline probabilities:" + C.RESET)
    query({})

    # ---- Q2: Person tests positive — how certain are they infected? ----
    print(C.BOLD + "Q2) Person tests POSITIVE — what is P(COVID)?" + C.RESET)
    query({'test': True}, "Positive test only")

    # ---- Q3: Positive test + fever + cough ----
    print(C.BOLD + "Q3) Positive test + fever + cough:" + C.RESET)
    query({'test': True, 'fever': True, 'cough': True},
          "Strong symptomatic evidence")

    # ---- Q4: Positive test but NO symptoms — false positive? ----
    print(C.BOLD + "Q4) Positive test, no fever, no cough (possible false positive?):" + C.RESET)
    query({'test': True, 'fever': False, 'cough': False},
          "Asymptomatic positive test")

    # ---- Q5: Close contact + symptoms, no test yet ----
    print(C.BOLD + "Q5) Close contact + fever, no test taken yet:" + C.RESET)
    query({'contact': True, 'fever': True},
          "Exposure + fever, awaiting test")

    # ---- Q6: Negative test despite symptoms ----
    print(C.BOLD + "Q6) Fever + cough but NEGATIVE test:" + C.RESET)
    query({'fever': True, 'cough': True, 'test': False},
          "Symptomatic but tested negative")

    # ---- Q7: Contact-network risk assessment ----
    print(C.CYAN + C.BOLD + "=" * 60 + C.RESET)
    print(C.CYAN + C.BOLD + "  CONTACT-NETWORK RISK ASSESSMENT" + C.RESET)
    print(C.CYAN + C.BOLD + "=" * 60 + C.RESET)
    print()
    print("Scenario: index person tested positive with fever & cough.")
    print("Who in their network is most at risk?\n")

    index_evidence = {'test': True, 'fever': True, 'cough': True}

    contacts = [
        {'name': 'Alice',   'fever': True,  'cough': False},
        {'name': 'Bob',     'fever': False, 'cough': False},
        {'name': 'Charlie', 'fever': True,  'cough': True},
        {'name': 'Diana',   'test': False},
        {'name': 'Eve'},  # no info at all
    ]

    contact_risk(index_evidence, contacts)

    print(C.CYAN + C.BOLD + "=" * 60 + C.RESET)
    print(C.CYAN + C.BOLD + "  INTERPRETATION" + C.RESET)
    print(C.CYAN + C.BOLD + "=" * 60 + C.RESET)
    print("""
  • A positive test alone raises P(COVID) substantially, but it is
    NOT 100%  due to false positives (specificity < 1).
  • Adding symptom evidence (fever, cough) pushes the probability
    much higher — the network combines multiple signals.
  • A positive test with NO symptoms is less certain — could be a
    false positive.
  • For the contact network, people showing symptoms (Charlie) are
    at highest risk; people who tested negative (Diana) are lowest.
  • This demonstrates how Bayesian inference lets us reason under
    uncertainty and prioritise public-health responses.
""")