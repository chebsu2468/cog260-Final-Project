"""
BOTTOM LEVEL — Implicit/Subsymbolic Processing (pyClarion Implementation)
==========================================================================

COGNITIVE CONCEPT:
    This implements the IMPLICIT (bottom) level of Clarion's two-level architecture
    using pyClarion's native BottomUp activation process.

    In the Stroop task:
    - Word reading is AUTOMATIC (weight = 1.0, strong pathway)
    - Ink color naming is LESS AUTOMATIC (weight = 0.6, weaker pathway)

    The bottom-up activation formula (from pyClarion docs):
        s_c = sum(w_ci * x_i) / f(w)
    where f(w) = 1 + |w|^2 = 1 + sum(w^2)

INTERFACE CONTRACT:
    Other modules should use:
        - create_bottom_level(word_weight, ink_weight) -> StroopBottomLevel
        - StroopBottomLevel.get_activations(ink_color, word_content) -> dict
        - get_bottom_activations(ink_color, word_content, ...) -> dict  (standalone)
        - COLORS, RESPONSES constants
"""

from pyClarion import Atom, Atoms, Agent, Input, ChunkStore, BottomUp
from pyClarion.knowledge import Buses, Bus, BusFamily, DataFamily, Root, DVPairs


# ============================================================
# CONSTANTS
# ============================================================

COLORS = ["red", "blue", "green"]
WORDS = ["red", "blue", "green", "neutral"]
RESPONSES = ["say_red", "say_blue", "say_green"]


# ============================================================
# KEYSPACE DEFINITION
# ============================================================

class InkColor(Atoms):
    """Ink color features — what color the word is printed in."""
    red: Atom
    blue: Atom
    green: Atom


class WordContent(Atoms):
    """Word content features — what the word says."""
    red: Atom
    blue: Atom
    green: Atom


class Main(Buses):
    """Data buses — channels for activation to flow through."""
    input: Bus


class StroopBuses(BusFamily):
    main: Main


class StroopData(DataFamily):
    ink: InkColor
    word: WordContent


class StroopRoot(Root):
    b: StroopBuses
    d: StroopData


# ============================================================
# AGENT DEFINITION
# Assembles pyClarion components: Input → BottomUp → ChunkStore
# Same pattern as the WCST Tutorial 0, adapted for Stroop.
# ============================================================

class StroopBottomAgent[R: Root, D: DVPairs](Agent):
    """
    A pyClarion agent with bottom-up activation for the Stroop task.

    Components:
        ipt: receives stimulus features (ink color + word content)
        chunks: stores response chunks (say_red, say_blue, say_green)
        bu: computes bottom-up activation from features to chunks
    """
    root: R
    ipt: Input
    chunks: ChunkStore[D]
    bu: BottomUp[D]

    def __init__(self, name: str, root: R, f: DataFamily, d: D) -> None:
        super().__init__(name, root)
        self.root = root
        with self:
            self.ipt = Input(f"{name}.ipt", d)
            self.chunks = ChunkStore(f"{name}.chunks", c=f, d=d)
            self.bu = self.ipt >> self.chunks.bottom_up(f"{name}.bu")


# ============================================================
# BOTTOM LEVEL CLASS
# ============================================================

class StroopBottomLevel:
    """
    Bottom-level implicit processing for the Stroop task.

    Creates a pyClarion agent with response chunks that have
    asymmetric weights: word features weighted more strongly than
    ink features, reflecting the automaticity of reading.

    Usage:
        bottom = StroopBottomLevel(word_weight=1.0, ink_weight=0.6)
        activations = bottom.get_activations("blue", "red")
        # returns: {"say_red": 0.4237, "say_blue": 0.2542, "say_green": 0.0}
    """

    def __init__(self, word_weight: float = 1.0, ink_weight: float = 0.6):
        self.word_weight = word_weight
        self.ink_weight = ink_weight

        # Create pyClarion agent
        self.root = StroopRoot()
        self.agent = StroopBottomAgent(
            "agent", self.root, self.root.d,
            (self.root.b.main, self.root.d)
        )

        # Handles to keyspace
        main = self.agent.root.b.main
        ink = self.agent.root.d.ink
        word = self.agent.root.d.word

        # -------------------------------------------------------
        # RESPONSE CHUNKS WITH ASYMMETRIC WEIGHTS
        # This is the core mechanism of automaticity.
        #
        # Each chunk = one possible response ("say red", "say blue", "say green")
        # Each chunk has TWO weighted connections:
        #   - ink pathway:  ink_weight (0.6) — weaker, less practiced
        #   - word pathway: word_weight (1.0) — stronger, automatic
        #
        # The asymmetry (1.0 vs 0.6) encodes that word reading is more
        # automatic than color naming. This is what produces Stroop interference.
        # -------------------------------------------------------
        chunk_defs = [
            "say_red" ^
            + ink_weight * main.input ** ink.red
            + word_weight * main.input ** word.red,

            "say_blue" ^
            + ink_weight * main.input ** ink.blue
            + word_weight * main.input ** word.blue,

            "say_green" ^
            + ink_weight * main.input ** ink.green
            + word_weight * main.input ** word.green,
        ]

        # Encode chunks — this computes and stores the bottom-up weights
        self.agent.system.schedule(self.agent.chunks.encode(*chunk_defs))
        self.agent.run_all()

        # Store references for building stimuli later
        self._main = main
        self._ink_atoms = {"red": ink.red, "blue": ink.blue, "green": ink.green}
        self._word_atoms = {"red": word.red, "blue": word.blue, "green": word.green}

    def get_activations(self, ink_color: str, word_content: str) -> dict:
        """
        Compute bottom-level activations for a given stimulus.

        Feeds stimulus into pyClarion's BottomUp process, which applies:
            s_c = sum(w_ci * x_i) / (1 + sum(w^2))

        Args:
            ink_color: "red", "blue", or "green"
            word_content: "red", "blue", "green", or "neutral"

        Returns:
            dict: {"say_red": float, "say_blue": float, "say_green": float}
        """
        main = self._main

        # Build stimulus: set active features
        stimulus = + main.input ** self._ink_atoms[ink_color]

        if word_content != "neutral":
            stimulus = stimulus + main.input ** self._word_atoms[word_content]

        # Feed to pyClarion and run bottom-up activation
        self.agent.system.schedule(self.agent.ipt.send(stimulus))
        self.agent.run_all()

        # Extract activations from pyClarion's NumDict
        data = self.agent.bu.main[0]
        activations = {resp: 0.0 for resp in RESPONSES}
        for key, value in data.d.items():
            chunk_name = str(key).split(":")[-1]
            if chunk_name in RESPONSES:
                activations[chunk_name] = value

        return activations


# ============================================================
# STANDALONE FUNCTION
# simulation.py calls this directly.
# ============================================================

_default_bottom_level = None


def get_bottom_activations(ink_color: str, word_content: str,
                           word_weight: float = 1.0,
                           ink_weight: float = 0.6) -> dict:
    """
    Compute bottom-level activations using pyClarion.

    Args:
        ink_color: "red", "blue", or "green"
        word_content: "red", "blue", "green", or "neutral"
        word_weight: word pathway strength (default 1.0)
        ink_weight: ink pathway strength (default 0.6)

    Returns:
        dict: {"say_red": float, "say_blue": float, "say_green": float}
    """
    global _default_bottom_level

    if (_default_bottom_level is None
            or _default_bottom_level.word_weight != word_weight
            or _default_bottom_level.ink_weight != ink_weight):
        _default_bottom_level = StroopBottomLevel(word_weight, ink_weight)

    return _default_bottom_level.get_activations(ink_color, word_content)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__": 
    print("=== Bottom Level Activation Tests (pyClarion) ===\n")

    bottom = StroopBottomLevel(word_weight=1.0, ink_weight=0.6)

    # Congruent: word "BLUE" in blue ink
    result = bottom.get_activations("blue", "blue")
    print(f"Congruent (blue/blue):    {result}")

    # Incongruent: word "RED" in blue ink
    result = bottom.get_activations("blue", "red")
    print(f"Incongruent (blue/red):   {result}")

    # Neutral: no color word, blue ink
    result = bottom.get_activations("blue", "neutral")
    print(f"Neutral (blue/neutral):   {result}")

    print("\n--- Expected pattern: ---")
    print("  Congruent:   one high value (both pathways converge)")
    print("  Incongruent: two competing values (pathways conflict)")
    print("  Neutral:     one moderate value (only ink pathway active)")

    print("\n--- Math verification: ---")
    denom = 1 + 0.6**2 + 1.0**2
    print(f"  f(w) = 1 + 0.6^2 + 1.0^2 = {denom:.2f}")
    print(f"  Incongruent say_red:  word_w/f(w) = 1.0/{denom:.2f} = {1.0/denom:.4f}")
    print(f"  Incongruent say_blue: ink_w/f(w)  = 0.6/{denom:.2f} = {0.6/denom:.4f}")
    print(f"  Congruent say_blue:   (ink_w + word_w)/f(w) = 1.6/{denom:.2f} = {1.6/denom:.4f}")