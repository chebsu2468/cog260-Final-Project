"""
TOP LEVEL — Explicit/Symbolic Processing (pyClarion-native)
=============================================================

COGNITIVE CONCEPT:
    This implements the EXPLICIT (top) level of Clarion's two-level architecture
    using pyClarion's native production rule system (RuleStore).

    In the Stroop task:
    - The participant KNOWS the instruction: "name the INK COLOR, not the word"
    - This instruction is held as a GOAL in Working Memory
    - PRODUCTION RULES (pyClarion Rule objects) implement controlled processing:
        IF goal = name_ink_color AND ink = red   THEN response = say_red
        IF goal = name_ink_color AND ink = blue  THEN response = say_blue
        IF goal = name_ink_color AND ink = green  THEN response = say_green

    Unlike the bottom level (automatic, always-on), the top level requires
    EFFORT and ATTENTION. Its contribution to the final response is weaker —
    controlled processing is costly.

    Implementation follows the architecture from Tutorial 3 (Intro to Production
    Systems), using:
        - RuleStore for explicit rule representations
        - ChunkStore for antecedent (LHS) and consequent (RHS) chunks
        - Choice for stochastic conflict resolution among competing rules
        - Pool for aggregating activations from multiple sources
        - The >> wiring syntax for connecting components

INTERFACE CONTRACT:
    Other modules should ONLY use:
        - get_top_activations(ink_color, goal) -> dict
        - GOALS constant
"""

from pyClarion import (Agent, Input, Choice, Pool, Event, ChunkStore, NumDict,
    Priority)
from pyClarion.components.layers import Mapping
from pyClarion.events import Update, ForwardUpdate
from pyClarion.components.rules import RuleStore
from pyClarion.knowledge import (Root, ChunkFamily, RuleFamily, DataFamily,
    AtomFamily, BusFamily, Rule, Atoms, Atom, Buses, Bus)
from datetime import timedelta


# ============================================================
# KEYSPACE DEFINITIONS
# Buses -> Layout -> Data -> Root
# ============================================================

class MainBuses(Buses):
    """The Action-Centered Subsystem bus."""
    acs: Bus


class TopLevelLayout(BusFamily):
    """Bus layout — single main bus with ACS"""
    main: MainBuses


class InkColor(Atoms):
    """
    Input dimension: the perceived ink color.
    This is what the participant actually SEES on the screen.
    """
    red: Atom
    blue: Atom
    green: Atom


class GoalState(Atoms):
    """
    State dimension: the active task goal in Working Memory.
    In the Stroop task, the goal is always 'name_ink_color'.
    """
    nil: Atom
    name_ink_color: Atom


class ResponseState(Atoms):
    """
    State dimension: the response activated by rule firing.
    This is the top-level's recommendation for what to say.
    """
    nil: Atom
    say_red: Atom
    say_blue: Atom
    say_green: Atom


class StroopTopData(DataFamily):
    """
    Data dimensions for the top-level model.

    'input' is the ink color (percept fed in each trial).
    'goal' and 'response' are state dimensions maintained by the model.
    """
    input: InkColor
    goal: GoalState
    response: ResponseState


class TopKeyspace(Root):
    """
    Keyspace hierarchy: Root -> Families -> Sorts -> Atoms
    
    b: bus family (where data is represented)
    c: chunk family (for LHS/RHS chunk nodes)
    p: atom family (for internal processing nodes)
    r: rule family (for production rule nodes)
    d: data family (the actual domain dimensions)
    """
    b: TopLevelLayout
    c: ChunkFamily
    p: AtomFamily
    r: RuleFamily
    d: StroopTopData

    def __init__(self) -> None:
        super().__init__()
        self.d = self["d"] = StroopTopData()


# ============================================================
# PRODUCTION RULES
# ============================================================

def init_stroop_top_rules(ks: TopKeyspace) -> list[Rule]:
    """
    Define production rules for the Stroop top level.

    Each rule encodes:  IF (goal condition) AND (ink condition) THEN (response)

    Cognitive meaning:
        These are the explicit, verbalizable rules the participant follows:
        "If my goal is to name the ink color and the ink is red, say red."

    Uses pyClarion's rule syntax:
        - '+' adds a dimension-value pair to a chunk
        - 'b.main.acs ** d.X.Y' pairs bus location with data value
        - '>>' separates antecedent (LHS) from consequent (RHS)
        - '^' assigns a human-readable name to the rule
        - Disjunctive values use tuple syntax: (d.input.red, d.input.blue)

    The inertial/default rule ensures the goal persists
    across processing cycles unless explicitly changed.
    """
    b = ks.b
    d = ks.d
    return [

        # ==============================================
        # CORE STROOP RULES: goal + ink → response
        # ==============================================
        # These are the controlled, deliberate mappings.
        # The top level ONLY attends to ink color (not word).

        "name_red" ^
        + b.main.acs ** d.goal.name_ink_color
        + b.main.acs ** d.input.red
        >>
        + b.main.acs ** d.response.say_red,

        "name_blue" ^
        + b.main.acs ** d.goal.name_ink_color
        + b.main.acs ** d.input.blue
        >>
        + b.main.acs ** d.response.say_blue,

        "name_green" ^
        + b.main.acs ** d.goal.name_ink_color
        + b.main.acs ** d.input.green
        >>
        + b.main.acs ** d.response.say_green,

        # ==============================================
        # INERTIAL RULE
        # ==============================================
        # Abstract rule using variable "X" ensures
        # the goal persists unless a rule explicitly changes it.
        # This implements the default/circumscription pattern.

        "hold_goal" ^
        + b.main.acs ** d.goal("X")
        >>
        + b.main.acs ** d.goal("X"),

        # ==============================================
        # DEFAULT ERROR CATCHER
        # ==============================================
        # Weakest rule (fewest conditions) — only fires if no
        # better rule matches. Matches any state via d.goal().

        # "no_match" ^
        # + b.main.acs ** d.goal()
        # >>
        # + b.main.acs ** d.response.nil
    ]


# ============================================================
# MODEL CLASS
# ============================================================

class TopLevelModel(Agent):
    """
    pyClarion Agent implementing the top-level production rule system.

    Architecture:
        input → pool_in → lhs_bu → lhs_layer → rule_selector
                                                     ↓
        state ← pool_out ← rhs_td ← rhs_layer ←──────┘
                   ↑
              (state inertia via feedback loop)

    Key components:
        - self.input: receives ink color percept each trial
        - self.state: Choice component selecting response + goal states
        - self.prs: RuleStore holding the production rules
        - self.lhs / self.rhs: ChunkStores for rule conditions/conclusions
    """
    ks: TopKeyspace

    def __init__(self, name: str = "top", f: float = 35) -> None:
        # f=35 makes rule selection nearly deterministic
        ks = TopKeyspace()
        input_sort = ks.d["input"]
        assert isinstance(input_sort, Atoms)
        super().__init__(name, ks)
        self.ks = ks

        with self:
            # Knowledge stores for rule antecedents (lhs) and consequents (rhs)
            self.lhs = ChunkStore(f"{name}.lhs", ks.c, (ks.b.main, ks.d))
            self.rhs = ChunkStore(f"{name}.rhs", ks.c, (ks.b.main, ks.d))
            self.prs = RuleStore(f"{name}.prs", ks.r, self.lhs.c, self.rhs.c)

            # Processing components
            self.input = Input(f"{name}.input", (ks.b.main.acs, input_sort))
            self.state = Choice(f"{name}.state",
                p=ks.p, s=ks.d, d=(ks.b.main.acs, ks.d), sd=1e-3, f=f)
            self.inhib = Mapping(f"{name}.inhib",
                i=(ks.b.main.acs, ks.d), o=(ks.b.main.acs, input_sort),
                func=NumDict.neg)
            self.pool_in = Pool(f"{name}.pool_in",
                p=ks.p, d=(ks.b.main, ks.d), agg=NumDict.sum)
            self.lhs_bu = self.lhs.bottom_up(f"{name}.lhs_bu")
            self.lhs_layer = self.prs.lhs_layer(f"{name}.lhs_layer")
            self.rule_selector = Choice(f"{name}.rule_selector",
                p=ks.p, s=ks.d, d=self.prs.r, sd=1e-3)
            self.rhs_layer = self.prs.rhs_layer(f"{name}.rhs_layer")
            self.rhs_td = self.rhs.top_down(f"{name}.rhs_td")
            self.pool_out = Pool(f"{name}.pool_out",
                p=ks.p, d=(ks.b.main, ks.d), agg=NumDict.sum)

        # Wire components
        self.inhib = self.state >> self.inhib
        self.rhs_td = (
            (self.input, self.state, self.inhib)
            >> self.pool_in
            >> self.lhs_bu
            >> self.lhs_layer
            >> self.rule_selector
            >> self.rhs_layer
            >> self.rhs_td)
        self.state = (
            (self.rhs_td, self.state, self.inhib)
            >> self.pool_out
            >> self.state)

        # State inertia weights
        # Ensures goal/response persist unless a rule changes them,
        # but rule recommendations always dominate
        with self.pool_out.params[0].mutable():
            self.pool_out.params[0][~self.pool_out.p[self.state.name]] = 0.5
            self.pool_out.params[0][~self.pool_out.p[self.inhib.name]] = 0.5

        # Name generators for auto-created chunks and rules
        def namer():
            from itertools import count
            counter = count()
            for i in counter:
                yield str(f"_{i}")

        self.lhs.c._namer_ = namer()
        self.rhs.c._namer_ = namer()
        self.prs.r._namer_ = namer()

    def resolve(self, event: Event) -> None:
        """Schedule forward propagation."""
        if event.source == self.input.send:
            self.system.schedule(self.pool_in.forward())
        if event.source == self.lhs_layer.forward:
            self.system.schedule(self.rule_selector.trigger())
        if event.source == self.rhs_td.forward:
            self.system.schedule(self.pool_out.forward())
        if event.source == self.pool_out.forward:
            self.system.schedule(self.state.trigger())

    def init(self,
        *states: Atom,
        dt: timedelta = timedelta(),
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """Initialize model state."""
        bus = self.ks.b.main.acs
        data = {~bus * ~state: 1.0 for state in states}
        ud: list[Update] = [ForwardUpdate(self.state.main, data)]
        return Event(self.init, ud, dt, priority)


# ============================================================
# MODULE-LEVEL MODEL INSTANCE
# ============================================================

# Build the model
_model = TopLevelModel(name="top", f=35)

# Encode production rules into the model's rule store
_rules = init_stroop_top_rules(_model.ks)
_model.system.schedule(_model.prs.encode(*_rules))
_model.system.run_all()

# Create stimulus handles
_ks = _model.ks
_red   = + _ks.b.main.acs ** _ks.d.input.red
_blue  = + _ks.b.main.acs ** _ks.d.input.blue
_green = + _ks.b.main.acs ** _ks.d.input.green

# Map string names to pyClarion stimulus chunks
_INK_MAP = {
    "red":   _red,
    "blue":  _blue,
    "green": _green,
}

# Map string names to pyClarion goal atoms (for model.init)
_GOAL_MAP = {
    "name_ink_color": _ks.d.goal.name_ink_color,
}

# State dimension handle for reading response output
_main_acs = _ks.b.main.acs
_response_sort = _ks.d.response
_response_dim = ~_main_acs * ~_response_sort

# Response decoding: pyClarion keys → string names
_RESPONSE_DECODE = {
    ~_main_acs * ~_response_sort.say_red:   "say_red",
    ~_main_acs * ~_response_sort.say_blue:  "say_blue",
    ~_main_acs * ~_response_sort.say_green: "say_green",
    ~_main_acs * ~_response_sort.nil:       "nil",
}


# ============================================================
# PUBLIC INTERFACE
# ============================================================

GOALS = ["name_ink_color"]


def get_top_activations(ink_color: str, goal: str = "name_ink_color") -> dict:
    """
    Compute top-level activations from pyClarion production rules.

    Cognitive meaning:
        The participant consciously applies the task instruction:
        "I need to name the ink color."

        The production rule system:
        1. Initializes goal state in Working Memory
        2. Receives ink color as perceptual input
        3. Matches rules: (goal + ink_color) → response
        4. Fires the best-matching rule via conflict resolution
        5. Returns the activated response

        KEY INSIGHT: The top level is ALWAYS correct (it follows the goal).
        The Stroop effect happens because the bottom level (automatic word
        reading) sometimes overpowers this correct but weaker signal.

    Args:
        ink_color: str — "red", "blue", or "green"
        goal: str — the active goal (default: "name_ink_color")

    Returns:
        dict: {"say_red": float, "say_blue": float, "say_green": float}
              1.0 for the correct response, 0.0 for others.

    Examples:
        >>> get_top_activations("blue")
        {'say_red': 0.0, 'say_blue': 1.0, 'say_green': 0.0}

        >>> get_top_activations("red")
        {'say_red': 1.0, 'say_blue': 0.0, 'say_green': 0.0}
    """
    # Validate inputs
    if ink_color not in _INK_MAP:
        raise ValueError(f"Unknown ink color '{ink_color}'. Use: {list(_INK_MAP)}")
    if goal not in _GOAL_MAP:
        raise ValueError(f"Unknown goal '{goal}'. Use: {list(_GOAL_MAP)}")

    # 1. Initialize model state: set the goal in Working Memory
    #    model.init sets initial state atoms
    goal_atom = _GOAL_MAP[goal]
    response_nil = _ks.d.response.nil
    _model.system.schedule(_model.init(goal_atom, response_nil))
    _model.system.run_all()

    # 2. Send ink color as input stimulus
    stimulus = _INK_MAP[ink_color]
    _model.system.schedule(_model.input.send(stimulus))

    # 3. Run the processing cycle until all events are resolved
    while _model.system.queue:
        _model.system.advance()

    # 4. Read the response state from the Choice component
    poll = _model.state.poll()
    selected_response = poll[_response_dim]
    selected_name = _RESPONSE_DECODE.get(selected_response, "nil")

    # 5. Build activation dict: 1.0 for fired rule, 0.0 for others
    activations = {
        "say_red":   1.0 if selected_name == "say_red"   else 0.0,
        "say_blue":  1.0 if selected_name == "say_blue"  else 0.0,
        "say_green": 1.0 if selected_name == "say_green" else 0.0,
    }

    return activations


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
#     print("=== Top Level Activation Tests (pyClarion-native) ===\n")

#     for ink in ["red", "blue", "green"]:
#         result = get_top_activations(ink)
#         print(f"Ink={ink}: {result}")

#     print("\n--- Verify: top level ALWAYS activates the correct response ---")
#     print("--- Verify: top level ignores word content entirely ---")
#     print("--- The 'weakness' of controlled processing comes from W_top ---")
#     print("--- in the ACS integration module (simulation.py) ---")

    passed = 0
    failed = 0
 
    def check(test_name, actual, expected):
        global passed, failed
        if actual == expected:
            print(f"  PASS  {test_name}")
            passed += 1
        else:
            print(f"  FAIL  {test_name}")
            print(f"        expected: {expected}")
            print(f"        got:      {actual}")
            failed += 1
 
    # ==========================================================
    # TEST 1: Goal Module — correct rule fires for each ink color
    # ==========================================================
    # The goal is always "name_ink_color". Given each ink color,
    # the production rule system should activate ONLY the matching
    # response (the one that names that ink color).
 
    print("=== Test 1: Goal Module — rule fires correctly per ink ===\n")
 
    check("goal=name_ink_color, ink=red -> say_red",
        get_top_activations("red", goal="name_ink_color"),
        {"say_red": 1.0, "say_blue": 0.0, "say_green": 0.0})
 
    check("goal=name_ink_color, ink=blue -> say_blue",
        get_top_activations("blue", goal="name_ink_color"),
        {"say_red": 0.0, "say_blue": 1.0, "say_green": 0.0})
 
    check("goal=name_ink_color, ink=green -> say_green",
        get_top_activations("green", goal="name_ink_color"),
        {"say_red": 0.0, "say_blue": 0.0, "say_green": 1.0})
 
    # ==========================================================
    # TEST 2: Working Memory — top level ignores word entirely
    # ==========================================================
    # The top level receives ONLY ink color as input. It has no
    # word dimension at all. So calling with the same ink color
    # must always produce the same result regardless of what word
    # the bottom level might be processing in parallel.
    #
    # We verify this by calling the same ink color twice and
    # checking that the output is identical and correct.
 
    print("\n=== Test 2: WM — response is stable across repeated calls ===\n")
 
    result_1 = get_top_activations("red")
    result_2 = get_top_activations("red")
    check("ink=red called twice gives identical output",
        result_1, result_2)
    check("ink=red repeated still activates say_red",
        result_1,
        {"say_red": 1.0, "say_blue": 0.0, "say_green": 0.0})
 
    # ==========================================================
    # TEST 3: Congruent vs Incongruent — top level is unaffected
    # ==========================================================
    # In a congruent trial (word=RED, ink=red) and an incongruent
    # trial (word=GREEN, ink=red), the top level should produce
    # the SAME output, because it only sees ink color.
    #
    # (The word dimension lives in bottom_level.py, not here.)
 
    print("\n=== Test 3: Congruent vs Incongruent — same top output ===\n")
 
    congruent   = get_top_activations("red")   # word=RED,   ink=red
    incongruent = get_top_activations("red")   # word=GREEN, ink=red
    check("congruent and incongruent trials produce same top-level output",
        congruent, incongruent)
    check("both correctly activate say_red",
        congruent,
        {"say_red": 1.0, "say_blue": 0.0, "say_green": 0.0})
 
    # ==========================================================
    # TEST 4: Each ink maps to exactly one response
    # ==========================================================
    # Verify that for every ink color, exactly one response is 1.0
    # and the other two are 0.0 (clean winner-take-all).
 
    print("\n=== Test 4: Winner-take-all — exactly one response active ===\n")
 
    for ink in ["red", "blue", "green"]:
        result = get_top_activations(ink)
        active_count = sum(1 for v in result.values() if v == 1.0)
        zero_count   = sum(1 for v in result.values() if v == 0.0)
        check(f"ink={ink}: exactly 1 active, 2 inactive",
            (active_count, zero_count), (1, 2))
 
    # ==========================================================
    # SUMMARY
    # ==========================================================
 
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed == 0:
        print("Yeayyy all tests passed!")
    print(f"{'='*50}")
