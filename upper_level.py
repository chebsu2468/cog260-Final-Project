from pyClarion.pyClarion import (
    Agent, # Base class for defining pyClarion agents
    ChunkStore, # A process that maintains chunk data
    Input, # A process that allows user inputs
    Pool, # A process that combines activations from different sources, with
          #    optional source weighting via parameters.
    TopDown, # A process that computes top-down activations
    BottomUp, # A process that computes bottom-up activations
    Layer, # A process that implements a linear associator
    Choice # A process that implements a Thurstonian choice model (Case V)
)
from pyClarion.pyClarion import (
    NumDict, # Numerical dictionary class. This is how pyClarion does math.
    Event, # Represents simulation events.
    Priority # Events have priority values to help specify what happens when
             #    they are scheduled to occur at the same point in time.
)
from pyClarion.pyClarion.knowledge import (Root, ChunkFamily, DataFamily,
                                           AtomFamily, BusFamily, Atoms, Atom, Buses, Bus, Chunk,
                                           RuleFamily)


class MainBuses(Buses):
    acs: Bus


class ModelLayout(BusFamily):
    main: MainBuses


class ModelKeyspace[D: DataFamily](Root):
    b: ModelLayout
    c: ChunkFamily
    p: AtomFamily
    r: RuleFamily
    d: D

    def __init__(self, data_type: type[D]) -> None:
        super().__init__()
        self.d = self["d"] = data_type()


class Model(Agent):
    cs: ChunkStore
    ipt: Input
    pool_i: Pool
    td: TopDown
    asn: Layer
    pool_o: Pool
    out: Choice

    def __init__(self, name: str, root: ModelKeyspace) -> None:
        super().__init__(name, root)
        pass
