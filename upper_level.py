from datetime import timedelta

from pyClarion.pyClarion import (
    Agent,  # Base class for defining pyClarion agents
    ChunkStore,  # A process that maintains chunk data
    Input,  # A process that allows user inputs
    Pool,  # A process that combines activations from different sources, with
    #    optional source weighting via parameters.
    TopDown,  # A process that computes top-down activations
    BottomUp,  # A process that computes bottom-up activations
    Choice  # A process that implements a Thurstonian choice model (Case V)
)
from pyClarion.pyClarion.components.layers import Layer
from pyClarion.pyClarion import (
    NumDict,  # Numerical dictionary class. This is how pyClarion does math.
    Event,  # Represents simulation events.
    Priority  # Events have priority values to help specify what happens when
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

    def __init__(self, name: str, root: ModelKeyspace, sd=1, f=1) -> None:
        super().__init__(name, root)
        ks = root
        name = self.name

        with self:
            self.cs = ChunkStore(
                f"{name}.cs",
                ks.c,
                (ks.b.main, ks.d))

            self.td = self.cs.top_down(f"{name}.td")

            self.bu = self.cs.bottom_up(f"{name}.bu")

            self.ipt = Input(
                f"{name}.ipt",
                self.cs.c)

            self.pool_o = Pool(
                name=f"{name}.pool_o",
                p=ks.p,
                d=self.cs.c,
                agg=NumDict.sum)

            self.pool_i = Pool(
                name=f"{name}.pool_i",
                p=ks.p,
                d=self.cs.c,
                agg=NumDict.sum)

            self.asn = Layer(
                f"{name}.asn",
                i=self.cs.c,
                o=self.cs.c)

            self.out = Choice(
                f"{name}.out",
                p=ks.p,
                s=ks.d,
                d=self.cs.c,
                sd=sd,
                f=f)

            with self:
                self.out = self.asn >> self.out

                self.pool_o = (
                        (self.ipt)
                        >> self.pool_i
                        >> self.td
                        >> self.pool_o
                )

                self.asn = self.pool_i >> self.asn

        def resolve(self, event: Event) -> None:

            # Start of cycle
            if event.source == self.begin_cycle:
                self.system.schedule(self.pool_i.forward())

            # After pool_i finishes → wait
            elif event.source == self.pool_i.forward:
                self.system.schedule(self.wait())

            # After waiting → propagate to output
            elif event.source == self.wait:
                self.system.schedule(self.pool_o.forward())

            # End of cycle
            elif event.source == self.pool_o.forward:
                self.system.schedule(self.end_cycle())

    def wait(self,
             dt: timedelta = timedelta(milliseconds=50),
             priority: Priority = Priority.PROPAGATION
             ) -> Event:
        """Wait a few milliseconds so that activations can propagate."""
        if dt <= timedelta():
            raise ValueError("Timedelta must be non-negative.")
        return Event(self.wait, [], dt, priority)

    def begin_cycle(self,
                    dt: timedelta = timedelta(),
                    priority: Priority = Priority.PROPAGATION
                    ) -> Event:
        return Event(self.begin_cycle, [], dt, priority)

    def end_cycle(self,
                  dt: timedelta = timedelta(),
                  priority: Priority = Priority.PROPAGATION
                  ) -> Event:
        return Event(self.end_cycle, [], dt, priority)
