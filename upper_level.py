from datetime import timedelta
from pyClarion import (
    Agent,  # Base class for defining pyClarion agents
    ChunkStore,  # A process that maintains chunk data
    Input,
    Pool,  # A process that combines activations from different sources, with
    #    optional source weighting via parameters.
    TopDown,  # A process that computes top-down activations
    Choice,
    # A process that implements a Thurstonian choice model (Case V)
)
from pyClarion import (
    NumDict,  # Numerical dictionary class. This is how pyClarion does math.
    Event,  # Represents simulation events.
    Priority  # Events have priority values to help specify what happens when
    #    they are scheduled to occur at the same point in time.
)
from pyClarion.knowledge import (Root, ChunkFamily, DataFamily,
                                           AtomFamily, BusFamily, Atoms, Atom, Buses, Bus, Chunk,
                                           RuleFamily)

class MainBuses(Buses):
    wm: Bus


class ModelLayout(BusFamily):
    main: MainBuses


class ModelKeyspace[D: DataFamily](Root):
    b: ModelLayout
    c: ChunkFamily
    p: AtomFamily
    d: D

    def __init__(self, data_type: type[D]) -> None:
        super().__init__()
        self.d = self["d"] = data_type()

class Model(Agent):
    cs: ChunkStore
    ipt: Input
    pool_i: Pool
    td: TopDown
    pool_o: Pool
    out: Choice

    def __init__(self, name: str, root: ModelKeyspace, sd: int, f=1) -> None:
        super().__init__(name, root)
        ks = root
        name = self.name

        with self:
            self.cs = ChunkStore(
                f"{name}.cs",
                ks.c,
                (ks.b.main, ks.d))

            self.td = self.cs.top_down(f"{name}.td")

            self.ipt = Input(
                f"{name}.ipt",
                self.cs.c)

            self.pool_o = Pool(
                name=f"{name}.pool_o",
                p=ks.p,
                d=self.cs.c,
                agg = NumDict.sum
            )

            self.pool_i = Pool(
                name=f"{name}.pool_i",
                p=ks.p,
                d=self.cs.c,
                agg = NumDict.sum
            )

            self.out = Choice(
                f"{name}.out",
                p=ks.p,
                s=ks.d,
                d=self.cs.c,
                sd=sd,
                f=f)

            with self:

                # Input → pool_i
                self.pool_i = self.ipt >> self.pool_i

                self.td = self.pool_i >> self.td

                self.out = self.td >> self.out

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


class Binary(Atoms):
    n: Atom
    y: Atom

class ModelData(DataFamily):
    red: Binary
    green: Binary
    blue: Binary


def init_chunks(root: ModelKeyspace[ModelData]) -> list[Chunk]:
    """
    Generate and return a list of new concept chunks.

    Returns a list representing the following concepts, in order.
        BIRD, REPTILE, CHICKEN, PENGUIN, CANARY, PLATYPUS, SNAKE, CROCODILE
    """
    b = root.b
    d = root.d
    return [

        red := "red" ^
                Chunk({"red"
                       }),  # Replace this with features
        green := "green" ^
                   Chunk({"green"
                          }),  # Replace this with features
        blue := "blue" ^
                 Chunk({"blue"
                        }),
        "red" ^
        + red
        + b.main.wm ** d.red.y
        - b.main.wm ** (d.blue.y, d.green.y),

        "green" ^
        + green
        + b.main.wm ** d.green.y
        - b.main.wm ** (d.red.y, d.blue.y),

        "blue" ^
        + blue
        - b.main.wm ** (d.red.y, d.green.y),
    ]


if __name__ == "__main__":
    root = ModelKeyspace(ModelData)
    model = Model("model", root, sd=5e-1)
    chunks = init_chunks(root)
    model.system.schedule(model.cs.encode(*chunks))
    model.system.run_all()

    (red, green, blue) = chunks
    with model.asn.weights[0].mutable() as d:
        d[~red * ~red] = 1
        d[~green * ~green] = 1
        d[~blue * ~blue] = 1


    with model.asn.bias[0].mutable() as d:
        d[~model.cs.c.nil] = 1

    stimuli = [red, green, blue]
