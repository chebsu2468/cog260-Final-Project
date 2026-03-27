from datetime import timedelta
from pyClarion.pyClarion import (
    Agent,  # Base class for defining pyClarion agents
    ChunkStore,  # A process that maintains chunk data
    Input,
    Pool,
    BottomUp,
    # A process that combines activations from different sources, with
    #    optional source weighting via parameters.
    TopDown,  # A process that computes top-down activations
    Choice,
    Chunk
    # A process that implements a Thurstonian choice model (Case V)
)

from pyClarion.pyClarion.components.layers import Layer

from pyClarion.pyClarion import (
    NumDict,  # Numerical dictionary class. This is how pyClarion does math.
    Event,  # Represents simulation events.
    Priority  # Events have priority values to help specify what happens when
)
from pyClarion.pyClarion.knowledge import (Root, ChunkFamily, DataFamily,
                                           AtomFamily, BusFamily, Atoms, Atom, Buses, Bus,
                                           ChunkFamily,
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
    bu: BottomUp
    asn: Layer
    pool_o: Pool
    out: Choice

    def __init__(self, name: str, root: ModelKeyspace, sd: float = 1, f: float = 1) -> None:
        super().__init__(name, root)
        ks = root
        name = self.name

        with self:
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

        self.ipt >> self.pool_i
        self.pool_i >> self.asn
        self.pool_i >> self.td
        self.td >> self.bu
        (self.bu, self.asn) >> self.pool_o
        (self.ipt, self.pool_o) >> self.pool_i
        self.pool_i >> self.asn
        self.pool_i >> self.td
        self.td >> self.bu
        (self.bu, self.asn) >> self.pool_o
        self.pool_o >> self.out

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
    text_red: Binary
    text_green: Binary
    text_blue: Binary
    color_red: Binary
    color_green: Binary
    color_blue: Binary


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
               + b.main.wm ** (d.color_red.y, d.text_red.y, d.text_green.y, d.text_blue.y),

        green := "green" ^
                 + b.main.wm ** (d.color_green.y, d.text_red.y, d.text_green.y, d.text_blue.y),

        blue := "blue" ^
                + b.main.wm ** (d.color_blue.y, d.text_red.y, d.text_green.y, d.text_blue.y),

        "color_red" ^
        + red
        + b.main.wm ** d.color_red.y
        - b.main.wm ** (d.color_blue.y, d.color_green.y),

        "color_green" ^
        + green
        + b.main.wm ** d.color_green.y
        - b.main.wm ** (d.color_blue.y, d.color_red.y),

        "color_blue" ^
        + blue
        + b.main.wm ** d.color_blue.y
        - b.main.wm ** (d.color_green.y, d.color_red.y),

        "text_red" ^
        + b.main.wm ** d.text_red.y,

        "text_green" ^
        + b.main.wm ** d.text_green.y,

        "text_blue" ^
        + b.main.wm ** d.text_blue.y
    ]


if __name__ == "__main__":
    root = ModelKeyspace(ModelData)
    model = Model("model", root, sd=3e-1)
    chunks = init_chunks(root)
    model.system.schedule(model.cs.encode(*chunks))
    model.system.run_all()

    (red, green, blue, color_red, color_green, color_blue, text_green, text_blue, text_red) = chunks
    with model.asn.weights[0].mutable() as d:
        d[~color_red * ~red] = 1
        d[~color_green * ~green] = 1
        d[~color_blue * ~blue] = 1
        d[~text_red * ~red] = 1
        d[~text_green * ~green] = 1
        d[~text_blue * ~blue] = 1

    with model.asn.bias[0].mutable() as d:
        d[~model.cs.c.nil] = 1
    stimuli = [color_red, text_red, color_blue]

    cycle_1i = {model.ipt.name: 1.0, model.pool_o.name: 0.0}
    cycle_1o = {model.bu.name: 1.0, model.asn.name: 0.0}
    cycle_2i = {model.ipt.name: 0.0, model.pool_o.name: 1.0}
    cycle_2o = {model.bu.name: 0.0, model.asn.name: 1.0}

    for stim in stimuli:
        print("Stimulus: ", ~stim)
        model.system.schedule(model.pool_i.set_params(**cycle_1i))
        model.system.schedule(model.pool_o.set_params(**cycle_1o))
        model.system.schedule(model.ipt.send({~stim: 1.0}))
        while model.system.queue:
            event = model.system.advance()
            if event.source == model.ipt.send:
                print(event.describe())
                model.system.schedule(model.begin_cycle())
            elif event.source == model.end_cycle:
                print("Similarities:")
                print(model.pool_o.main[0])
        model.system.schedule(model.pool_i.set_params(**cycle_2i))
        model.system.schedule(model.pool_o.set_params(**cycle_2o))
        model.system.schedule(model.begin_cycle())
        while model.system.queue:
            event = model.system.advance()
            if event.source == model.end_cycle:
                print("Category Activations:")
                print(model.pool_o.main[0])
                model.system.schedule(model.out.trigger())
            elif event.source == model.out.select:
                print(event.describe())
                print("Response: ", model.out.poll()[~model.cs.c])
                print()
