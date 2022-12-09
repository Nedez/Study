# %%
from durable.lang import *

# %%
with ruleset("test"):

    # antecedent
    @when_all(m.subject == "World")
    def say_hello(c):
        # consequent
        print(f"Hello {c.m.subject}")

    @when_any(m.subject == "Chuck", m.subject == "Norris")
    def greet_chuck_norris(c):
        print("Chuck Norris can kill two stones with one bird.")


# %%
post("test", {"subject": "World"})
post("test", {"subject": "Chuck"})
post("test", {"subject": "Norris"})


# %%
post("test", {"subject": "SAIT"})


# %%
with ruleset("animal"):
    # will be triggered by 'Kermit eats flies'
    @when_all((m.predicate == "eats") & (m.object == "flies"))
    def frog(c):
        c.assert_fact({"subject": c.m.subject, "predicate": "is", "object": "frog"})

    # will be chained after asserting 'Kermit is frog'
    @when_all((m.predicate == "is") & (m.object == "frog"))
    def green(c):
        c.assert_fact({"subject": c.m.subject, "predicate": "is", "object": "green"})

    @when_all(+m.subject)
    def output(c):
        print(f"Fact: {c.m.subject} {c.m.predicate} {c.m.object}")


# %%
assert_fact("animal", {"subject": "Kermit", "predicate": "eats", "object": "flies"})

# %%
post("animal", {"subject": "Toad", "predicate": "eats", "object": "flies"})


# %%
from nltk import Prover9, Prover9Command
from nltk.sem import Expression

read_expr = Expression.fromstring
p = Prover9()
path = r"C:\Program Files (x86)\Prover9-Mace4\bin-win32"
p._binary_location = path
assumptions = [
    read_expr(p)
    for p in [
        # for every city, if it is rainy then it is cloudy.
        "rainy(x) -> cloudy(x)",
        # calgary is rainy
        "rainy(calgary)",
    ]
]
goal = read_expr(
    # calgary is cloudy
    "cloudy(calgary)",
)
prover = Prover9Command(goal=goal, assumptions=assumptions, prover=p)
print(prover.prove())


# %%
from nltk import Prover9, Prover9Command
from nltk.sem import Expression

read_expr = Expression.fromstring
p = Prover9()
path = r"C:\Program Files (x86)\Prover9-Mace4\bin-win32"
p._binary_location = path
assumptions = [
    read_expr(p)
    for p in [
        # for every city, if it is rainy then it is cloudy.
        "all x.( rainy(x) -> cloudy(x) )",
        # calgary is rainy
        "rainy(calgary)",
    ]
]
goal = read_expr(
    # there exists a city that is cloudy
    "exists x.(cloudy(x))",
)

prover = Prover9Command(goal=goal, assumptions=assumptions, prover=p)
print(prover.prove())

# %%
