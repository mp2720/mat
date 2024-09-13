from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar, Generic, assert_never


# region FOL
class Symbol:
    def __init__(self, name: str, arity: int):
        self.name = name
        self.arity = arity

    def __repr__(self):
        return self.name


class PredicateSymbol(Symbol):
    pass


class FunctionalSymbol(Symbol):
    pass


T = PredicateSymbol("T", 0)
F = PredicateSymbol("F", 0)


class Signature:
    def __init__(self, ps: list[PredicateSymbol], fs: list[FunctionalSymbol]):
        ps += [T, F]

        ps_names = [p.name for p in ps]
        fs_names = [f.name for f in fs]

        assert len(set(ps_names + fs_names)) == len(ps_names + fs_names), "found duplicates in PS or FS"

        self.ps = dict(zip(ps_names, ps))
        self.fs = dict(zip(fs_names, fs))


class IndividualVariable:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


class TermOfFunctionalSymbol:
    def __init__(self, f: FunctionalSymbol, args: list[Term]):
        assert f.arity == len(args), "mismatched arity"
        self.f = f
        self.args = args

    def __repr__(self) -> str:
        s = repr(self.f)

        if self.f.arity != 0:
            s += "(" + ", ".join(map(str, self.args)) + ")"

        return s


Term = IndividualVariable | TermOfFunctionalSymbol
"""
Терм ::= v,
Терм ::= f1,
Терм ::= f2(t1,...,tn),

где:
* v - индивидуальная переменная,
* f1 - 0-местный функциональный символ,
* f2 - m-местный функциональный символ, причём m > 0,
* t1,...,tn - термы.
"""


class Atom:
    """
    Атом ::= p1,
    Атом ::= p2(t1,...,t2),

    где:
    * p1 - 0-местный предикатный символ,
    * p2 - m-местный предикатный символ, причём m > 0,
    * t1,...,tn - термы.
    """

    def __init__(self, p: PredicateSymbol, args: list[Term]):
        assert p.arity == len(args), "mismatched arity"
        self.p = p
        self.args = args

    def __repr__(self) -> str:
        s = repr(self.p)

        if self.p.arity != 0:
            s += "(" + ", ".join(map(str, self.args)) + ")"

        return s


class BinaryLogicalConnective(str, Enum):
    OR = '∨'
    AND = '^'
    IMPL = '⊃'
    EQ = '≡'


class FormulaWithBinConnective:
    def __init__(self, a: Formula, b: Formula, connective: BinaryLogicalConnective):
        self.a = a
        self.b = b
        self.connective = connective

    def __repr__(self) -> str:
        return f"({self.a} {self.connective.value} {self.b})"


class FormulaWithNegation:
    def __init__(self, a: Formula):
        self.a = a

    def __repr__(self) -> str:
        return f"¬{self.a}"


class Quantifier(str, Enum):
    UNIVERSAL = '∀'
    EXISTENTIAL = '∃'


class FormulaWithQuantification:
    def __init__(self, quantifier: Quantifier, var: IndividualVariable, a: Formula):
        self.quantifier = quantifier
        self.var = var
        self.a = a

    def __repr__(self) -> str:
        return f"{self.quantifier.value}{self.var.name} {self.a}"


Formula = Atom | FormulaWithBinConnective | FormulaWithNegation | FormulaWithQuantification
"""
Формула ::= a,
Формула ::= ¬A,
Формула ::= ∃A,
Формула ::= ∀A,
Формула ::= (A ∨ B),
Формула ::= (A ^ B),
Формула ::= (A ⊃ B),

где:
* a - атом,
* A и B - формулы.
"""


def parse_formula(s: str, signature: Signature) -> Formula:
    class Lexeme(Enum):
        NEGATION = 0
        QUANTIFIER = 1
        IDENTIFIER = 2
        COMMA = 3
        LEFT_PARENTHESIS = 4
        RIGHT_PARENTHESIS = 5
        BINARY_CONNECTIVE = 6

    @dataclass
    class Token:
        lexeme: Lexeme
        value: any

    token: Token | None = None

    def next_token() -> Token:
        nonlocal s, token

        lex_rules = [
            (Lexeme.NEGATION, r"^\~"),
            (Lexeme.QUANTIFIER, r"^(A|E)"),
            (Lexeme.IDENTIFIER, r"^([_A-Za-z0-9]+)"),
            (Lexeme.COMMA, r"^\,"),
            (Lexeme.LEFT_PARENTHESIS, r"^\("),
            (Lexeme.RIGHT_PARENTHESIS, r"^\)"),
            (Lexeme.BINARY_CONNECTIVE, r"^(\+|\*|\>|\=)"),
        ]

        # Remove blank from left.
        s = s.lstrip()

        for lexeme, pattern in lex_rules:
            if (m := re.search(pattern, s)) is not None:
                token = Token(lexeme, s[:m.end(0)])
                s = s[m.end(0):]
                return token

        assert not s, "invalid token"

    def parse_identifier() -> IndividualVariable | FunctionalSymbol | PredicateSymbol | None:
        if token.lexeme != Lexeme.IDENTIFIER:
            return None

        name = token.value
        next_token()

        if name in signature.fs:
            return signature.fs[name]
        elif name in signature.ps:
            return signature.ps[name]
        else:
            return IndividualVariable(name)

    def parse_args() -> list[Term]:
        """
        Args ::= ('(' Term (',' Term)* ')')?
        """

        args = []

        if token.lexeme != Lexeme.LEFT_PARENTHESIS:
            return []

        next_token()
        while True:
            t = parse_term()
            assert t is not None, "expected a term as an argument"

            args.append(t)

            if token.lexeme == Lexeme.RIGHT_PARENTHESIS:
                next_token()
                break

            assert token.lexeme == Lexeme.COMMA, "expected comma as an argument separator"
            next_token()

        return args

    def parse_term() -> Term | None:
        """
        Term ::= IV | FS | FS Args
        """

        if (ident := parse_identifier()) is not None and isinstance(ident, IndividualVariable):
            return ident

        if not isinstance(ident, FunctionalSymbol):
            return None

        return TermOfFunctionalSymbol(ident, parse_args())

    def parse_atom() -> Atom | None:
        """
        Atom ::= P | P Args
        """

        if (p := parse_identifier()) is None or not isinstance(p, PredicateSymbol):
            return None

        return Atom(p, parse_args())

    def parse_formula_() -> Formula | None:
        """
        Formula ::= Atom | '~' Formula | ('A' | 'E') LOWERCASE_NAME Formula
            | '(' Formula ('+' | '*' | '>' | '=') Formula ')'
        """

        if (a := parse_atom()) is not None:
            return a

        if token.lexeme == Lexeme.NEGATION:
            next_token()

            a = parse_formula_()
            assert a is not None, "expected a formula after the negation"

            return FormulaWithNegation(a)

        if token.lexeme == Lexeme.QUANTIFIER:
            quantifier = {
                'A': Quantifier.UNIVERSAL,
                'E': Quantifier.EXISTENTIAL
            }[token.value]

            next_token()
            assert token.lexeme == Lexeme.IDENTIFIER, "expected an individual variable after the quantifier"
            v = IndividualVariable(token.value)

            next_token()
            a = parse_formula_()
            assert a is not None, "expected a formula after the variable and quantifier"

            return FormulaWithQuantification(quantifier, v, a)

        if token.lexeme == Lexeme.LEFT_PARENTHESIS:
            next_token()

            a = parse_formula_()
            assert a is not None, "expected a formula after the '('"

            assert token.lexeme == Lexeme.BINARY_CONNECTIVE, "expected a binary connective after the '(' and formula"
            conn = {
                "+": BinaryLogicalConnective.OR,
                "*": BinaryLogicalConnective.AND,
                ">": BinaryLogicalConnective.IMPL,
                "=": BinaryLogicalConnective.EQ
            }[token.value]
            next_token()

            b = parse_formula_()
            assert b is not None, "expected a formula after the binary connective"

            next_token()

            return FormulaWithBinConnective(a, b, BinaryLogicalConnective(conn))

    next_token()
    formula = parse_formula_()
    assert not s
    return formula


_mock_signature = Signature(
    [
        PredicateSymbol("P", 1),
        PredicateSymbol("Q", 2),
        PredicateSymbol("R", 3),
        PredicateSymbol("S", 0)
    ],
    [
        FunctionalSymbol("f", 1),
        FunctionalSymbol("g", 2),
        FunctionalSymbol("h", 0)
    ]
)
assert repr(parse_formula("((S + S) + S)", _mock_signature)) == "((S ∨ S) ∨ S)"
assert repr(parse_formula("Ax P(x)", _mock_signature)) == "∀x P(x)"
assert repr(parse_formula("Ax P(f(x))", _mock_signature)) == "∀x P(f(x))"
assert repr(parse_formula("Ex Ay (Q(x, y) + (R(f(x), g(z, f(h)), h) > S))", _mock_signature)) == \
       "∃x ∀y (Q(x, y) ∨ (R(f(x), g(z, f(h)), h) ⊃ S))"

# endregion

# region Semantics

DT = TypeVar("DT")


@dataclass
class Interpretation(Generic[DT]):
    # Callable для n-арных PS и FS, где n > 0.
    ps: dict[str, Callable[[...], bool] | bool]
    fs: dict[str, Callable[[...], DT] | DT]


class Semantics(Generic[DT]):
    def __init__(
            self,
            interpretation: Interpretation[DT],
            environment: dict[str, DT],
            domain_set_to_iterate: Iterable[DT]
    ):
        self.interpretation = interpretation
        self.interpretation.ps["T"] = True
        self.interpretation.ps["F"] = False

        self.environment = environment

        self.domain_set_to_iterate = domain_set_to_iterate

    def term_to_dt(self, term: Term, bounded_vars: dict[str, DT]) -> DT:
        match term:
            case IndividualVariable():
                if term.name in bounded_vars:
                    return bounded_vars[term.name]

                if term.name in self.environment:
                    return self.environment[term.name]

                assert False, "free individual variable's value not found in the environment"

            case TermOfFunctionalSymbol():
                f = self.interpretation.fs[term.f.name]

                if not term.args:
                    return f

                args = [self.term_to_dt(arg, bounded_vars) for arg in term.args]

                return f(*args)

            case _ as never:
                assert_never(never)

    def atom_to_bool(self, atom: Atom, bounded_vars: dict[str, DT]) -> bool:
        p = self.interpretation.ps[atom.p.name]

        if not atom.args:
            return p

        args = [self.term_to_dt(arg, bounded_vars) for arg in atom.args]

        return p(*args)

    def formula_to_bool(self, formula: Formula, bounded_vars: dict[str, DT]) -> bool:
        match formula:
            case Atom():
                return self.atom_to_bool(formula, bounded_vars)

            case FormulaWithBinConnective():
                # Значение вычисляется лениво.
                a = self.formula_to_bool(formula.a, bounded_vars)

                def lazy_b():
                    return self.formula_to_bool(formula.b, bounded_vars)

                match formula.connective:
                    case BinaryLogicalConnective.OR:
                        return a or lazy_b()
                    case BinaryLogicalConnective.AND:
                        return a and lazy_b()
                    case BinaryLogicalConnective.IMPL:
                        return not a or lazy_b()
                    case BinaryLogicalConnective.EQ:
                        return a == lazy_b()
                    case _ as never:
                        assert_never(never)

            case FormulaWithNegation():
                return not self.formula_to_bool(formula.a, bounded_vars)

            case FormulaWithQuantification():
                is_universal = formula.quantifier == Quantifier.UNIVERSAL

                for x in self.domain_set_to_iterate:
                    if self.formula_to_bool(formula.a, {**bounded_vars, formula.var.name: x}) != is_universal:
                        return not is_universal

                return is_universal

            case _ as never:
                assert_never(never)


# endregion

def arithmetics_semantics_example():
    signature = Signature(
        [
            PredicateSymbol("Q", 2),
            PredicateSymbol("D", 2)
        ],
        [
            FunctionalSymbol("Plus", 2),
            FunctionalSymbol("Mul", 2),
            FunctionalSymbol("S", 1),
            FunctionalSymbol("0", 0)
        ]
    )
    nat_interpretation = Interpretation(
        {
            "Q": lambda a, b: a == b,
            "d": lambda a, b: a % b == 0,
        },
        {
            "Plus": lambda a, b: a + b,
            "Mul": lambda a, b: a * b,
            "S": lambda a: a + 1,
            "0": 0
        }
    )
    # Больший диапазон приводит к большему времени работы алгоритма.
    domain_set_to_iterate = range(0, 100)

    # Формула для простого числа.
    # x != 0, x != 1 и для любого y верно: y|x влечёт x = y или y = 1.
    prime_number_formula = parse_formula(
        "(~Q(x, S(0)) * Ay (Ew Q(x, Mul(y, w)) > (Q(x, y) + Q(y, S(0))))))",
        signature
    )
    primes = list(filter(
        lambda x: Semantics(
            nat_interpretation,
            {"x": x},
            domain_set_to_iterate
        ).formula_to_bool(
            prime_number_formula,
            {}
        ),
        domain_set_to_iterate
    ))
    assert primes == [x for x in domain_set_to_iterate if re.match(r"^1?$|^(11+)\1+$", "1" * x) is None]
    print("primes:", primes)

    # z - НОД x и y:
    # z|x и z|y, и для любого w если w|x и w|y, то w <= z.
    gcd_formula = parse_formula(
        """((Ev Q(x, Mul(z, v)) * Ev Q(y, Mul(z, v))) * Aw ((Ev Q(x, Mul(w, v)) * Ev Q(y, Mul(w, v))) >
            Ev Q(z, Plus(v, w))))""",
        signature
    )
    x, y = 90, 45
    for i in domain_set_to_iterate:
        if Semantics(nat_interpretation, {"x": x, "y": y, "z": i}, domain_set_to_iterate) \
                .formula_to_bool(gcd_formula, {}):
            print(f"gcd({x}, {y})={i}")

    # z - остаток от деления x на y, причём y != 0:
    # z < y, y != 0 и существует такое w, что x = w * y + z;
    rem_formula = parse_formula(
        "(~Q(y, 0) * (Ew Q(y, Plus(w, S(z))) * Ew Q(x, Plus(z, Mul(w, y)))))",
        signature
    )
    x, y = 94, 19
    for i in domain_set_to_iterate:
        if Semantics(nat_interpretation, {"x": x, "y": y, "z": i}, domain_set_to_iterate) \
                .formula_to_bool(rem_formula, {}):
            assert x % y == i
            print(f"{x} % {y} = {i}")

    # x - степень простого числа (x=a^b, где a и b - натуральные числа, в том числе 0).
    # Для любых y и z верно, что если y|x и z|x, то y = 1 или z = 1 или y = z или y = x или z = x.
    prime_pow_formula = parse_formula(
        """Ay Az ((Ew Q(x, Mul(y, w)) * Ew Q(x, Mul(z, w))) > (((Q(y, S(0)) + Q(z, S(0))) + (Q(y, x) +
         Q(z, x))) + Q(y, z)))""",
        signature
    )
    prime_pows = list(filter(
        lambda x: Semantics(
            nat_interpretation,
            {"x": x},
            domain_set_to_iterate
        ).formula_to_bool(prime_pow_formula, {}),
        domain_set_to_iterate
    ))
    print("power of primes:", prime_pows)


arithmetics_semantics_example()