"""
Генератор вывода секвенций в исчислении генценовского типа.

Поиск вывода осуществляется сверху вниз: 
+ Если секвенция - аксиома, то вывод найден.
+ Если секвенция - заключение из правила, то ищется вывод для посылок этого правила.
+ Иначе секвенция не выводима в исчислении.
"""

import itertools
from collections import namedtuple
from enum import Enum
from random import randint, choice


class Formula:
    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        return repr(self) == repr(other)


class Var(Formula):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Constant(Formula):
    def __init__(self, value: bool):
        self.value = value

    def __repr__(self) -> str:
        return 'T' if self.value else 'F'


class Negation(Formula):
    def __init__(self, a: Formula):
        self.a = a

    def __repr__(self) -> str:
        return '¬' + repr(self.a)


class BinaryOperation(Enum):
    OR = 0
    AND = 1
    IMPL = 2
    EQ = 3


class BinaryConnective(Formula):
    def __init__(self, left_op: Formula, right_op: Formula, operation: BinaryOperation):
        self.a = left_op
        self.b = right_op
        self.operation = operation

    def __repr__(self) -> str:
        op_s = {
            BinaryOperation.OR: '∨',
            BinaryOperation.AND: '^',
            BinaryOperation.IMPL: '⊃',
            BinaryOperation.EQ: '≡'
        }[self.operation]
        return f'({self.a} {op_s} {self.b})'


def calculate_bool_value(f: Formula, m: dict[str, bool]) -> bool:
    if isinstance(f, Var):
        return m[f.name]
    if isinstance(f, Negation):
        return not calculate_bool_value(f.a, m)
    if isinstance(f, Constant):
        return f.value
    if isinstance(f, BinaryConnective):
        op_f = {
            BinaryOperation.OR: lambda a, b: a or b,
            BinaryOperation.AND: lambda a, b: a and b,
            BinaryOperation.IMPL: lambda a, b: not a or b,
            BinaryOperation.EQ: lambda a, b: a == b
        }[f.operation]
        return op_f(calculate_bool_value(f.a, m), calculate_bool_value(f.b, m))


def parse_pf(s: str) -> Formula:
    class Lexeme:
        VAR = 0
        PARENTHESIS = 1
        BIN_OP = 2
        NEGATION = 3
        CONST = 4
        END = 5

    Token = namedtuple('Token', 'lexeme value')

    i = 0

    def next_token() -> Token | None:
        nonlocal i

        if i >= len(s):
            return Token(Lexeme.END, None)

        c = s[i]
        i += 1

        if c in ('T', 'F'):
            return Token(Lexeme.CONST, c == 'T')
        if c.isalpha():
            return Token(Lexeme.VAR, c)
        if c in ('(', ')'):
            return Token(Lexeme.PARENTHESIS, c)
        if c == '~':
            return Token(Lexeme.NEGATION, None)
        if c in ('+', '*', '>', '='):
            return Token(Lexeme.BIN_OP, c)
        if c in ' \t':
            return next_token()

        assert False, 'unknown symbol'

    tok = next_token()

    def parse() -> Formula:
        rules = [parse_const, parse_var, parse_neg, parse_bin]
        for rule in rules:
            if (expr := rule()) is not None:
                return expr

        assert False, 'unknown syntax rule'

    def parse_var() -> Var | None:
        if tok.lexeme == Lexeme.VAR:
            return Var(tok.value)

    def parse_const() -> Constant | None:
        if tok.lexeme == Lexeme.CONST:
            return Constant(tok.value)

    def parse_neg() -> Negation | None:
        nonlocal tok
        if tok.lexeme == Lexeme.NEGATION:
            tok = next_token()
            return Negation(parse())

    def parse_bin() -> BinaryConnective | None:
        nonlocal tok

        if tok.lexeme != Lexeme.PARENTHESIS:
            return

        tok = next_token()

        left_op = parse()

        tok = next_token()
        assert tok.lexeme == Lexeme.BIN_OP, 'expected an operator'

        operator = {
            '+': BinaryOperation.OR,
            '*': BinaryOperation.AND,
            '>': BinaryOperation.IMPL,
            '=': BinaryOperation.EQ
        }[tok.value]

        tok = next_token()
        right_op = parse()

        tok = next_token()
        assert tok.lexeme == Lexeme.PARENTHESIS, 'mismatched parenthesis'

        return BinaryConnective(left_op, right_op, operator)

    res = parse()
    assert not s[i:].strip(), 'found some symbols after the formula'
    return res


assert repr(parse_pf('(a > (b > a))')) == '(a ⊃ (b ⊃ a))'
assert repr(parse_pf('((a > (b > c)) > ((a > b) > (a > c)))')) == '((a ⊃ (b ⊃ c)) ⊃ ((a ⊃ b) ⊃ (a ⊃ c)))'
assert repr(parse_pf('(~~a > a)')) == '(¬¬a ⊃ a)'
assert repr(parse_pf('(F > a)')) == '(F ⊃ a)'


class Sequent:
    def __init__(self, antecedent: set[Formula], succedent: set[Formula]):
        self.antecedent = antecedent
        self.succedent = succedent

    def __repr__(self) -> str:
        a_s = list(sorted([repr(a) for a in self.antecedent]))
        a_s = ', '.join(a_s)
        if a_s:
            a_s += ' '

        s_s = list(sorted([repr(s) for s in self.succedent]))
        s_s = ', '.join(s_s)
        if s_s:
            s_s = ' ' + s_s
        return a_s + '⟶' + s_s


def parse_sequent(s: str) -> Sequent:
    tokens = s.split('->')
    assert len(tokens) == 2, 'sequent should have antecedent and succedent delimited by `->`'

    antecedent_s = tokens[0].strip()
    succedent_s = tokens[1].strip()

    antecedent = set()
    for antecedent_token in antecedent_s.split(','):
        antecedent_token = antecedent_token.strip()
        if not antecedent_token:
            continue

        antecedent.add(parse_pf(antecedent_token))

    succedent = set()
    for succedent_token in succedent_s.split(','):
        succedent_token = succedent_token.strip()
        if not succedent_token:
            continue

        succedent.add(parse_pf(succedent_token))

    return Sequent(antecedent, succedent)


assert repr(parse_sequent('-> (~(p * q) > (~p + ~q))')) == '⟶ (¬(p ^ q) ⊃ (¬p ∨ ¬q))'
assert repr(parse_sequent('q-> q ,~p')) == 'q ⟶ q, ¬p'


def get_axiom_type(seq: Sequent) -> int | None:
    # 1. Γ, a -> Δ, a.
    for ant in seq.antecedent:
        for suc in seq.succedent:
            if repr(ant) == repr(suc):
                return 1

    # 2. Γ, F -> Δ.
    for ant in seq.antecedent:
        if repr(ant) == 'F':
            return 2

    # 3. Γ -> Δ, T
    for suc in seq.succedent:
        if repr(suc) == 'T':
            return 3


assert get_axiom_type(parse_sequent('q -> q, ~p')) == 1
assert get_axiom_type(parse_sequent('p -> q, ~p')) is None
assert get_axiom_type(parse_sequent('a -> a')) == 1
assert get_axiom_type(parse_sequent('a, b, c -> d, e, c')) == 1
assert get_axiom_type(parse_sequent('F ->')) == 2
assert get_axiom_type(parse_sequent('-> T')) == 3
assert get_axiom_type(parse_sequent('-> a, T, b, T')) == 3
assert get_axiom_type(parse_sequent('c -> a, F, b')) is None


class RuleApplication:
    def __init__(self, rule: str, premise: list[Sequent] | Sequent, conclusion: Sequent):
        self.rule = rule
        self.premise = premise if isinstance(premise, list) else [premise]
        self.conclusion = conclusion


def get_application_rule(conclusion: Sequent) -> RuleApplication | None:
    # Γ ⟶ Δ, a
    # ---------- (¬⟶)
    # Γ, ¬a ⟶ Δ
    for ant in conclusion.antecedent:
        if isinstance(ant, Negation):
            g = conclusion.antecedent - {ant}
            d = conclusion.succedent.union({ant.a})
            return RuleApplication('¬⟶', Sequent(g, d), conclusion)

    # Γ, a ⟶ Δ
    # ---------- (⟶¬)
    # Γ ⟶ Δ, ¬a
    for suc in conclusion.succedent:
        if isinstance(suc, Negation):
            g = conclusion.antecedent.union({suc.a})
            d = conclusion.succedent - {suc}
            return RuleApplication('⟶¬', Sequent(g, d), conclusion)

    # Γ, a, b ⟶ Δ
    # ------------- (^⟶)
    # Γ, a ^ b ⟶ Δ
    for ant in conclusion.antecedent:
        if isinstance(ant, BinaryConnective) and ant.operation == BinaryOperation.AND:
            g = conclusion.antecedent.union({ant.a, ant.b}) - {ant}
            d = conclusion.succedent
            return RuleApplication('^⟶', Sequent(g, d), conclusion)

    # Γ ⟶ Δ, a; Γ ⟶ Δ, b
    # -------------------- (⟶^)
    #    Γ ⟶ Δ, a ^ b
    for suc in conclusion.succedent:
        if isinstance(suc, BinaryConnective) and suc.operation == BinaryOperation.AND:
            g = conclusion.antecedent
            d = conclusion.succedent - {suc}
            return RuleApplication(
                '⟶^',
                [
                    Sequent(g, d.union({suc.a})),
                    Sequent(g, d.union({suc.b}))
                ],
                conclusion
            )

    # Γ, a ⟶ Δ; Γ, b ⟶ Δ
    # -------------------- (∨⟶)
    #    Γ, a ∨ b ⟶ Δ
    for ant in conclusion.antecedent:
        if isinstance(ant, BinaryConnective) and ant.operation == BinaryOperation.OR:
            g = conclusion.antecedent - {ant}
            d = conclusion.succedent
            return RuleApplication(
                '∨⟶',
                [
                    Sequent(g.union({ant.a}), d),
                    Sequent(g.union({ant.b}), d)
                ],
                conclusion
            )

    # Γ ⟶ Δ, a, b
    # ------------- (⟶∨)
    # Γ ⟶ Δ, a ∨ b
    for suc in conclusion.succedent:
        if isinstance(suc, BinaryConnective) and suc.operation == BinaryOperation.OR:
            g = conclusion.antecedent
            d = conclusion.succedent.union({suc.a, suc.b}) - {suc}
            return RuleApplication('⟶∨', Sequent(g, d), conclusion)

    # Γ ⟶ Δ, a; Γ, b ⟶ Δ
    # -------------------- (⊃⟶)
    #    Γ, a ⊃ b ⟶ Δ
    for ant in conclusion.antecedent:
        if isinstance(ant, BinaryConnective) and ant.operation == BinaryOperation.IMPL:
            g = conclusion.antecedent - {ant}
            d = conclusion.succedent
            return RuleApplication(
                '⊃⟶',
                [
                    Sequent(g, d.union({ant.a})),
                    Sequent(g.union({ant.b}), d)
                ],
                conclusion
            )

    # Γ, a ⟶ Δ, b
    # ------------- (⟶⊃)
    # Γ ⟶ Δ, a ⊃ b
    for suc in conclusion.succedent:
        if isinstance(suc, BinaryConnective) and suc.operation == BinaryOperation.IMPL:
            g = conclusion.antecedent.union({suc.a})
            d = conclusion.succedent.union({suc.b}) - {suc}
            return RuleApplication('⟶⊃', Sequent(g, d), conclusion)


def expand_eq_in_sequent(seq: Sequent) -> Sequent:
    def expand_eq_in_formula(formula: Formula) -> Formula:
        if isinstance(formula, BinaryConnective) and formula.operation == BinaryOperation.EQ:
            return BinaryConnective(
                BinaryConnective(
                    formula.a,
                    formula.b,
                    BinaryOperation.IMPL
                ),
                BinaryConnective(
                    formula.b,
                    formula.a,
                    BinaryOperation.IMPL
                ),
                BinaryOperation.AND
            )
        else:
            return formula

    antecedent = set()
    for ant in seq.antecedent:
        antecedent.add(expand_eq_in_formula(ant))

    succedent = set()
    for suc in seq.succedent:
        succedent.add(expand_eq_in_formula(suc))

    return Sequent(antecedent, succedent)


def check_sequent_is_inferable(seq: Sequent, level: int = 0, verbose=True) -> bool:
    if verbose:
        s = '.   ' * level + repr(seq)
        print('{:<50}'.format(s), end='')

    seq = expand_eq_in_sequent(seq)

    if (a_type := get_axiom_type(seq)) is not None:
        if verbose:
            print(f'[axiom of type {a_type}]')
        return True

    if (ra := get_application_rule(seq)) is None:
        if verbose:
            print('[not inferable]')
        return False

    if verbose:
        print(f'[by rule {ra.rule}]')

    inferable = True
    for p in ra.premise:
        if not check_sequent_is_inferable(p, level + 1):
            inferable = False

    return inferable


laws = [
    '((p * q) = (q * p))',
    '(((p * q) * r) = (p * (q * r)))',
    '((p + q) = (q + p))',
    '(((p + q) + r) = (p + (q + r)))',
    '((p * (q + r)) = ((p * q) + (p * r)))',
    '((p + (q * r)) = ((p + q) * (p + r)))',
    '((p + (p * q)) = p)',
    '((p * (p + q)) = p)',
    '(~(p * q) = (~p + ~q))',
    '(~(p + q) = (~p * ~q))',
    '(p + ~p)',
    '(p = ~~p)',
    '((p > q) = (~q > ~p))',
    '(~p > (p > q))',
    '((p > q) = (~p + q))',

    '(a > (b > a))',
    '((a > (b > c)) > ((a > b) > (a > c)))',
    '(a > (b > (a * b)))',
    '((a * b) > a)',
    '((a * b) > b)',
    '((a > c) > ((b > c) > ((a + b) > c)))',
    '(a > (a + b))',
    '(b > (a + b))',
    '((a > b) > ((a > ~b) > ~a))',
    '(~~a > a)',
    '(F > a)',
    '(A > T)'
]

for law in laws:
    assert check_sequent_is_inferable(parse_sequent('-> ' + law))
    print("\n\n")
    # To check the inverse:
    assert not check_sequent_is_inferable(parse_sequent('-> ~' + law))
    print("\n\n")