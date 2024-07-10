from collections.abc import Iterable, Sized
from enum import Enum, auto
from typing import Protocol

__all__ = [
    "Four",
    "IterableAndSized",
    "RankedVote",
    "Ranking",
    "Three",
    "Two",
    "UtilityFunc",
    "UtilityVote",
]


class Two(Enum):
    A = auto()
    B = auto()


class Three(Enum):
    A = auto()
    B = auto()
    C = auto()


class Four(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()


type Ranking[T] = dict[T, int]

type RankedVote[T] = tuple[int, Ranking[T]]

type UtilityFunc[T] = dict[T, float]

type UtilityVote[T] = tuple[int, UtilityFunc[T]]


class IterableAndSized[T](Iterable[T], Sized, Protocol):
    pass
