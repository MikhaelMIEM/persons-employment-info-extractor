from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product, chain
from typing import List, Tuple, Optional


class EntityType(Enum):
    NONE = auto()
    PER = auto()
    ORG = auto()
    JOB = auto()
    TIME = auto


@dataclass
class Token:
    text: str
    start_pos: int
    end_pos: int
    norm_text: str
    norm_start_pos: int
    norm_end_pos: int
    entity: EntityType

    def concat(self, token: Token) -> None:
        self.text = (self.text + ' ' + token.text).strip()
        self.end_pos = token.end_pos
        self.norm_text = (self.norm_text + ' ' + token.norm_text).strip()
        self.norm_end_pos = token.norm_end_pos

    @property
    def coordinates(self) -> Tuple[int, int]:
        return self.start_pos, self.end_pos

    @property
    def norm_coordinates(self) -> Tuple[int, int]:
        return self.norm_start_pos, self.norm_end_pos

    def eval_distance(self, token: Token) -> int:
        return min(abs(p[0] - p[1]) for p in product(self.coordinates, token.coordinates))

    def eval_norm_distance(self, token: Token) -> int:
        return min(abs(p[0] - p[1]) for p in product(self.norm_coordinates, token.norm_coordinates))

    def eval_closest_token(self, tokens: List[Token]) -> Optional[Token]:
        closest = closest_distance = None
        for t in tokens:
            distance = self.eval_distance(t)
            if closest_distance is None or distance < closest_distance:
                closest = t
                closest_distance = distance
        return closest


@dataclass
class Sentence:
    tokens: List[Token]

    @property
    def entities(self) -> List[Token]:
        concat_tokens = []
        for token in self.tokens:
            if not concat_tokens:
                concat_tokens.append(copy(token))
                continue
            lst = concat_tokens[-1]
            if token.entity != EntityType.NONE and lst.entity == token.entity:
                lst.concat(token)
            else:
                concat_tokens.append(copy(token))
        return [token for token in concat_tokens if token.entity != EntityType.NONE]

    def calc_entities_by_type(self, entity_type: EntityType) -> List[Token]:
        return list(e for e in self.entities if e.entity == entity_type)


@dataclass
class Text:
    text: str
    norm_text: str
    sentences: List[Sentence]


@dataclass
class Work:

    person: Token
    companies: List[Token]
    jobs: List[Token]
    start_time: Optional[TimeStamp] = None
    end_time: Optional[TimeStamp] = None

    @property
    def jobs_norm_names(self) -> List[str]:
        return [job.norm_text for job in self.jobs]

    @property
    def companies_norm_names(self) -> List[str]:
        return [company.norm_text for company in self.companies]


@dataclass
class TimeStamp:
    year: int
    month: int
    day: int


@dataclass
class TextPersonInfo:
    name_tokens: set[str]
    work: List[Work]

    @property
    def norm_name(self) -> Optional[str]:
        return max((i.person.norm_text for i in self.work), key=len)

    @property
    def name(self) -> Optional[str]:
        return max((i.person.text for i in self.work), key=len)

    @property
    def jobs_norm_names(self) -> List[str]:
        return list(chain.from_iterable(work.jobs_norm_names for work in self.work))

    @property
    def companies_norm_names(self) -> List[str]:
        return list(chain.from_iterable(work.companies_norm_names for work in self.work))


class TextMatch(ABC):
    start: int
    end: int
