from dataclasses import dataclass
from typing import Optional

from yargy import Parser, rule, and_, or_
from yargy.interpretation import fact
from yargy.predicates import (
    lte,
    gte,
    dictionary,
    in_
)

from person.employment_info.domain import TextMatch, TimeStamp

Interval = fact(
    'Interval',
    ['start_year', 'start_month', 'start_day', 'end_year', 'end_month', 'end_day']
)

MONTHS = {
    'январь': 1,
    'зима': 1,
    'февраль': 2,
    'март': 3,
    'апрель': 4,
    'весна': 4,
    'мая': 5,
    'июнь': 6,
    'июль': 7,
    'лето':  7,
    'август': 8,
    'сентябрь': 9,
    'октябрь': 10,
    'осень': 10,
    'ноябрь': 11,
    'декабрь': 12
}
MONTH_NAME = dictionary(MONTHS.keys())
MONTH_NUM = and_(gte(1), lte(12))
DAY = and_(gte(1), lte(31))
YEAR = and_(gte(1900), lte(2100))
DELIMITER = in_({'-', '.', '_', '/', ','})

interval_rule = rule(
    rule(
        dictionary({'с'}).optional(),
        DAY.optional().interpretation(Interval.start_day.custom(int)),
        DELIMITER.optional(),
        or_(
            MONTH_NAME.interpretation(Interval.start_month.normalized().custom(MONTHS.get)),
            MONTH_NUM.interpretation(Interval.start_month.custom(int)),
        ).optional(),
        DELIMITER.optional(),
        YEAR.interpretation(Interval.start_year.custom(int)),
        dictionary({'г', 'г.', 'год'}).optional(),
    ).optional(),
    dictionary({'до', 'по'}).optional(),
    rule(
        DAY.optional().interpretation(Interval.end_day.custom(int)),
        DELIMITER.optional(),
        or_(
            MONTH_NAME.interpretation(Interval.end_month.normalized().custom(MONTHS.get)),
            MONTH_NUM.interpretation(Interval.end_month.custom(int)),
        ).optional(),
        DELIMITER.optional(),
        YEAR.interpretation(Interval.end_year.custom(int)),
        dictionary({'г', 'г.', 'год'}).optional()
    ),
).interpretation(Interval)

parser = Parser(interval_rule)


@dataclass
class TimeIntervalMatch(TextMatch):
    interval: Interval

    @property
    def start_time(self) -> Optional[TimeStamp]:
        if self.interval.start_year:
            return TimeStamp(self.interval.start_year, self.interval.start_month, self.interval.start_day)

    @property
    def end_time(self) -> Optional[TimeStamp]:
        if self.interval.end_year:
            return TimeStamp(self.interval.end_year, self.interval.end_month, self.interval.end_day)


def parse_date_intervals(text: str) -> list[TimeIntervalMatch]:
    intervals = []
    for match in parser.findall(text):
        start, end = match.tokens[0].span.start, match.tokens[-1].span.stop
        intervals.append(TimeIntervalMatch(text[start:end], start, end, match.fact))
    return intervals


def parse_date_interval(text: str) -> Optional[TimeIntervalMatch]:
    intervals = parse_date_intervals(text)
    if intervals:
        return intervals[0]
