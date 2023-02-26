# -*- coding: utf-8 -*-

from yargy import rule, and_, or_
from yargy.interpretation import fact
from yargy.predicates import (
    lte,
    gte,
    dictionary,
    in_
)

from time_interval_parser import parse_date_intervals

Date = fact(
    'Date',
    ['year', 'month', 'day']
)

Interval = fact(
    'Interval',
    ['start_year', 'start_month', 'start_day', 'end_year', 'end_month', 'end_day']
)

MONTHS = {
    'январь': 1,
    'февраль': 2,
    'март': 3,
    'апрель': 4,
    'мая': 5,
    'июнь': 6,
    'июль': 7,
    'август': 8,
    'сентябрь': 9,
    'октябрь': 10,
    'ноябрь': 11,
    'декабрь': 12
}
MONTH_NAME = dictionary(MONTHS.keys())
MONTH_NUM = and_(
    gte(1),
    lte(12)
)
DAY = and_(
    gte(1),
    lte(31)
)
YEAR = and_(
    gte(1900),
    lte(2100)
)
DELIMITER = in_({'-', '.', '_', '/', ','})

date_rule = or_(
    rule(
        DAY.optional().interpretation(Date.day.custom(int)),
        DELIMITER.optional(),
        or_(
            MONTH_NAME.interpretation(Date.month.normalized().custom(MONTHS.get)),
            MONTH_NUM.interpretation(Date.month.custom(int)),
        ),
        DELIMITER.optional(),
        YEAR.interpretation(Date.year.custom(int))
    ),
).interpretation(Date)

# parser = Parser(date_rule)

if __name__ == '__main__':
    text = '''
18 июня 2016
18 06  2016
06  2016
июне 2016
06 2016
18-июня-2016

c июня 2017 по июль 2018 
c июня 2017 до июля 2018 
    '''

    text = """
         в 1993 году Василий Абобович работал в компании NetCracker.
    с 1992 по 2003 год Игорь Абраменко работал в компании LinkTech.
    """

    inters = parse_date_intervals(text)
    print([(i.start_time, i.end_time) for i in inters])
