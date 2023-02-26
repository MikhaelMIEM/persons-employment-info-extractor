from dataclasses import dataclass
from importlib import resources
from typing import Iterator, Optional

from ahocorasick import Automaton

from person.employment_info.domain import TextMatch


@dataclass
class JobMatch(TextMatch):
    match: str
    start: int
    end: int


class JobTitlesParser:

    def __init__(self, titles: Optional[Iterator] = None):
        def _load_titles():
            with resources.open_text('core.person.employment_info.job_titles', 'normalized_dict.txt') as dict_file:
                for line in dict_file:
                    if line:
                        yield line.strip()
        titles = titles if titles is not None else _load_titles()
        self.ahocorasick = Automaton()
        for title in titles:
            self.ahocorasick.add_word(title, title)
            self.ahocorasick.add_word(title.lower(), title.lower())
        self.ahocorasick.make_automaton()

    def findall(self, text: str) -> list[JobMatch]:
        return [JobMatch(match, end - len(match) + 1, end) for end, match in self.ahocorasick.iter(text)]
