from dataclasses import dataclass
from importlib import resources
from typing import Iterator, Optional

from ahocorasick import Automaton

from persons_employment_info.domain import TextMatch


@dataclass
class JobMatch(TextMatch):
    start: int
    end: int
    match: str


class JobTitlesParser:

    def __init__(self, titles: Optional[Iterator] = None):
        def _load_titles():
            with resources.open_text('job_titles', 'normalized_dict.txt') as gzf:
                for line in gzf:
                    yield line.strip()
        titles = titles if titles is None else _load_titles()
        self.ahocorasick = Automaton()
        for title in titles:
            self.ahocorasick.add_word(title, title)
            self.ahocorasick.add_word(title.lower(), title.lower())
        self.ahocorasick.make_automaton()

    def findall(self, text: str) -> list[JobMatch]:
        return [JobMatch(end - len(match) + 1, end, match) for end, match in self.ahocorasick.iter(text)]
