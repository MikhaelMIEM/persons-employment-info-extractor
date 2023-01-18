import logging
from collections import namedtuple

import ahocorasick

Match = namedtuple('Match', ['start', 'end', 'match'])


def load_titles():
    with open('normalized_dict.txt') as gzf:
        for line in gzf:
            yield line.strip()


def longest_match(matches):
    try:
        longest = next(matches)
        if longest is None:
            return
    except StopIteration:
        return
    for elt in matches:
        if (elt.start >= longest.start and elt.end <= longest.end) or \
           (longest.start >= elt.start and longest.end <= elt.end):
            longest = max(longest, elt, key=lambda x: x.end - x.start)
        else:
            yield longest
            longest = elt
    yield longest


class JobTitlesParser:

    def __init__(self, ignore_case=True, titles=None, extra_titles=None):
        titles = titles if titles else load_titles()
        logging.info('building job title searcher')
        autom = ahocorasick.Automaton()
        for title in titles:
            autom.add_word(title, title)
            if ignore_case:
                autom.add_word(title.lower(), title.lower())
        if extra_titles:
            for title in extra_titles:
                autom.add_word(title, title)
                if ignore_case:
                    autom.add_word(title.lower(), title.lower())
        autom.make_automaton()
        self.autom = autom
        logging.info('building done')

    def findall(self, string, use_longest=True):
        return list(self.finditer(string, use_longest=use_longest))

    def finditer(self, string, use_longest=True):
        if use_longest:
            return longest_match(self.find_raw(string))
        else:
            return self.find_raw(string)

    def find_raw(self, string):
        for end, match in self.autom.iter(string):
            start = end - len(match) + 1
            yield Match(start=start, end=end, match=match)
