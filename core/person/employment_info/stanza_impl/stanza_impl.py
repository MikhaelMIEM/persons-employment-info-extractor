from __future__ import annotations

from typing import List

import nltk
import stanza
from ahocorasick import Automaton

from person.employment_info.domain import EntitiesRecognizer, Text, EntityType, Sentence, Token, normalize_text
from person.employment_info.domain import TextMatch
from person.employment_info.static import read_raw_job_titles
from person.employment_info.time_interval.time_interval_parser import parse_date_intervals


class JobTitlesParserAhocorasick:
    def __init__(self):
        titles = read_raw_job_titles()
        self.ahocorasick = Automaton()
        for title in titles:
            title = normalize_text(title)
            self.ahocorasick.add_word(title, title)
            self.ahocorasick.add_word(title.lower(), title.lower())
        self.ahocorasick.make_automaton()

    def findall(self, norm_text: str) -> list[TextMatch]:
        return [TextMatch(match, end - len(match) + 1, end) for end, match in self.ahocorasick.iter(norm_text)]


class StanzaEntitiesRecognizer(EntitiesRecognizer):

    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        stanza.download('ru')
        self.nlp = stanza.Pipeline(lang='ru', processors='tokenize,ner')
        self.jobs_parser = JobTitlesParserAhocorasick()

    def recognize_entities(self, text: str) -> Text:
        norm_text, sentences = self.__recognize_named_entities_stanza(text)
        self._set_entity(sentences, norm_text, self.jobs_parser.findall, EntityType.JOB)
        self._set_entity(sentences, norm_text, parse_date_intervals, EntityType.TIME)
        return Text(text, norm_text, sentences)

    def __recognize_named_entities_stanza(self, text: str) -> (str, List[Sentence]):
        def _eval_entity(token: stanza.models.common.doc.Token) -> EntityType:
            if 'ORG' in token.ner:
                return EntityType.ORG
            elif 'PER' in token.ner:
                return EntityType.PER
            return EntityType.NONE

        norm_text = ''
        sentences = []
        for sentence in self.nlp(text).sentences:
            tokens = []
            for token in sentence.tokens:
                norm_token = normalize_text(token.text)
                tokens.append(
                    Token(
                        token.text, token.start_char, token.end_char,
                        norm_token, len(norm_text), len(norm_text) + len(norm_token),
                        _eval_entity(token)
                    )
                )
                norm_text += norm_token + ' '
            sentences.append(Sentence(tokens))
        return norm_text, sentences
