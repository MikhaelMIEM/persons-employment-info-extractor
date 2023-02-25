import string
from typing import Callable

import nltk
import pymorphy2
import stanza
import unicodedata

from core.persons_employment_info.domain import *
from core.persons_employment_info.time_interval.time_interval_parser import parse_date_intervals, parse_date_interval
from persons_employment_info.job_titles import JobTitlesParser

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# stanza.download('ru')


jobs_parser = JobTitlesParser()
morph = pymorphy2.MorphAnalyzer()
nlp = stanza.Pipeline(lang='ru', processors='tokenize,ner')


# API

def extract_persons_info(text: str) -> List[TextPersonInfo]:
    return group_persons_by_normalized_name(group_entities_by_person(recognize_entities(text)))


# Implementation

def group_persons_by_normalized_name(persons_info: List[Work]) -> List[TextPersonInfo]:
    # O(n^2) could be optimized?
    text_persons = []
    for person_info in persons_info:
        norm_tokens = set(person_info.person.norm_text.strip().split())
        for text_person in text_persons:
            larger, smaller = (norm_tokens, text_person.name_tokens) \
                if len(norm_tokens) > len(text_person.name_tokens) else (text_person.name_tokens, norm_tokens)
            if smaller.issubset(larger):
                text_person.name_tokens = larger
                text_person.work.append(person_info)
                break
        else:
            text_persons.append(TextPersonInfo(norm_tokens, [person_info]))
    return text_persons


def group_entities_by_person(recognized_text: Text) -> List[Work]:
    persons_info = []
    for sentence in recognized_text.sentences:
        names = sentence.calc_entities_by_type(EntityType.PER)
        if not names:
            continue
        sentence_persons_info = {name.norm_text: Work(name, [], []) for name in names}
        for time_ in sentence.calc_entities_by_type(EntityType.TIME):
            time_match = parse_date_interval(time_.norm_text)
            closest_person = time_.eval_closest_token(names)
            work = sentence_persons_info[closest_person.norm_text]
            work.start_time = time_match.start_time
            work.end_time = time_match.end_time
        for company in sentence.calc_entities_by_type(EntityType.ORG):
            closest_person = company.eval_closest_token(names)
            sentence_persons_info[closest_person.norm_text].companies.append(company)
        for job in sentence.calc_entities_by_type(EntityType.JOB):
            closest_person = job.eval_closest_token(names)
            sentence_persons_info[closest_person.norm_text].jobs.append(job)
        persons_info.extend(sentence_persons_info.values())
    return persons_info


def recognize_entities(text: str) -> Text:

    def _eval_entity(token: stanza.models.common.doc.Token) -> EntityType:
        if 'ORG' in token.ner:
            return EntityType.ORG
        elif 'PER' in token.ner:
            return EntityType.PER
        return EntityType.NONE

    def _set_entity(sentences: List[Sentence], norm_text: str, tokens_interval_func: Callable[[str], list[TextMatch]],
                    entity_type: EntityType) -> None:
        def _is_token_in_intervals(token: Token, token_intervals: list[TextMatch]) -> bool:
            for interval in token_intervals:
                if interval.start <= token.norm_start_pos < interval.end:
                    return True
            return False
        tokens_intervals = tokens_interval_func(norm_text)
        for sentence in sentences:
            for token in sentence.tokens:
                if token.entity == EntityType.NONE and _is_token_in_intervals(token, tokens_intervals):
                    token.entity = entity_type

    norm_text = ''
    sentences = []
    for sentence in nlp(text).sentences:
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
    _set_entity(sentences, norm_text, jobs_parser.findall, EntityType.JOB)
    _set_entity(sentences, norm_text, parse_date_intervals, EntityType.TIME)
    return Text(text, norm_text, sentences)


def normalize_text(text: str) -> str:
    norm_text = text
    norm_text = unicodedata.normalize("NFKD", norm_text)
    norm_text = ' '.join(
        morph.parse(word)[0].normal_form.lower()
        for word in nltk.word_tokenize(norm_text)
    )
    allowed_symbols = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя 1234567890' + string.ascii_lowercase
    for s in set(norm_text):
        if s not in allowed_symbols:
            norm_text = norm_text.replace(s, '')
    return norm_text.strip()
