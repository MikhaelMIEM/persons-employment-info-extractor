import string

import nltk
import pymorphy2
import stanza
import unicodedata

from job_titles_parser import JobTitlesParser
from persons_employment_info_extractor.domain import *

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

    def _is_token_job(token: Token, recognized_jobs: list) -> bool:
        for match in recognized_jobs:
            if match.start == token.norm_start_pos:
                return True
        return False

    def _set_job_entity(sentences: List[Sentence], norm_text: str) -> None:
        recognized_jobs = jobs_parser.findall(norm_text)
        for sentence in sentences:
            for token in sentence.tokens:
                if token.entity == EntityType.NONE and _is_token_job(token, recognized_jobs):
                    token.entity = EntityType.JOB

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
    _set_job_entity(sentences, norm_text)
    return Text(text, norm_text, sentences)


def normalize_text(text: str) -> str:
    norm_text = text
    norm_text = unicodedata.normalize("NFKD", norm_text)
    norm_text = ' '.join(
        morph.parse(word)[0].normal_form.lower()
        for word in nltk.word_tokenize(norm_text)
    )
    allowed_symbols = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя ' + string.ascii_lowercase
    for s in set(norm_text):
        if s not in allowed_symbols:
            norm_text = norm_text.replace(s, '')
    return norm_text.strip()
