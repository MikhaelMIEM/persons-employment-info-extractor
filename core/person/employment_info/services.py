from __future__ import annotations

from dataclasses import dataclass
from typing import List

from person.employment_info.domain import TextPersonInfo, Work, EntityType, Text, EntitiesRecognizer
from person.employment_info.natasha_impl.natasha_impl import NatashaEntitiesRecognizer
from person.employment_info.stanza_impl.stanza_impl import StanzaEntitiesRecognizer
from person.employment_info.time_interval.time_interval_parser import parse_date_interval


@dataclass
class PersonInfoExtractor:

    nlp: EntitiesRecognizer = NatashaEntitiesRecognizer()

    def extract(self, text: str) -> List[TextPersonInfo]:
        entities_text = self.nlp.recognize_entities(text)
        works = self.__group_entities_by_person(entities_text)
        return self.__group_persons_by_normalized_name(works)

    @staticmethod
    def __group_persons_by_normalized_name(persons_info: List[Work]) -> List[TextPersonInfo]:
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

    @staticmethod
    def __group_entities_by_person(recognized_text: Text) -> List[Work]:
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
                work.start_time = time_match.start_time if time_match else None
                work.end_time = time_match.end_time if time_match else None
            for company in sentence.calc_entities_by_type(EntityType.ORG):
                closest_person = company.eval_closest_token(names)
                sentence_persons_info[closest_person.norm_text].companies.append(company)
            for job in sentence.calc_entities_by_type(EntityType.JOB):
                closest_person = job.eval_closest_token(names)
                sentence_persons_info[closest_person.norm_text].jobs.append(job)
            persons_info.extend(work for work in sentence_persons_info.values() if work.companies or work.jobs)
        return persons_info
