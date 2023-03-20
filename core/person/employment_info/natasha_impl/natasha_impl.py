from __future__ import annotations

from typing import List, Optional

from natasha import NewsEmbedding, Segmenter, NewsNERTagger, MorphVocab, Doc
from natasha.doc import DocToken, DocSent
from natasha.norm import normalize, syntax_normalize

from person.employment_info.domain import EntitiesRecognizer, Text, EntityType, Sentence, Token, normalize_text
from person.employment_info.static import read_raw_job_titles
from person.employment_info.time_interval.time_interval_parser import parse_date_intervals


class NatashaEntitiesRecognizer(EntitiesRecognizer):
    def __init__(self):
        self.emb = NewsEmbedding()
        self.segmenter = Segmenter()
        self.ner_tagger = NewsNERTagger(self.emb)
        self.morph_vocab = MorphVocab()
        self.tokenized_norm_job_titles = self.__eval_tokenized_norm_job_titles()

    def __eval_tokenized_norm_job_titles(self) -> list[list[str]]:
        tokenized_norm_job_titles = []
        raw_titles = read_raw_job_titles()
        for raw_title in raw_titles:
            doc = Doc(raw_title)
            doc.segment(self.segmenter)
            tokenized_norm_job_title = [normalize_text(token.text) for token in doc.tokens]
            tokenized_norm_job_titles.append(tokenized_norm_job_title)
        return tokenized_norm_job_titles

    def recognize_entities(self, text: str) -> Text:
        sentences, norm_text = self.__recognize_named_entities_natasha(text)
        self.__set_job_entities(sentences)
        self._set_entity(sentences, norm_text, parse_date_intervals, EntityType.TIME)
        return Text(text, norm_text, sentences)

    def __eval_doc(self, text: str) -> Doc:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)
        for span in doc.spans:
            span.normalize(self.morph_vocab)
        return doc

    def __recognize_named_entities_natasha(self, text: str) -> (List[Sentence], str):
        doc = self.__eval_doc(text)
        norm_text = ''
        sentences = []
        for sent in doc.sents:
            tokens = []
            for token in sent.tokens:
                token_entity_str = self.__eval_entity(token, sent)
                named_entity = self.__map_entity(token_entity_str)
                token_norm_text = self.__normalize(token, token_entity_str) if named_entity != EntityType.NONE \
                    else normalize_text(token.text)
                domain_token = Token(
                    token.text, token.start, token.stop,
                    token_norm_text, len(norm_text), len(norm_text) + len(token_norm_text),
                    named_entity
                )
                tokens.append(domain_token)
                norm_text += token_norm_text.strip() + ' '
            sentences.append(Sentence(tokens))
            norm_text = norm_text.strip() + '.'
        return sentences, norm_text

    @staticmethod
    def __eval_entity(token: DocToken, sent: DocSent) -> str:
        for span in sent.spans:
            for span_token in span.tokens:
                if span_token is token:
                    return span.type

    @staticmethod
    def __map_entity(type: str) -> EntityType:
        if type == 'ORG':
            return EntityType.ORG
        elif type == 'PER':
            return EntityType.PER
        return EntityType.NONE

    def __normalize(self, token: DocToken, named_entity_type: Optional[str] = None) -> str:
        normalizer = syntax_normalize if named_entity_type == 'ORG' else normalize
        return normalizer(self.morph_vocab, [token])

    def __set_job_entities(self, sentences: List[Sentence]) -> None:
        for sentence in sentences:
            sentence_norm_tokens = [token.norm_text for token in sentence.tokens]
            for job_title in self.tokenized_norm_job_titles:
                for start in range(len(sentence_norm_tokens) - len(job_title) + 1):
                    end = start + len(job_title)
                    if job_title == sentence_norm_tokens[start:end]:
                        for token in sentence.tokens[start:end]:
                            token.entity = EntityType.JOB
