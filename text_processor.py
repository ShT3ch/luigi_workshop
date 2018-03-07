import logging

logger = logging.getLogger('TEXT_PROC')

from string import punctuation

from nltk import RegexpTokenizer
from pymorphy2 import MorphAnalyzer

z_ord = ord('z')

class TextProcessor():
    def __init__(self, ):
        self.morph = MorphAnalyzer()
        self.trans_table = {ord(_): u" " for _ in punctuation}
        self.tokenizer = RegexpTokenizer("\W+", gaps=True)

        self.latin_percent_treshold = 0.9

    def remove_punctuation(self, text: str):
        txt = text.replace(u"\n", u" ").replace(u"\r", u" ").lower().strip().translate(self.trans_table)
        return txt

    def tokenize_text(self, txt):
        tokens = self.tokenizer.tokenize(txt)
        return tokens

    def detect_text_lang(self, text: str):
        if sum(map(lambda x: ord(x) <= z_ord, text)) / len(text) > self.latin_percent_treshold:
            return 'en'
        else:
            return 'ru'

    def _normalize_ru_token(self, token: str):
        return self.morph.parse(token)[0].normal_form

    def normalize_ru_tokens(self, tokens):
        return [self._normalize_ru_token(token) for token in tokens]

    def _normalize_token_auto(self, token: str):
        lang = self.detect_text_lang(token)
        if lang == 'ru':
            return self._normalize_ru_token(token)
        else:
            raise ValueError('Unknown lang')

    def normalize_text_auto(self, tokens):
        return [self._normalize_token_auto(token) for token in tokens]

    def process_text(self, txt, lang='auto'):
        txt = self.remove_punctuation(txt)
        tokens = self.tokenize_text(txt)

        if lang == 'ru':
            res = self.normalize_ru_tokens(tokens)
        elif lang == 'auto':
            res = self.normalize_text_auto(tokens)
        else:
            raise ValueError('Only ["ru", "auto"] acceptable for lang parameter.')

        return res