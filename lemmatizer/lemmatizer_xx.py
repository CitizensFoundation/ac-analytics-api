"""
python3 -m spacy download en_core_web_sm
python3 -m spacy download zh_core_web_sm
python3 -m spacy download de_core_news_sm
python3 -m spacy download da_core_news_sm
python3 -m spacy download ja_core_news_sm
python3 -m spacy download lt_core_news_sm
python3 -m spacy download nb_core_news_sm
python3 -m spacy download pl_core_news_sm
python3 -m spacy download es_core_news_sm
python3 -m spacy download es_core_news_sm
python3 -m spacy download pt_core_news_sm
python3 -m spacy download it_core_news_sm
python3 -m spacy download nl_core_news_sm
python3 -m spacy download el_core_news_sm
python3 -m spacy download ro_core_news_sm
python3 -m spacy download xx_ent_wiki_sm
"""

from spacy import load
from argparse import ArgumentParser

import en_core_web_sm
import zh_core_web_sm
import da_core_news_sm
import de_core_news_sm
import fr_core_news_sm
import ja_core_news_sm
import lt_core_news_sm
import nb_core_news_sm
import pl_core_news_sm
import es_core_news_sm
import pt_core_news_sm
import it_core_news_sm
import nl_core_news_sm
import el_core_news_sm
import ro_core_news_sm
import xx_ent_wiki_sm

class LemmatizerMultilanguageClass:
    english = en_core_web_sm.load()
    chinese = None
    danish = None
    german = None
    french = None
    japanese = None
    lithuanian = None
    norwegian = None
    polish = None
    spanish = None
    portuguese = None
    italian = None
    dutch = None
    greek = None
    romanian = None
    multilanguage = None


    def getLemmatizedText(self, language, text):
        out = [str(l) for l in self.lemmatize(language, text)]
        return ' '.join(out)

    def get_words(self, language, text):
        words = None
        if language=='en':
            words = self.english(text)
        elif language=='zh':
            if  self.chinese==None:
                self.chinese = zh_core_web_sm.load()
            words =  self.chinese(text)
        elif language=='da':
            if self.danish==None:
                self.danish = da_core_news_sm.load()
            words =  self.danish(text)
        elif language=='de':
            if self.german==None:
                self.german = de_core_news_sm.load()
            words =  self.german(text)
        elif language=='fr':
            if self.french==None:
                self.french = fr_core_news_sm.load()
            words =  self.french(text)
        elif language=='ja':
            if self.japanese==None:
                self.japanese = ja_core_news_sm.load()
            words =  self.japanese(text)
        elif language=='lt':
            if self.lithuanian==None:
                self.lithuanian = lt_core_news_sm.load()
            words =  self.lithuanian(text)
        elif language=='nb':
            if self.norwegian==None:
                self.norwegian = nb_core_news_sm.load()
            words =  self.norwegian(text)
        elif language=='pl':
            if self.polish==None:
                self.polish = pl_core_news_sm.load()
            words = self.polish(text)
        elif language=='es':
            if self.spanish==None:
                self.spanish = es_core_news_sm.load()
            words = self.spanish(text)
        elif language=='pt':
            if self.portuguese==None:
                self.portuguese = pt_core_news_sm.load()
            words = self.portuguese(text)
        elif language=='it':
            if self.italian==None:
                self.italian = it_core_news_sm.load()
            words = self.italian(text)
        elif language=='nl':
            if self.dutch==None:
                self.dutch = nl_core_news_sm.load()
            words = self.dutch(text)
        elif language=='el':
            if self.greek==None:
                self.greek = el_core_news_sm.load()
            words = self.greek(text)
        elif language=='ro':
            if self.romanian==None:
                self.romanian = ro_core_news_sm.load()
            words = self.romanian(text)
        else:
            if self.multilanguage==None:
                self.multilanguage = xx_ent_wiki_sm.load()
            words = self.multilanguage(text)
        return words

    def lemmatize(self, language, text):
        words = self.get_words(language, text)
        for token in words:
            if token.lemma_ == '-PRON-':
                yield token
            else:
                yield token.lemma_
