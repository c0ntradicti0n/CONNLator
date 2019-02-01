import en_coref_sm


class GrammarParser:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = en_coref_sm.load()
        return None

    def process(self, line):
        doc = self.nlp(line)
        return doc
