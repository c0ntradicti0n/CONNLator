import re
import nltk
from nested_list_tools import stack_matryoshka

class Preprocessor:
    def __init__(self):
        self.quotes = {}


    class Replacement(object):
        def __init__(self, dict, fun, replacement):
            self.replacement = replacement
            self.occurrences = []
            self.dict = dict
            self.fun = fun

        def __call__(self, match):
            matched = match.group(0)
            replaced = self.fun(match.expand(self.replacement))
            self.dict.update({matched: replaced})

            self.occurrences.append((matched, replaced))
            return replaced

    def tokenize_text_to_sentences (self,text):
        try:
            all_sentences = nltk.tokenize.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            all_sentences = nltk.tokenize.sent_tokenize(text)
        return all_sentences

    def tokenize_text_to_words (self,sentence):
        return nltk.word_tokenize(sentence)

    def clean_string (self, string):
        if isinstance(string, list):
            strings = []
            for s in string:
                strings.append(self.process(s))
            return strings

        # string = string.lower()                                          # big letters
        # string = re.sub(ur"([‚‘])",                              "'", string)
        string = re.sub(r"""e\.g\.""", " exempli gratia ", string)
        string = re.sub(r"""e\.g""", " exempli gratia ", string)

        string = re.sub(r"""([\d]+[\w]+.)""", "", string)

        string = re.sub(r"""(\{[\(\)A-Za-z,.;:\-\s"']*\})""", "", string)  # original text annotations
        string = re.sub(r"""(\([\w‚‘\s,.;:"']+\))""", "", string)          # text in parentheses
        string = re.sub(r"""\[([A-Za-z,.;:\-\s"']*)\]""", "\\1", string)   # own manipulation

        def modifier(s):
            return " " + s.replace(" ", "").replace("-", "").title().strip()

        replace_the_cites = self.Replacement(self.quotes, modifier, " \\2") # other language layer by quoting
        string = re.sub(r"""(?:^|\s)(["'])((?:\\.|[^\\])*?)(\1)""",
                        replace_the_cites, string)

        # for matched, replaced in replace_the_cites.occurrences:
        #    print (matched, '=>', replaced)

        string = re.sub("[:;]", ",", string)                               # punctuation of sentences
        string = string.replace("-", "")                                   # dashs
        string = string.replace("—", ", ")

        string = " ".join(string.split())                                  # all kinds of whitespaces, tabs --> one space
        return string

    def extract_subdivision_structure(self, division_markers, all_sentences):

        def regex_from_marker(marker_s):
            if isinstance(marker_s, list):
                r = "(" + "|".join(marker_s) + ")" + "\\s+\\d+\\s"
            else:
                r = marker_s + "\\s+\\d+"+"\\s"

            return r

        def recognize_heading (marker):
            def condition_fun (s):
                r = regex_from_marker(marker)
                m = re.search(r, s)
                return m
            return condition_fun

        def splitlist(L, cond):
            if not L: return []

            level = [[]]
            headings = []
            l = 0
            for i, s in enumerate (L):
                m = cond(s)
                if m:
                    l += 1
                    level.append([])
                    headings.append(m.group())
                level[l].append(i)
            if not level[0]:
                level = level[1:]
            return level


        def delete_markers_in_sentences(all_sentences, marker_s):
            r = regex_from_marker(marker_s)
            for i, s in enumerate(all_sentences):
                s = re.sub(r, "", s)
                all_sentences[i] = s
            return all_sentences

        levels = []
        for marker in division_markers:
            level = splitlist(all_sentences, recognize_heading(marker))
            levels.append(level)
            print (level)

        subdivision_structure = stack_matryoshka(levels)
        all_sentences = delete_markers_in_sentences(all_sentences, division_markers)

        return all_sentences, subdivision_structure



    def process(self, text):
        cleaned_text = self.clean_string(text)
        tokenized_text = self.tokenize_text_to_sentences(cleaned_text)
        return tokenized_text