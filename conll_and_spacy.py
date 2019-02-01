import re
import os
from itertools import count

import numpy as np
import pandas as pd
import copy

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


def find_position_in_doc_by_approx(doc, text_token, pos, deviation=10):
    deviator = iterate_away(pos, deviation)
    for p in deviator:
        if p < 0:
            continue
        if p >= len(doc):
            continue
        if text_token == doc[p].text or ((text_token in ['’'] or len(text_token)<2) and text_token in doc[p].text):
            return p
    else:
        logging.error("Token '%s' not seen in spacy doc (search tokens: '%s')! returning starting position, '%s" %
                      (text_token,
                       str([w.text for w in doc[pos - deviation:pos + deviation]]),
                       str(doc[pos])))
        return pos

def iterate_away(pos, deviation):
    yield pos
    for d in range(1, deviation):
        yield pos + d
        yield pos - d



class ConllSpacyUpdater:
    def __init__(self, import_dir = None, export_dir =  None):
        if not export_dir:
            raise AttributeError("Export dir must be given!")
        self.import_dir =  import_dir
        self.export_dir =  export_dir
        return None

    def read_one_conll (fname):
        sentence = []
        conll_lines = []

        with open(fname, 'r') as fh:
            for i, line in enumerate (fh):
                try:
                    sentence.append(re.search(r'(?:^\d+\t)([^\t]+)', line).group(1))
                    conll_lines.append(ConllSpacyUpdater.conll_line2match(line))
                except AttributeError:
                    logging.error ("in file %s, line %d with line:'%s'" % (fname, i, line))
                    raise
                if not line.strip():
                    line = last
                    break
                last = line

                pass
        return conll_lines, " ".join(sentence)

    def load_all_conlls (export_path):
        all_sentences = []
        all_conlls = []
        import fnmatch
        for filename in sorted(os.listdir(export_path), key =lambda x: int(''.join(filter(str.isdigit, x)) )):
            if fnmatch.fnmatch(filename, '*.conll'):
                filename = os.path.join(export_path, filename)
                conll_lines, sentence = ConllSpacyUpdater.read_one_conll(filename)
                all_sentences.append (sentence)
                all_conlls.append (conll_lines)
        return all_sentences, all_conlls

    def load_conll (self, i, corpus_path):
        if isinstance(i, list):
            docs = []
            for j in i:
                print (j)
                docs.append(self.load_conll(j, corpus_path))
            return docs

        fname = corpus_path + "/" + str (i) + '.conll'
        sentence = []
        last = ''
        with open(fname, 'r') as fh:
            for line in fh:
                try:
                    sentence.append(re.search(r'(?:^\d+\t)([^\t]+)', line).group(1))
                except AttributeError:
                    print (i, "'"+line+"'")
                    raise
                if not line.strip():
                    line = last
                    break
                last = line

                pass
        doc = self.nlp(" ".join(sentence))
        new_doc = self.conll_over_spacy(doc, fname)
        return new_doc

    pattern = re.compile(   r"""(?P<id>.*?)        # quoted name
                                 \t(?P<text>.*?)    # whitespace, next bar, n1
                                 \t(?P<nothing1>.*?)# whitespace, next bar, n1
                                 \t(?P<pos_>.*?)    # whitespace, next bar, n2
                                 \t(?P<tag_>.*?)    # whitespace, next bar, n1
                                 \t(?P<nothing2>.*?)# whitespace, next bar, n1
                                 \t(?P<head_id>.*?) # whitespace, next bar, n2
                                 \t(?P<dep_>.*?)    # whitespace, next bar, n2
                                 \t(?P<spacy_i>.*?)# whitespace, next bar, n1
                                 \t(?P<coreference>.*?)# whitespace, next bar, n1
                                 """, re.VERBOSE)
    col_set = ['i','text', 'lemma','pos','tag','nothing','head','dep','spacy_i','coreference']

    def conll_line2match(line):
        match = ConllSpacyUpdater.pattern.match(line)
        return match


    def conll_over_spacy(self, doc, dir, i, no_cols={}):
        to_change = set(self.col_set) - set(no_cols)
        fname = str (i) + '.conll'
        path  = dir + "/" + fname

        # read conll_files, may manipulated over spacy
        with open(path) as f:
            for line in f:
                match = ConllSpacyUpdater.conll_line2match(line)
                i = int(match.group("id")) - 1
                head_i = int(match.group("head_id")) - 1
                doc[i].set_extension('coref', default = list(), force=True)
                try:
                    if 'head' in to_change:
                        doc[i].head = doc[head_i]
                    if 'lemma' in to_change:
                        doc[i].lemma_ = match.group("pos_")
                    if 'pos' in to_change:
                        doc[i].pos_ = match.group("pos_")
                    if 'tag' in to_change:
                        doc[i].tag_ = match.group("tag_")
                    if 'dep' in to_change:
                        doc[i].dep_ = match.group("dep_")
                    #if 'spacy_i' in to_change:
                    #    doc[i].i      = match.group("spacy_i")
                    if 'coreference' in to_change:
                        doc[i]._.coref= match.group("coreference")

                except IndexError:
                    raise ValueError("Shape of the spacy doc and conll file incongruent, look for the number of tokens! '%s'" % (str(doc)))
        return doc

    conll_format = "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s"
    def export_dict (self, doc, index=None):
        res = []
        w_counter = count(0)
        start_i = doc[0].i

        for word in doc:
            i = next(w_counter)
            if word.head is word:
                head_idx = 0
            else:
                head_idx = doc[i].head.i

            # indices must be +1 because of the conll format
            res.append(
                          {  's_id'   : index,
                             'i'      : i+1,
                             'text'   : word.text,
                             'lemma'  : word.lemma_,
                             'pos'    : word.pos_,
                             'tag'    : word.tag_,
                             'unknown': '_',
                             'head'   : head_idx  - start_i  + 1,
                             'dep'    : word.dep_,
                             'corp_id': str(index)+'-'+str(word.i  + 1),
                             'doc_i'  : word.i,
                             #'coref'  : coref
                          }
                      )
        return res

    def commonize_values (df, col_with_lists, col_to_index):
        """Select rows with overlapping values
        """
        v = df.merge(df, on=col_with_lists)
        common_cols = set(
            np.sort(v.iloc[:, [0, -1]].query(str('%s_x != %s_y' % (col_to_index, col_to_index)) ), axis=1).ravel()
        )
        return df[df[col_to_index].isin(common_cols)].groupby(col_to_index)[col_with_lists].apply(list)

    def explode(df, column_to_explode):
        """
        Similar to Hive's EXPLODE function, take a column with iterable elements, and flatten the iterable to one element
        per observation in the output table

        :param df: A dataframe to explod
        :type df: pandas.DataFrame
        :param column_to_explode:
        :type column_to_explode: str
        :return: An exploded data frame
        :rtype: pandas.DataFrame
        """

        # Create a list of new observations
        new_observations = list()

        # Iterate through existing observations
        for row in df.to_dict(orient='records'):

            # Take out the exploding iterable
            explode_values = row[column_to_explode]
            del row[column_to_explode]

            # Create a new observation for every entry in the exploding iterable & add all of the other columns
            for explode_value in explode_values:
                # Deep copy existing observation
                new_observation = copy.deepcopy(row)

                # Add one (newly flattened) value from exploding iterable
                new_observation[column_to_explode] = explode_value

                # Add to the list of new observations
                new_observations.append(new_observation)

        # Create a DataFrame
        return_df = pd.DataFrame(new_observations)

        # Return
        return return_df

    def annotate_corefs (self, doc, df):
        df['coref'] =  [[] for _ in range(len(df))]

        def element_rest (l):
            for i, e in  enumerate (l):
                yield e, l[:i]+l[i+1:]
        def ref_from_row (r):
            try:
                row = df.query('doc_i in @r')
            except KeyError:
                print ("not found?")

            if row.empty:
                #logging.error("df empty?")
                return "out of the margins"
            return  str(row.s_id.values[0] ) + "->" + "[" + str(row.i.values[0]) + ":" + str(row.i.values[-1]+1) + "]"

            return ",".join(other_sents)

        if doc._.has_coref:
            for cl in doc._.coref_clusters:
                for ment, rest_ments in element_rest (cl):
                    ids = range(ment.start, ment.end)
                    other_sents = [ref_from_row(range(r.start, r.end)) for r in rest_ments]
                    df.loc[df['doc_i'].isin(ids), 'coref'] += other_sents

        df.coref = df.coref.apply (lambda x: ",".join(x) if x else '_')
        return None

    def write_conll_by_df_group(self, x):
        x = x
        conll_lines = []
        for row in x.itertuples():
            conll_lines.append(ConllSpacyUpdater.conll_format % (
                row.i,  # There's a word.i attr that's position in *doc*
                row.text,
                row.lemma,
                row.pos,  # Coarse-grained tag
                row.tag,  # Fine-grained tag
                row.unknown,
                row.head,
                row.dep,  # Relation
                row.corp_id,  # Generation_i
                row.coref))

        conll_path = self.export_dir + '/' + str(row.sent_id) + '.conll'
        with open(conll_path, 'w+') as f:
            f.write ("\n".join (conll_lines) +'\n')

        return None




