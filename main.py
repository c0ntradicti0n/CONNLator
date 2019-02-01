import errno
import os
import logging

import edit_distance as edit_distance

logging.getLogger(__name__).addHandler(logging.NullHandler())

import string_preprocessing
import parse_grammar
import conll_and_spacy
import numpy as np
import pandas as pd
from itertools import count

import argparse
parser = argparse.ArgumentParser(description='corpus builder for Sokrates')
parser.add_argument("-t","--txt", help="txt-file to generate new corpus from")
parser.add_argument("-p","--path", help="directory for new corpus")
parser.add_argument("-sd", "--subdivision", nargs="+", help="filter subdivision headings structure")
parser.add_argument("-nc","--no_conll", help="no conll-files")
parser.add_argument("-o", "--overwrite", help="delete some existing corpus and start from scratch")
parser.add_argument("-cbc","--copy_beware_columns", nargs="+", help="only overwrite some columns in the conll-files. The text and the sentence segmentation is inferred by the conll-files, can be list of: %s"% (str(conll_and_spacy.ConllSpacyUpdater.col_set)))
args = parser.parse_args()

import editdistance


def new_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        logging.error("Corpus '%s' already exists" % path)
        if args.overwrite:
            os.rmdir(path)
            os.makedirs(path)
        if e.errno != errno.EEXIST:
            raise

def batch_splitter (X, chunk_width, overlap_margin = None):
    X = np.array(X)
    no_full_chunks, rest_chunk = divmod(len(X), chunk_width)


    if no_full_chunks > 1:
        cm = []
        non_overlapping_intervall = []
        boundaries = [chunk_width * i for i in range(1, no_full_chunks)]

        chunks = np.split(X, boundaries)

        if overlap_margin:
            for ci, c in enumerate(chunks):

               if ci>0:
                   lower_border = overlap_margin
                   under = chunks[ci-1][-lower_border:]
               else:
                   lower_border = 0
                   under = []

               if ci<len(chunks)-1:
                   upper_border = overlap_margin
                   upper = chunks[ci+1][:upper_border]
               else:
                   upper_border = 0
                   upper = []
               chunk_with_marging = list(under) + list(c) + list(upper)
               cm.append(chunk_with_marging)
               non_overlapping_intervall.append ((lower_border,len(chunk_with_marging)-upper_border))
            return cm, non_overlapping_intervall

        return chunks
    else:
        return X


def main():

    # Build new directories
    new_dir(args.path)
    export_path = args.path + "/export_conll"
    new_dir (export_path)
    import_path = args.path + "/import_conll"
    new_dir(import_path)

    StringProcessor = string_preprocessing.Preprocessor()

    # Get source text
    if args.txt:
        with open(args.txt) as f:
            raw_text = f.read()
        all_sentences = StringProcessor.process(raw_text)
    elif args.copy_beware_columns:
        all_sentences, all_conlls = conll_and_spacy.ConllSpacyUpdater.load_all_conlls(export_path)
    else:
        raise OSError ("Either txt or conll must be given to know the text")

    if args.subdivision:
        all_sentences, subdivision_structure = StringProcessor.extract_subdivision_structure(args.subdivision, all_sentences)


    GrammarParser = parse_grammar.GrammarParser()
    ConllUpdater = conll_and_spacy.ConllSpacyUpdater(export_dir=export_path, import_dir=import_path)

    # Fit spacy sparses with bad sentence chunking into well chunked nltk sentences and build the new conlls
    chunk_width = 10
    line_chunks, non_over_lapping_intervall =  batch_splitter(all_sentences, chunk_width, overlap_margin=5)

    s_counter = count(0) # Count generator for
    j = next(s_counter) # Index of sentence in the Corpus

    conll_df = pd.DataFrame()

    # Compute the parts of the corpus als blocks of sentence lists, ignore the margins for coref resolution, handle corefs for each block.
    for i, chintervall  in enumerate(list(zip(line_chunks, non_over_lapping_intervall))):
        corpus_index = chunk_width * i
        ch, intervall = chintervall

        chunk_text = " ".join(ch)
        spacy_neucoref_doc = GrammarParser.process(chunk_text)
        spacy_position = 0

        conll_dict = []

        for ch_j, sentence_from_chunk in enumerate (ch):

                tokens = StringProcessor.tokenize_text_to_words(sentence_from_chunk)
                start_token = tokens[0]
                start_pos = 0
                try:
                    end_pos, last_token = next((i,t) for i, t in list(enumerate(tokens))[::-1] if t not in ['.','?','!'])
                except StopIteration:
                    logging.error ("no last token here? '%s' for sentence no %d: '%s'" % (str(tokens), j, str(sentence_from_chunk)))

                if '/' in last_token:
                    last_token=last_token.split('/')[-1]
                sent_start = conll_and_spacy.find_position_in_doc_by_approx(spacy_neucoref_doc,start_token, spacy_position + start_pos)
                sent_end   = conll_and_spacy.find_position_in_doc_by_approx(spacy_neucoref_doc,last_token, spacy_position + end_pos)

                if tokens [-1] in ['.','?','!']:
                    dot = 1
                else:
                    dot = 0
                sentence_from_spacy = spacy_neucoref_doc[sent_start:sent_end+1+dot]

                spacy_position = sent_start + len(tokens)

                if not ch_j in range(*intervall):
                    continue

                if args.copy_beware_columns:
                    ConllUpdater.conll_over_spacy(sentence_from_spacy, import_path, j, no_cols=args.copy_beware_columns)

                conll_dict.extend(ConllUpdater.export_dict(sentence_from_spacy, index=j))
                j = next(s_counter)

        single_chunk_conll_df = pd.DataFrame(conll_dict)
        ConllUpdater.annotate_corefs(spacy_neucoref_doc, single_chunk_conll_df)
        # Updates the df with the coref-annotations as whole doc block, because the coref annotations are not complete, asking the tokens.

        #single_chunk_conll_df = single_chunk_conll_df.iloc[range(*intervall)]
        conll_df = conll_df.append(single_chunk_conll_df, ignore_index=True)

    # Groupby df sometimes doesn't contain the column, that it is grouped by. Copy this!
    conll_df['sent_id'] = conll_df['s_id']

    # Write all the conll files
    conll_df.groupby(['s_id']).apply(
        lambda x: ConllUpdater.write_conll_by_df_group(x))

    with open(export_path + "/lemmas.txt", 'w+') as f:
        f.write (" ".join(conll_df['lemma'].tolist()))

    with open(export_path + "/subdivision.txt", 'w+') as f:
        f.write (str(subdivision_structure))

    test_fun =  test_equality_of_sentences(all_sentences)
    test_df = conll_df.groupby(['sent_id']).apply(lambda x: test_fun(x))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
       print (test_df)

    return 0


def test_equality_of_sentences(all_sentences):
    def f(x):
        w_list = []
        s = x.s_id.values[0]
        for row in x.itertuples():
            w_list.append(
                row.text
            )
        if editdistance.eval(" ".join(w_list), all_sentences[s]) > 20:
            logging.error("%s in conlls is not %s in string sentence from the start of the prog!" % (
            " ".join(w_list), all_sentences[s]))
            return (False, " ".join(w_list), all_sentences[s])
        else:
            return (editdistance.eval(" ".join(w_list), all_sentences[s]), " ".join(w_list), all_sentences[s])
    return f


if __name__ == '__main__':
    main()