
import nltk
import nltk.translate.gleu_score as gleu
import nltk.translate.bleu_score as bleu

import numpy
import os
import argparse

__author__ = "Gwena Cunha"


""" Class that provides methods to calculate similarity between two files
    Each line can be composed of multiple sentences
    
    python evaluate.py -hyp pred_en-de.txt -ref data/multi30k/test.de -out scores_en-de.txt
"""

# Constants
BLEU_NAME = "BLEU"
GOOGLE_BLEU_NAME = "GLEU"  # "Google-BLEU"
WER_NAME = "WER"


def project_dir_name():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    project_dir = os.path.abspath(current_dir + "/../") + "/"
    return project_dir


class TextScore:

    def __init__(self):
        print("Initialize Machine Translation text score")

        # Needed to separate sentences in CONTENT
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def score_multiple_from_file(self, ref_file, hyp_file, scores_file, score_type=BLEU_NAME, average_prec="corpus"):
        # Clean scores_file if existent
        open(scores_file, 'w').close()

        scores = []
        if BLEU_NAME in score_type:
            scores.append(self.score_one_from_file(ref_file, hyp_file, scores_file, score_type=BLEU_NAME, average_prec=average_prec))

        if GOOGLE_BLEU_NAME in score_type:
            scores.append(self.score_one_from_file(ref_file, hyp_file, scores_file, score_type=GOOGLE_BLEU_NAME, average_prec=average_prec))

        if WER_NAME in score_type:
            scores.append(self.score_one_from_file(ref_file, hyp_file, scores_file, score_type=WER_NAME, average_prec=average_prec))

        return scores

    def score_one_from_file(self, ref_file, hyp_file, scores_file, score_type="BLEU", average_prec="corpus"):
        """ Calculates score of file where each re line is a text corresponding to the same hyp line
        Doesn't treat cases of multiple references for the same hypotheses

        :param ref_file: text file of reference sentences
        :param hyp_file: text file of sentences generated by model
        :param scores_file: text file with scores
        :param score_type: BLEU, Google-BLEU, GLEU, WER, TER
        :param average_prec: "corpus", "sent_average" or both ("corpus sent_average")
        :return: final score
        """

        hf = open(hyp_file, "r")
        hypothesis = hf.read().split("\n")
        num_sentences = len(hypothesis) - 1

        rf = open(ref_file, "r")
        reference = rf.read().split("\n")

        sf = open(scores_file, "a+")

        list_of_references = []
        hypotheses = []
        real_num_sentences = 0
        for i in range(0, num_sentences):
            if len(reference[i].strip()) != 0 or len(hypothesis[i].strip()) != 0:
                # Previous: split line which has multiple sentences into words
                # ref, hypo = reference[i].lower().split(), hypothesis[i].lower().split()
                # Current: added a for loop to separate sentences in each line
                hypothesis_i = hypothesis[i].split('</s>')[0]
                sentences_ref = nltk.sent_tokenize(reference[i].lower())
                sentence_hyp = nltk.sent_tokenize(hypothesis_i.lower())
                for sent_ref, sent_hyp in zip(sentences_ref, sentence_hyp):
                    ref, hypo = sent_ref.split(), sent_hyp.split()
                    list_of_references.append([ref])
                    hypotheses.append(hypo)
                    real_num_sentences += 1

        print("Sentences: " + str(real_num_sentences))
        scores_str = ""
        score_corpus, score_sent = None, None

        # Corpus: only relevant for BLEU and GLEU (Google-BLEU)
        if "corpus" in average_prec and (WER_NAME not in score_type):
            score_corpus = self.corpus_score(list_of_references, hypotheses, score_type=score_type)
            scores_str += score_type + " corpus: " + str(format(score_corpus, '.4f')) + "\n"
        if "sent_average" in average_prec:
            score_sent = self.sentence_average_score(list_of_references, hypotheses, score_type=score_type)
            scores_str += score_type + " sent_average: " + str(format(score_sent, '.4f')) + "\n"

        scores_str += "\n"
        sf.write(scores_str)
        sf.close()

        return score_corpus, score_sent

    def corpus_score(self, list_of_references, hypotheses, score_type="BLEU"):
        """ Score specifically implemented for corpus

        :param list_of_references: list of reference texts
        :param hypotheses: hypotheses relative to reference
        :param score_type: metric being used
        :return: corpus score
        """

        corpus_score = None
        if BLEU_NAME in score_type:
            corpus_score = bleu.corpus_bleu(list_of_references, hypotheses)
        elif GOOGLE_BLEU_NAME in score_type:
            corpus_score = gleu.corpus_gleu(list_of_references, hypotheses)

        print("%s corpus score: %.4f" % (score_type, corpus_score))
        return corpus_score

    def sentence_average_score(self, list_of_references, hypotheses, score_type="BLEU"):
        """ Averages score applied for every sentence

        :param list_of_references: list of reference texts (separated into words)
        :param hypotheses: hypotheses relative to reference (separated into words)
        :param score_type: metric being used
        :return: average sentences score
        """

        sent_average_score = 0
        if BLEU_NAME in score_type:
            for ref, hyp in zip(list_of_references, hypotheses):
                sent_average_score += bleu.sentence_bleu(ref, hyp)  # gram: default is between 1 and 4
        elif GOOGLE_BLEU_NAME in score_type:
            for ref, hyp in zip(list_of_references, hypotheses):
                sent_average_score += gleu.sentence_gleu(ref, hyp)  # gram: default is between 1 and 4
        elif WER_NAME in score_type:
            for ref, hyp in zip(list_of_references, hypotheses):
                sent_average_score += self.wer_score(ref[0], hyp)  # Assumes only 1 reference

        sent_average_score /= len(list_of_references)

        print("%s sentence average score: %.4f" % (score_type, sent_average_score))
        return sent_average_score

    def wer_score(self, ref, hyp):
        """ Calculation of WER with Levenshtein distance.

        Time/space complexity: O(nm)

        Source: https://martin-thoma.com/word-error-rate-calculation/

        :param ref: reference text (separated into words)
        :param hyp: hypotheses text (separated into words)
        :return: WER score
        """

        # Initialization
        d = numpy.zeros([len(ref) + 1, len(hyp) + 1], dtype=numpy.uint8)
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # print(d)

        # Computation
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        # print(d)
        return d[len(ref)][len(hyp)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-hyp', required=True,
                        help='Path to candidate document')
    parser.add_argument('-ref', required=True,
                        help='Path to reference document')
    parser.add_argument('-out', required=True,
                        help='Path to output text file with calculated scores')

    args = parser.parse_args()
    # params = vars(args)

    print("Test score file")

    # Initialize handler for text scores
    text_score = TextScore()
    text_score.score_multiple_from_file(ref_file=args.ref, hyp_file=args.hyp, scores_file=args.out,
                                        score_type=BLEU_NAME + GOOGLE_BLEU_NAME + WER_NAME,
                                        average_prec="corpus, sent_average")
