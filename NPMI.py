from math import log10
from collections import Counter
from nltk import tokenize
from joblib import Parallel,delayed
from utils import get_nouns
from collections import defaultdict
import numpy as np
from pprint import pprint

def gen_bigrams(filename):
    # generate co-occurrence bigrams in sentence
    for line in open(filename).readlines():
        if line.startswith('parsed_review: '):
            review = line.split('parsed_review: ')[-1][1:-3].split(', ')
            sentences =[]
            n=0
            for ind, term in enumerate(review):
                POS = term.split('/')[-1]
                if POS=='eos':
                    sentences.append (review[n:ind])
                    n = ind+1

            for sen in sentences:
                words = get_nouns(sen)
                for idx in xrange(len(words)):
                    current_word = words[idx]
                    for next_word in words[idx+1:]:
                        yield (current_word, next_word)



def get_sentences_in_review(review):
    sentences = [[] for _ in xrange(review.count('eos'))]
    sent_idx = 0
    for w in review:
        if w == 'eos':
            sent_idx += 1
            continue
        else:
            sentences[sent_idx].append(w)
    return sentences

def  get_doc_count(bigram,all_reviews):
    doc_cnt = 0
    w1,w2 = bigram
    for review in all_reviews:
        if w1 in review and w2 in review:
            sentences_in_review = get_sentences_in_review(review)
            for sent in sentences_in_review:
                sent = set (sent)
                if w1 in sent and w2 in sent:
                    doc_cnt += 1
                    # print doc_cnt
                    # print review
                    # print sent
                    # raw_input()
                    break
    return doc_cnt

def get_bigram_doc_count_dict (filename, all_reviews):
    bigrams = gen_bigrams(filename)
    bigram_doc_count_map = {}
    for bg in bigrams:
        if bg in bigram_doc_count_map.iterkeys():
            #bigram alread processed, see 4th line below
            continue
        doc_cnt = get_doc_count(bg,all_reviews)
        bigram_doc_count_map[bg] = doc_cnt
        bigram_doc_count_map[(bg[1],bg[0])] = doc_cnt
    # print bigram_doc_count_map[('screen', 'life')], \
    #     bigram_doc_count_map[('life', 'screen')]
    # raw_input()
    return bigram_doc_count_map


def get_doc_count_dict(unique_words,filename):
    all_reviews = []
    for paragraph in open(filename).xreadlines():
        if paragraph.startswith('parsed_review: '):
            review = paragraph.split('parsed_review: ')[-1][1:-3].split(', ')
            review = [word.split('/')[0].lower() for word in review]
            all_reviews.append(review)
    word_docfreq_map = {w:0 for w in unique_words}
    for w in unique_words:
        for review in all_reviews:
            if w in set(review):
                word_docfreq_map[w] += 1
    # print word_docfreq_map
    # raw_input()
    return word_docfreq_map, all_reviews



def get_T_statistical_matrix(cand_asps_feats, filename):
    f_xi, all_reviews = get_doc_count_dict(set(cand_asps_feats),
                                           filename)
    fxi_and_xj = get_bigram_doc_count_dict(filename, all_reviews)
    N = float(len(all_reviews))

    all_cooccuring_words = set(fxi_and_xj.keys())

    T = np.zeros(shape=(len(cand_asps_feats),
                        len(cand_asps_feats)))
    for i,wi in enumerate(cand_asps_feats):
        for j,wj in enumerate(cand_asps_feats):
            if i == j:
                T[i, j] = 1.0
            if i < j:
                if (wi,wj) in all_cooccuring_words:
                    fxixj = fxi_and_xj[(wi,wj)]
                else:
                    T[i,j] = 0
                    continue

                nr = log10((N*fxixj)/(f_xi[wi]*f_xi[wj]))
                dr = -log10(fxixj/N)
                npmi = nr/dr
                T[i,j] = npmi
            else:
                T[i, j] = T[j, i]

    T = (T + 1)/2.0
    pprint(T)
    return T

