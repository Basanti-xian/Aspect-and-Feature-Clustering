import re
from collections import Counter
from nltk.corpus import stopwords
import ast


stop = set(stopwords.words('english'))
POSTagsToLookFor = ['NN', 'NNP', 'JJ']

def load_CandidateAspTerms(filename, most_commom_num=20):
    """
    get the candidate nouns aspects

    :param filename, most_commom_num: num of the most frequent words
    :return: frequent_CandidateAspTerms, CandidateAspTerms (list of string)
    """
    f = open(filename).xreadlines()
    CandidateAspTermsPos = []
    for sen in f:
        if sen.startswith('parsed_review: '):
            sen = sen.split('parsed_review: ')[-1][1:-3].strip().split(', ')
            try:
                for Term in sen:
                    AspTerm, POS = Term.split('/')
                    if POS in POSTagsToLookFor:
                        CandidateAspTermsPos.append ((AspTerm.strip().lower(), POS))
                    #elif POS.startswith ('VB'):
                    #    CandidateAspTermsPos.append ((AspTerm.strip().lower(), POS))
                    else:
                        continue
            except:
                pass

    CandidateAspTerms = [Term for Term, Pos in CandidateAspTermsPos if (Term not in stop) and (len(Term)>2)]
    frequent_CandidateAspTerms = [w[0] for w in Counter(CandidateAspTerms).most_common(most_commom_num)]
    print frequent_CandidateAspTerms
    return frequent_CandidateAspTerms, CandidateAspTerms


def get_nouns(data):
    """
    :param data (list of string)
    :return: NounTerms (list of string)
    """
    NounTerms = []
    try:
        for Term in data:
            Asp, POS = Term.split('/')
            if POS in POSTagsToLookFor:
                NounTerms.append (Asp.strip().lower())
            else:
                continue
    except:
        pass
    NounTerms = [Term for Term in NounTerms if Term not in stop]
    return NounTerms



def load_gold_standard(filename):
    lines = [l.strip() for l in open(filename).xreadlines()]
    aspects = [l.split('\t')[0] for l in lines]
    features = [l.split('\t')[1][1:-1] for l in lines]
    features = [fstring.split(', ') for fstring in features]
    return aspects,features

def stopwords_tokens(data):
    validwords = []
    for i in data:
      if i not in stop and not i.isdigit():
          validwords.append(i)
    return validwords


def get_k_aspects(freq_of_all_words, clusters, k):
    """
    :param words: all the candidate aspects
    :param clusters,k
    :return: k_most_frequent_aspects, k_most_frequent_clusters
    """
    cluster_count = []
    aspects = []
    for cluster in clusters:
        value = [freq_of_all_words.get(element) for element in cluster]
        aspects.append(cluster[value.index(max(value))])
        cluster_count.append(sum(value))
    if k < len(clusters):
        cluster_map = zip(cluster_count, clusters, aspects)
        k_most_map = sorted(cluster_map, reverse=True, key=lambda x: x[0])[:k]
        k_most_frequent_clusters = [w[1] for w in k_most_map]
        k_most_frequent_aspects = [w[2] for w in k_most_map]
        return k_most_frequent_aspects, k_most_frequent_clusters
    else:
        return aspects, clusters

def get_most_frequent_member(cluster, freq_of_all_words):
    #get the most frequent member for the cluster.
    if len(cluster) == 1:
        #singleton cluster
        return cluster[0]

    freq_counts_of_mems_in_cluster = [freq_of_all_words[mem]
                                       for mem in cluster]
    max_freq = max(freq_counts_of_mems_in_cluster)
    most_frequent_member = cluster[freq_counts_of_mems_in_cluster.index(max_freq)]
    return most_frequent_member