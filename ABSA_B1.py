from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from NPMI import get_T_statistical_matrix
from UMBC import get_G_semantic_matrix
from utils import *
from nltk.corpus import wordnet as wn


#prams
most_frequent_words_num = 100
distance_threshold = 0.7
k_aspect = 5
w_t = 0.2
w_gt = 0.5
w_g = 1-w_t-w_gt


def dump_G_T_matrix(data, output_filename):
    data.dump(output_filename)

def load_G_T_matrix(output_filename):
    return np.load(output_filename)


def get_similarity(G, T, index_i, index_j):
    sim_g = cosine_similarity(G[index_i], G[index_j])
    sim_t = cosine_similarity(T[index_i], T[index_j])
    sim_gt = max(cosine_similarity(G[index_i], G[index_j]), cosine_similarity(T[index_i], T[index_j]))
    sim = w_g*sim_g + w_t*sim_t + w_gt*sim_gt
    return float(sim)


def calc_distance(G, T, vocab, cluster_l, cluster_m):
    sim = float(1.0)
    for c_l in cluster_l:
        for c_m in cluster_m:
            sim = sim*(1-get_similarity(G, T, vocab.index(c_l), vocab.index(c_m)))
    dist_avg = float(sim/(len(cluster_l)*len(cluster_m)))
    r_c_l = [w[0] for w in Counter(cluster_l).most_common(1)]
    r_c_m = [w[0] for w in Counter(cluster_m).most_common(1)]
    print r_c_l, r_c_m
    dist_rep = 1 - get_similarity(G, T, vocab.index(r_c_l[0]), vocab.index(r_c_m[0]))
    distance = max(dist_avg, dist_rep)
    return distance



def main():
    filename = '../data/imdb_labelled.txt'
    G_matrix_filename = 'G_matrix.dat'
    T_matrix_filename = 'T_matrix.dat'
    # vocab = load_vocab_words(filename)
    frequent_words, words = load_frequent_words(filename, most_frequent_words_num)
    frequent_words = [w[0] for w in Counter(words).most_common(20)]

    clusters = [[i] for i in frequent_words]

    # get the semantic and statistical matrix
    #'''
    G = np.array(get_G_semantic_matrix(frequent_words))    # vocab
    #G.dump(G_matrix_filename)
    T = np.array(get_T_statistical_matrix(frequent_words, words, filename))  # vocab
    #T.dump(T_matrix_filename)

    '''
    G = np.load(G_matrix_filename)
    print G.shape
    T = np.load(T_matrix_filename)
    print T.shape
    '''

    # while there exist a pair of mergeable cluster
    for idx, current_cluster in enumerate(clusters):
        distance = []
        for next_cluster in clusters[idx+1:]:
            dist = calc_distance(G, T, frequent_words, current_cluster, next_cluster)  # vocab
            if dist< distance_threshold:
                distance.append(dist)
        if distance == []: continue
        else:
            mergeable_cluster = clusters[idx+1+distance.index(min(distance))]
            current_cluster.extend(mergeable_cluster)
            new_cluster = current_cluster
            clusters.remove(mergeable_cluster)
            clusters.remove(current_cluster)
            clusters.append(new_cluster)

    clusters = get_k_aspects(words, clusters, k_aspect)


    '''
    # Select the closest clusters
    for x in vocab.remove(frequent_words):
        x = [x]
        for cluster in clusters:
            if calc_distance(G, T, frequent_words, x, cluster)< distance_threshold:
                cluster.extend(x)
                continue
    '''
    print clusters








if __name__ == '__main__':
    main()


