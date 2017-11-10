from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from NPMI import get_T_statistical_matrix
from UMBC import get_G_semantic_matrix
from nltk import WordNetLemmatizer
from utils import *



#prams
most_frequent_words_num = 500
distance_threshold = 0.8
k_aspect_num = 46
w_t = 0.2
w_gt = 0.6
w_g = 1-w_t-w_gt


def get_similarity(index_i, index_j):
    sim_g = cosine_similarity(G[index_i], G[index_j])
    sim_t = cosine_similarity(T[index_i], T[index_j])
    sim_gt = max(sim_g, sim_t)
    sim = w_g*sim_g + w_t*sim_t + w_gt*sim_gt
    return float(sim)

def get_combined_similarity (G,T):
    comb_sim = np.zeros(shape=T.shape)
    for i in xrange(comb_sim.shape[0]):
        for j in xrange(comb_sim.shape[1]):
            comb_sim[i, j] = get_similarity(i, j)
    return comb_sim


def calc_distance(cluster_l, cluster_m,
                  asp_index_map, cand_features_count):
    print asp_index_map
    raw_input()
    sim = 0
    for c_l in cluster_l:
        for c_m in cluster_m:
            sim = sim + (1-get_similarity(asp_index_map[c_l],
                                          asp_index_map[c_m]))
    dist_avg = float(sim/(len(cluster_l)*len(cluster_m)))
    r_c_l = get_most_frequent_member(cluster_l, cand_features_count)
    r_c_m = get_most_frequent_member(cluster_m, cand_features_count)
    dist_rep = 1 - get_similarity(asp_index_map[r_c_l],
                                  asp_index_map[r_c_m])
    distance = max(dist_avg, dist_rep)
    print 'dist: ', cluster_l, cluster_m, dist_avg, dist_rep, distance
    # raw_input()
    return distance



def main():
    dataset_filename = '../data/Lu_Justin_dataset/reviews_Cell-phones.txt'
    gold_standard = '../data/Lu_Justin_dataset/gold_standard/aspectCluster_Cell-phones.txt'
    gold_standard_aspects, gold_standard_features = load_gold_standard(gold_standard)
    aspect_to_features_map = dict(zip(gold_standard_aspects, gold_standard_features))
    print 'loaded {} aspect terms and corresponding features from gold std. file'.format(len(aspect_to_features_map))
    # print 'gold_standard_aspects: ', aspect_to_features_map
    # print '*'*15

    max_freq_aspects = 10
    G_matrix_filename = 'G_matrix_{}.dat'.format(max_freq_aspects)
    T_matrix_filename = 'T_matrix_{}.dat'.format(max_freq_aspects)
    cand_aspects, cand_features = load_CandidateAspTerms(dataset_filename,
                                                   max_freq_aspects)

    cand_features_count = Counter(cand_features)
    clusters = {ind:[asp] for ind, asp in enumerate(cand_aspects)}

    asp_index_map = {word: ind for ind, word in enumerate(cand_aspects)}

    # get the semantic and statistical matrix
    global G, T
    #'''
    G = get_G_semantic_matrix(cand_aspects) #vocab
    # G.dump(G_matrix_filename)
    T = get_T_statistical_matrix(cand_aspects,dataset_filename)  # vocab
    #T.dump(T_matrix_filename)

    # comb_sim = get_combined_similarity (G,T)

    # G = np.load(G_matrix_filename)
    # print G.shape
    # T = np.load(T_matrix_filename)
    # print T.shape

    # while there exist a pair of mergeable cluster
    for cluster_num, current_cluster in enumerate(clusters.itervalues()):
        distance = []
        # try:
        for next_cluster in clusters.values()[cluster_num+1:]:
            dist = calc_distance(current_cluster,
                                 next_cluster,
                                 asp_index_map, cand_features_count)
            if dist < distance_threshold:
                distance.append(dist)
        if distance:
            #mergeable_cluster = clusters[cluster_num + 1 + distance.index(min(distance))]
            clusters[cluster_num + 1 + distance.index(min(distance))] += current_cluster
            del clusters[cluster_num]
        # except:
        #     all clusters processed
        #     pass


    aspects, clusters = get_k_aspects(words, clusters, k_aspect_num)


    
    # Select the closest clusters
    '''
    for x in vocab.remove(frequent_words):
        x = [x]
        for cluster in clusters:
            if calc_distance(G, T, frequent_words, x, cluster)< distance_threshold:
                cluster.extend(x)
    '''
    
    print '*' * 15
    print 'result aspects: ', aspects
    print '*' * 15
    print 'result features: ', clusters
    print '*' * 15

    aspect_N_gold = len(gold_standard_aspects)
    print 'aspect_N_gold', aspect_N_gold
    aspect_N_result = len(aspects)
    print 'aspect_N_result', aspect_N_result
    lemmatizer = WordNetLemmatizer()
    aspect_N_agree = sum([1 for element in aspects if lemmatizer.lemmatize(element) in gold_standard_aspects])
    print 'aspect_N_agree', aspect_N_agree
    aspect_precision = float(aspect_N_agree)/aspect_N_result
    print 'precision', aspect_precision
    aspect_recall = float(aspect_N_agree)/aspect_N_gold
    print 'recall', aspect_recall
    aspect_F_score = float(2*aspect_precision*aspect_recall)/(aspect_precision+aspect_recall)
    print 'F score', aspect_F_score



if __name__ == '__main__':
    main()


