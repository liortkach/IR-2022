import math

import numpy as np
from contextlib import closing
from inverted_index_gcp import MultiFileReader


def get_candidate_documents(query_to_search, words, pls):
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            current_list = (pls[words.index(term)])
            candidates += current_list
    candidates = [i[0] for i in candidates]
    return np.unique(candidates)

class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    index: inverted index
    """

    def __init__(self, index, DL, index_type, page_rank, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.page_rank = page_rank
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.index_type = index_type
        self.bucket_name = "ir-208892166"

    def read_posting_list(self, index, w):
        TUPLE_SIZE = 6
        TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        with closing(MultiFileReader(self.bucket_name)) as reader:
            locs = index.posting_locs[w]
            b = reader.read(locs, index.df[w] * TUPLE_SIZE, self.index_type)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))

        return posting_list

    def get_posting_gen(self, index, query):
        """
        This function returning the generator working with posting list.
        Parameters:
        ----------
        index: inverted index
        """
        w_pls_dict = {}
        for term in query:
            temp_pls = self.read_posting_list(index, term)
            w_pls_dict[term] = temp_pls
        return w_pls_dict

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, N=3, bm25Weight=0.333):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        w_pls_dict = self.get_posting_gen(self.index, query)
        words = tuple(w_pls_dict.keys())
        pls = tuple(w_pls_dict.values())
        term_frequencies_dict = {}
        for term in query:
            if term in self.index.df:
                term_frequencies_dict[term] = dict(pls[words.index(term)])
        candidates = []
        for term in np.unique(query):
            if term in words:
                current_list = (pls[words.index(term)])
                candidates += current_list
        candidates = np.unique([c[0] for c in candidates])
        return sorted([(doc_id, round(self._score(query, doc_id, term_frequencies_dict)*bm25Weight, 5)) for doc_id in candidates],key=lambda x: x[1], reverse=True)[:N]

    def _score(self, query, doc_id, term_frequencies_dict):
        """
        This function calculate the bm25 score for given query and document.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        idf = self.calc_idf(query)
        # maxPageRank = self.page_rank["3434750"]
        if doc_id not in self.DL.keys():
            return -math.inf
        doc_len = self.DL[doc_id]
        for term in query:
            if doc_id in term_frequencies_dict[term]:
                freq = term_frequencies_dict[term][doc_id]
                numerator = idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))

                score += (numerator / denominator) #+ self.page_rank.get(doc_id, 0) / maxPageRank

        return score


def merge_results(title_scores, body_scores, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).
    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.
    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    merged_dict = {}

    for item in title_scores + body_scores:
        key, value = item
        merged_dict[key] = merged_dict.get(key, 0) + value

    return sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)[:N]
