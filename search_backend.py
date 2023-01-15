import math
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
from contextlib import closing
from inverted_index_gcp import MultiFileReader
from BM25 import *
from CosineSim import *
import json
import requests
from time import time


bucket_name = "ir-208892166"

def read_posting_list(inverted, w, index_type):
    TUPLE_SIZE = 6
    TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
    with closing(MultiFileReader(bucket_name)) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, index_type)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list

def get_posting_gen(index, query, index_type=""):
    """
    This function returning the generator working with posting list.
    Parameters:
    ----------
    index: inverted index
    """
    w_pls_dict = {}
    for term in query:
        temp_pls = read_posting_list(index, term, index_type)
        w_pls_dict[term] = temp_pls
    return w_pls_dict


def getDocListResult(index, query, index_type, N, titleWeight):
    newDict = {}
    w_pls_dict = get_posting_gen(index, query, index_type)
    for term in query:
        if term in w_pls_dict.keys():
            for doc_id, tf in w_pls_dict[term]:
                newDict[doc_id] = newDict.get(doc_id, 0) + 1

    newDict = {k: v*titleWeight for k,v in newDict.items()}
    sorted_docs = sorted(newDict.items(), key=lambda x: x[1], reverse=True)[:N]
    return sorted_docs

def getDocListResultWithPageRank(index, query, index_type, N, pageRank,titleWeight):
    newDict = {}
    w_pls_dict = get_posting_gen(index, query, index_type)
    for term in query:
        if term in w_pls_dict.keys():
            for doc_id, tf in w_pls_dict[term]:
                newDict[doc_id] = newDict.get(doc_id, 0) + 1

    maxPage = pageRank["3434750"]
    newDict = {k: v*titleWeight + pageRank.get(k,0) / maxPage for k,v in newDict.items()}
    sorted_docs = sorted(newDict.items(), key=lambda x: x[1], reverse=True)[:N]
    return sorted_docs

def combineListTuplesIntoDict(bm25List=None, cosineList=None, binaryList=None, pageRank=None):

    maxPage = pageRank["3434750"]

    args = [bm25List, cosineList, binaryList]
    merged_dict = defaultdict(float)
    mergeL = []
    for arg in args:
        if arg is not None:
            mergeL += arg
    for doc_id, score in mergeL:
        merged_dict[doc_id] += score

    if pageRank is not None:
        merged_list = [(key, value + pageRank.get(key,0) / maxPage) for key, value in merged_dict.items()]
    else:
        merged_list = merged_dict.items()

    return merged_list

def examineWeights(query_tokens, bm25Algo, cosineAlgo, doc_title_dict, indexTile, pageRank):

    resultsList = []

    bm25Weights = np.arange(0, 1, 0.05)

    for bm25Weight in bm25Weights():
        binaryWeight = 1 - bm25Weight
        resBM25Body = bm25Algo.search(query_tokens, 100, bm25Weight)
        resTitle = getDocListResult(indexTile, query_tokens, "_title_stem", 100, binaryWeight)
        res = combineListTuplesIntoDict(bm25List=resBM25Body, cosineList=None, binaryList=resTitle,
                                        pageRank=None)
        res = [(str(doc_id), doc_title_dict[doc_id]) for doc_id, score in res]

        resultsList.append(res)

    return resultsList



def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)

