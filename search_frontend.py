import concurrent.futures
from flask import Flask, request, jsonify
import pickle
from google.cloud import storage
from nltk.corpus import stopwords

import BM25
from Tokenizer import Tokenizer
from search_backend import *
from BM25 import *
import csv
import pandas as pd
import re
from CosineSim import *
import numpy as np

class MyFlaskApp(Flask):

    def run(self, host=None, port=None, debug=None, **options):
        bucket_name = "ir-208892166"
        self.tokenizer = Tokenizer()
        client = storage.Client()
        self.my_bucket = client.bucket(bucket_name=bucket_name)
        self.page_rank = {}
        self.tempDict = {}
        for blob in client.list_blobs(bucket_name):
            if blob.name == "body_index.pkl":
                with blob.open('rb') as openfile:
                    self.index_body = pickle.load(openfile)

            elif blob.name == "title_index.pkl":
                with blob.open('rb') as openfile:
                    self.index_title = pickle.load(openfile)

            elif blob.name == "anchor_index.pkl":
                with blob.open('rb') as openfile:
                    self.index_anchor = pickle.load(openfile)

            elif blob.name == "body_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.body_stem_index = pickle.load(openfile)

            elif blob.name == "title_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.title_stem_index = pickle.load(openfile)

            elif blob.name == "body_DL.pkl":
                with blob.open('rb') as openfile:
                    self.DL_body = pickle.load(openfile)

            elif blob.name == "title_DL.pkl":
                with blob.open('rb') as openfile:
                    self.DL_title = pickle.load(openfile)

            elif blob.name == "doc2vec.pkl":
                with blob.open('rb') as openfile:
                    self.doc_title_dict = pickle.load(openfile)

            elif blob.name == "doc_norm_size.pkl":
                with blob.open('rb') as openfile:
                    self.doc_norm = pickle.load(openfile)

            elif blob.name == "pageRank.pkl":
                with blob.open('rb') as openfile:
                    self.tempDict = pickle.load(openfile)
                    self.page_rank = {str(k): v for k, v in self.tempDict.items()}

            elif blob.name == "pageViews.pkl":
                with blob.open('rb') as openfile:
                    self.doc_norm = pickle.load(openfile)

            elif blob.name == "pageViews.pkl":
                with blob.open('rb') as openfile:
                    self.page_views = pickle.load(openfile)

        self.BM25_body = BM25(self.body_stem_index, self.DL_body, "_body_stem", app.page_rank, k1=1.5, b=0.6)
        self.BM25_title = BM25(self.title_stem_index, self.DL_title,  "_title_stem", app.page_rank, k1=2.2, b=0.85)
        self.cosine_body = CosineSim(self.body_stem_index, self.DL_body, "_body_stem", app.page_rank, self.doc_norm)
        self.cosine_title = CosineSim(self.title_stem_index, self.DL_title, "_title_stem", app.page_rank, self.doc_norm)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank.py, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = list(set(app.tokenizer.tokenize(query, True)))

    resBM25Body = app.BM25_body.search(query_tokens, 100, 0.25)
    resTitle = getDocListResult(app.title_stem_index, query_tokens, "_title_stem", 100, 0.75)
    res = merge_results(resTitle, resBM25Body, 100)
    res = [(str(doc_id), app.doc_title_dict[doc_id]) for doc_id, score in res]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = []
    res_list = app.cosine_body.calcCosineSim(app.tokenizer.tokenize(query, False), app.index_body, app.DL_body, N=100)
    res_list = [(doc_id, app.doc_title_dict[doc_id]) for doc_id, score in res_list]
    res = res_list
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = getDocListResult(app.index_title, app.tokenizer.tokenize(query, False), "_title", 100, 1)
    res = [(str(doc_id), app.doc_title_dict[doc_id]) for doc_id, score in res]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = getDocListResult(app.index_anchor, app.tokenizer.tokenize(query, False), "_anchor", 100, 1)
    res = [(str(doc_id), app.doc_title_dict[doc_id]) for doc_id, score in res]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank.py values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank.py scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [app.page_rank[wiki_id] for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [app.page_views[wiki_id] for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_test_titles")
def search_test_titles():

    with open('new_train.json', 'rt') as f:
        queries = json.load(f)



    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        query_tokens = list(set(app.tokenizer.tokenize(q, True)))

        t_start = time()

        resBM25 = app.BM25_title.search(query_tokens, 100, 1)
        # future_title = app.executor.submit(app.BM25_title.search,  query_tokens, 100)
        # resBM25 = merge_results(future_body.result(), future_title.result(), title_weight=0, text_weight=0.2, N=100)
        resCosBody = app.cosine_title.calcCosineSim(query_tokens, app.index_body, 100, 1)
        resTitle = getDocListResultWithPageRank(app.title_stem_index, query_tokens, "_title_stem", 100, 0.75, app.page_rank)
        res = merge_results(resTitle, resBM25Body, 100)

        pred_wids = [tup[0] for tup in res]
        res = [(str(doc_id), app.doc_title_dict[doc_id]) for doc_id, score in res]
        duration = time() - t_start

        ap = average_precision(true_wids, pred_wids)

        qs_res.append((q, duration, ap))

    averageAP = sum([tup[2] for tup in qs_res if tup[2] is not None]) / len(qs_res)
    averageTime = sum([tup[1] for tup in qs_res if tup[1] is not None]) / len(qs_res)

    file_name = f"TitleWeight={0.75}, BodyWeight={0.25}, duration={averageTime}, ap={averageAP}, time={time()}, PageRank Only Test"
    blob = app.my_bucket.blob(f"Title+Body/{file_name}")  # change depndes on path
    blob.upload_from_string(file_name)

    return jsonify(res)

@app.route("/search_test_body_title")
def search_test_body_title():

    with open('new_train.json', 'rt') as f:
        queries = json.load(f)



    qs_res = []
    queries = list(queries.items())[-10:]
    for q, true_wids in queries:
        duration, ap = None, None
        query_tokens = list(set(app.tokenizer.tokenize(q, True)))

        t_start = time()

        resBM25Body = app.BM25_body.search(query_tokens, 100, 0.25)
        # future_title = app.executor.submit(app.BM25_title.search,  query_tokens, 100)
        # resBM25 = merge_results(future_body.result(), future_title.result(), title_weight=0, text_weight=0.2, N=100)
        # resCosBody = app.cosine_body.calcCosineSim(query_tokens, app.index_body, 100, 0.25)
        resTitle = getDocListResultWithPageRank(app.title_stem_index, query_tokens, "_title_stem", 100, 0.75, app.page_rank)
        res = merge_results(resTitle, resBM25Body, 100)
        pred_wids = [tup[0] for tup in res]
        res = [(str(doc_id), app.doc_title_dict[doc_id]) for doc_id, score in res]

        duration = time() - t_start

        ap = average_precision(true_wids, pred_wids)

        qs_res.append((q, duration, ap))

    averageAP = sum([tup[2][0] for tup in qs_res if tup[2][0] is not None]) / len(qs_res)
    averageRecall = sum([tup[2][1] for tup in qs_res if tup[2][1] is not None]) / len(qs_res)
    averageTime = sum([tup[1] for tup in qs_res if tup[1] is not None]) / len(qs_res)

    file_name = f"TitleWeight={0.75}, BodyWeight={0.25}, duration={averageTime}, ap={averageAP}, recall={averageRecall}"
    blob = app.my_bucket.blob(f"Title+Body/{file_name}")  # change depndes on path
    blob.upload_from_string(file_name)

    return jsonify(res)

@app.route("/search_test_bm25body")
def search_test_bm25body():

    with open('new_train.json', 'rt') as f:
        queries = json.load(f)

    BM251 = BM25(app.body_stem_index, app.DL_body, "_body_stem", k1=1.5, b=0.65)
    BM252 = BM25(app.body_stem_index, app.DL_body, "_body_stem", k1=1.4, b=0.6)
    BM253 = BM25(app.body_stem_index, app.DL_body, "_body_stem", k1=1.8, b=0.6)

    finList = []

    bmList = [BM251, BM252, BM253]

    for bm in bmList:
        qs_res = []
        for q, true_wids in queries.items():
            duration, ap = None, None
            query_tokens = list(set(app.tokenizer.tokenize(q, True)))

            t_start = time()
            res = bm.search(query_tokens, 100, 1)
            pred_wids = [tup[0] for tup in res]
            res = [(str(doc_id), app.doc_title_dict[doc_id]) for doc_id, score in res]
            duration = time() - t_start

            ap = average_precision(true_wids, pred_wids)

            qs_res.append((q, duration, ap))

        averageAP = sum([tup[2] for tup in qs_res if tup[2] is not None]) / len(qs_res)
        averageTime = sum([tup[1] for tup in qs_res if tup[1] is not None]) / len(qs_res)

        finList.append(f"{bm}, {averageTime}, {averageAP}")

        # file_name = f"TitleWeight={0.75}, BodyWeight={0.25}, duration={averageTime}, ap={averageAP}, time={time()}, PageRank"
        # blob = app.my_bucket.blob(f"Title+Body/{file_name}")  # change depndes on path
        # blob.upload_from_string(file_name)

    return jsonify(finList)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
