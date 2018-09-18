# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:22:38 2018

@author: sterlinm.hub
"""

import numpy as np
import pandas as pd
import datetime, nltk, re, itertools, jsonpickle,json, copy,sys,os
from github import Github, GithubException, InputGitTreeElement
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from scipy.sparse import csc_matrix
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, LsiModel, TfidfModel
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans,SpectralClustering,DBSCAN,Birch,AgglomerativeClustering
from sklearn.metrics import pairwise
from sklearn.manifold import MDS, TSNE
from tqdm import tqdm
from gensim.sklearn_api import TfIdfTransformer
import matplotlib.pyplot as plt
from time import sleep
nltk.download('stopwords')
from modules.METAFILE import METAFILE


class QUANTLET:
    def __init__(self, github_token=None, user=None):
        """Constructor of the QUANTLET class.

        Arguments:
        github_token -- (string) Add Github token to have higher access limits or to access private reposotories, default: None.
        user -- (string) user/organization name in which Quantlets shall be searched for, if None user associated to github_token is used here, default: None.
        """
        assert user is not None or github_token is not None, 'Either github_token or user have to be not none!'
        self.quantlets = dict()
        self.repos = dict()
        self.github_token = github_token
        if user is None:
            self.g = Github(github_token).get_user()
        else:
            self.g = Github(github_token).get_user(user)
        self.errors = []

    def stop_until_rate_reset(self,at_least_remaining=None):
        """Checks the limit rate that is given by Github and pauses function if rate is too small.

        at_least_reamining -- (int) minimum number of api calls too remain, default: None. If None it is set to 0.
        """

        rate = Github(self.github_token).get_rate_limit().rate

        if at_least_remaining is None:
            at_least_remaining = 0
        assert isinstance(at_least_remaining,int)
        if rate.remaining <= at_least_remaining:
            print('\nPause until around %s' % (rate.reset.strftime('%Y-%m-%d %H:%M:%S')))
            while True:
                rate = Github(self.github_token).get_rate_limit().rate
                if rate.remaining > at_least_remaining:
                    break
                t = rate.reset - datetime.datetime.utcnow() + datetime.timedelta(seconds=1)
                sleep(max([np.ceil(t.total_seconds()), 120]))
            print('\nPause end\n')
        return True
    def __download_metafiles_from_repository(self, repo, server_path='.', override=None):
        """Downloads all Quantlets within the server_path of a repository repo or in a subfolder.

        repo -- repository
        server_path -- path within repo
        override -- override existing Metafile information already saved, default: None.
        """
        self.stop_until_rate_reset(0)
        try:
            # get repo content from directory server_path
            contents = repo.get_dir_contents(server_path)
        except GithubException as e:
            if e.args[0] == 404 and e.args[1]["message"] == 'This repository is empty.':
                return ()
            self.errors.append([repo.name, server_path, e])
            return ()
        for content in contents:
            if content.type == 'dir':
                self.__download_metafiles_from_repository(repo, server_path=content.path)
            elif content.name.lower() == 'metainfo.txt':
                key = '/'.join([repo.name, content.path])
                tmp_bool = key in self.quantlets.keys()
                if (not tmp_bool) or (tmp_bool and override):
                    print('\t%s' % (content.path))
                    commits_all = list(repo.get_commits(path='/'.join(content.path.split('/')[:-1])))
                    commits = [commits_all[0], commits_all[-1]]
                    self.quantlets.update({key: METAFILE(file=content, repo=repo, content=contents,commits=commits)})
                    self.repos.update({repo.name: repo})
    def update_existing_metafiles(self):
        """Searches for Quantlets that have been changed and updates the saved information."""
        repos2update = self.get_recently_changed_repos(since=self.get_last_commit())
        qs = {k:v for k,v in self.quantlets.items() if v.repo_name in repos2update}
        if not qs:
            return None
        repo = self.g.get_repo(qs[list(qs.keys())[0]].repo_name)
        for k,v in tqdm(qs.items()):
            self.stop_until_rate_reset(0)
            if v.repo_name != repo.name:
                repo = self.g.get_repo(v.repo_name)
            path = v.directory.lstrip(v.repo_name).lstrip('/')
            commits = repo.get_commits(path=path)
            if not commits.get_page(0):
                del self.quantlets[k]
                continue
            if commits.get_page(0)[0].sha != v.commit_last['sha']:
                contents = repo.get_dir_contents(path)
                content = [i for i in contents if i.name.lower() == 'metainfo.txt'][0]
                self.quantlets[k] = METAFILE(file=content, repo=repo, content=contents,commits=commits)
    def update_all_metafiles(self,since=None):
        """Updates all metafiles, thus executes update_existing_metafiles and searches for new Quantlets.

        since -- (datetime.datetime) Searches for Quantlets changed after that time point.
        """
        self.update_existing_metafiles()
        if since is None:
            since = self.get_last_commit()
        repos = self.get_recently_changed_repos(since=since)
        if repos:
            self.download_metafiles_from_user(repos,override=False)
    def download_metafiles_from_user(self, repo_name=None, override=True):
        """ Downlaod repositories with name repo_name in it.

        Keyword arguments:
        repo_name -- name of repositories to be downloaded (String or List of String, default None)
        override -- if True overriding existing metainfo files (default: True)
        """
        if repo_name is None:
            repos = self.g.get_repos()
        else:
            if not isinstance(repo_name, (list, tuple)):
                repo_name = [repo_name]
            repos = []
            for _, _n in enumerate(repo_name):
                repos.append(self.g.get_repo(_n))
        for repo in repos:
            print('%s' % (repo.name))
            self.__download_metafiles_from_repository(repo, '.', override=override)
    def save(self, filepath):
        """Saves the class instance of QUANTLET into a json file (using jsonpickle).

        filepath -- (str) specifies location and name where the file is saved.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(jsonpickle.encode(self))
    def load(filepath):
        """Loads saved class instance of QUANTLET from file and returns class instance.

        filepath -- (str) filename of saved class instance
        """
        with open(filepath, 'r') as f:
            output = f.read()
        return jsonpickle.decode(output)
    def grading(self, save_path=None,grades_equals=None):
        """ Extracts the grading information from the metainfo files and optionally saves them to csv

        Arguments:
        save_path -- if not None it saves the grading information to a csv file (default None)
        """
        ret = []
        for _, v in self.quantlets.items():
            d = dict()
            if v.is_debuggable:
                d.update(v.grading_output)
                d.update({'q_directory': v.directory, 'author': v.metainfo_debugged['author'], 'repo': v.repo_name})
            else:
                d.update({'q_directory': v.directory, 'q_quali': 'F', 'comment': 'Not debuggable', 'repo': v.repo_name})
            ret.append(d)
        grades = pd.DataFrame(ret)
        if save_path is not None:
            if grades_equals is None:
                name = save_path + '_' + datetime.datetime.now().strftime('%Y%m%d') + '.csv'
                grades.to_csv(name)
            else:
                name = save_path + '_' + ''.join(grades_equals) + '_' + datetime.datetime.now().strftime('%Y%m%d') + '.csv'
                grades.loc[grades['q_quali'].isin(grades_equals)].to_csv(name)
        return grades
    def get_last_commit(self):
        """Returns the time point of the last commit in the data.
        """
        last_commit_in_quantlet = sorted([v.commit_last['commit']['author']['date'] for k, v in self.quantlets.items()])[-1]
        last_commit = datetime.datetime.strptime(last_commit_in_quantlet, '%Y-%m-%dT%H:%M:%SZ')
        return last_commit
    def get_recently_changed_repos(self, since):
        """Returns list of repository of repositories which were changed after since.

        since -- (datetime.datetime) time point after which changes have not yet been included in data.
        """
        assert isinstance(since, datetime.datetime), \
            "Variable since must be of type datetime.datetime, e.g. datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')"
        since += datetime.timedelta(seconds=1)
        self.stop_until_rate_reset(0)
        ret = []
        for repo in tqdm(list(self.g.get_repos())):
            self.stop_until_rate_reset(0)
            try:
                if repo.get_commits(since=since).get_page(0):
                    ret.append(repo.name)
            except GithubException as e:
                if e.args[0] == 409:
                    pass
                else:
                    raise
        return ret
    def create_readme(self,repos=None):
        """Creates readmes in the repositories in the repository list repos.

        repos -- (list of repositories)  default: None, if None all repositories of user are used.
        """
        def __readme_template(name_of_quantlet, metainfo_original, pics, quantlet):
            """README template.

            Key arguments:
                name_of_quantlet -- name of the quantlet (string)
                metainfo_origional -- metainfo file content (string)
                pics -- list of strings containing the names of the pictures

            """
            strl = [
                '[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)']
            strl.append(
                '## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **')
            strl[-1] += name_of_quantlet
            strl[-1] += '** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)'
            strl.append('```yaml')
            strl.append(metainfo_original + '\n```')
            for i, n in enumerate(pics):
                strl.append('![Picture%i](%s)' % (i + 1, n))
            for i in quantlet:
                lang = i.split('.')[-1].lower()
                if lang == 'py':
                    lang = 'python'
                elif lang == 'm':
                    lang = 'matlab'
                strl.append('### %s Code\n```%s' % (lang.upper(), lang))
                strl.append([j for j in contents if j.name == i][0].decoded_content.decode() + '\n```')
                strl.append('automatically created on %s' % datetime.datetime.today().strftime('%Y-%m-%d'))
            readme = '\n\n'.join(strl)
            return readme

        self.stop_until_rate_reset(0)
        if repos is None:
            repos = list(self.g.get_repos())

        for repo in tqdm(repos):
            self.stop_until_rate_reset(0)
            qs = {k:v for k,v in self.quantlets.items() if repo.name == v.repo_name}
            for k,v in qs.items():
                try:
                    self.stop_until_rate_reset(0)
                    contents = repo.get_contents(v.directory.lstrip(v.repo_name))
                    if [i for i in contents if 'README.md'.lower() == i.name.lower()]:
                        continue
                    if not v.is_debuggable:
                        continue
                    quantlet = [i.name for i in contents if '.'.join(i.name.split('.')[:-1]).lower() == v.metainfo_debugged[
                        'name of quantlet'].lower() and i.name.split('.')[-1].lower() in ['r', 'm', 'sas', 'py']]
                    pics = sorted([i.name for i in contents if i.name.rsplit('.')[-1] in ['png', 'jpg', 'jpeg']])
                    readme = __readme_template(name_of_quantlet=v.metainfo_debugged['name of quantlet'],
                                               metainfo_original=v.metainfo_undebugged, pics=pics, quantlet=quantlet)
                except:
                    continue
                try:
                    commit_message = 'created README.md (automatically)'
                    master_ref = repo.get_git_ref('heads/master')
                    master_sha = master_ref.object.sha
                    base_tree = repo.get_git_tree(master_sha)
                    # see https://developer.github.com/v3/git/trees/#create-a-tree
                    #                    element = InputGitTreeElement('/'.join(b.path.split('/')[:-1]) +'/'+'README.md', '100644', 'blob', readme.decode())
                    #                        element = InputGitTreeElement(v.directory +'/'+'README.md', '100644', 'blob', readme)
                    element = InputGitTreeElement(v.directory.lstrip(v.repo_name).lstrip('/') + '/' + 'README.md',
                                                  '100644', 'blob', readme)
                    tree = repo.create_git_tree([element], base_tree)
                    parent = repo.get_git_commit(master_sha)
                    commit = repo.create_git_commit(commit_message, tree, [parent])
                    master_ref.edit(commit.sha)
                except:
                    continue
    def get_corpus_dictionary(self, leave_tokens_out_with_less_than_occurence=1,
                              leave_tokens_out_with_ratio_of_occurence=1,
                              include_keywords=True,include_description=False,include_whole_metainfo=False,
                              with_bigrams = False, with_trigrams = False):
        """ Returns the corpus and dictionary from the selected text base (created with Gensim).

        Keyword arguments:
        leave_tokens_out_with_less_than_occurence -- Leaves out tokens which have less than 'leave_tokens_out_with_less_than_occurence' occurences out of the dictionary (default: 1, leaves out none)
        leave_tokens_out_with_ratio_of_occurence -- Leaves out tokens which occure in more than 'leave_tokens_out_with_ratio_of_occurence' Quantlets (default: 1, leaves out none)
        include_keywords -- including keywords in the text base to create corpus and dictionary (default: True)
        include_description -- including description in the text base to create corpus and dictionary (default: False)
        include_whole_metainfo -- including entire metainfo in the text base to create corpus and dictionary (default: False)
        with_bigrams -- including bigrams (2 word phrases) as tokens (Default: False)
        with_trigrams -- including trigrams (3 word phrases) as tokens (Default: False, if True bigrams are as well included)
        """
        assert include_keywords or include_description or include_whole_metainfo, 'at least one text must be included, set include_keywords, include_description, include_whole_metainfo to True'
        if include_whole_metainfo:
            include_keywords = False
            include_description = False

        if with_trigrams:
            with_bigrams = True

        def text_preprocessing(text):
            regex_pun = re.compile(r"[\!\"\#\$\%\&\'(\)\*\+\.\/\:\<\=\>\?\@\[\\\]\^\`\{\|\}\~\-_\,\;]", re.UNICODE)
            regex_dig = re.compile(r'\d')
            parsed_text = regex_pun.sub(' ', text)
            parsed_text = regex_dig.sub(' ', parsed_text)
            Nostopwords = [word.lstrip().lower() for word in parsed_text.split() if word.lower() not in stopwords]
            tokens = [WordNetLemmatizer().lemmatize(t) for t in Nostopwords]
            return tokens

        stopwords = nltk.corpus.stopwords.words("english")
        nltk.download('wordnet')

        # creating cleaned (see text_preprocessing) texts
        docs_clean = {k: '' for k, v in self.quantlets.items() if v.is_debuggable}
        for k, v in self.quantlets.items():
            if v.is_debuggable:
                text = ''
                if include_keywords:
                    text += v.metainfo_debugged['keywords']
                if include_description:
                    text += v.metainfo_debugged['description']
                if include_whole_metainfo:
                    text = v.metainfo_undebugged
                docs_clean[k] = text_preprocessing(text)
        # excluding empty texts
        docs_clean = {k:v for k,v in docs_clean.items() if len(v) > 0}
        # {k:v for k,v in docs_clean.items() for i in v if len(i)>15}

        # Add bigrams and trigrams to docs_clean (only ones that appear 10 times or more).
        if with_bigrams:
            bigram = Phraser(Phrases([v for k,v in docs_clean.items()],min_count=leave_tokens_out_with_less_than_occurence,
                                     delimiter=b' ',threshold=leave_tokens_out_with_less_than_occurence))
            for k, v in docs_clean.items():
                for token in bigram[docs_clean[k]]:
                    if ' ' in token:
                        docs_clean[k].append(token)
        if with_trigrams:
            trigram = Phraser(Phrases(bigram[[v for k,v in docs_clean.items()]],min_count=leave_tokens_out_with_less_than_occurence,
                                      delimiter=b' ',threshold = leave_tokens_out_with_less_than_occurence))
            for k, v in docs_clean.items():
                for token in trigram[docs_clean[k]]:
                    if ' ' in token:
                        docs_clean[k].append(token)

        def txt_to_list(txt):
            if isinstance(txt,list):
                return txt
            else:
                return [txt]
        dictionary = Dictionary([txt_to_list(v) for k, v in docs_clean.items()])
        self.keywords_stats = dict()
        self.keywords_stats.update(total_number_of_unique_terms_in_documents = len(dictionary))

        # Filter out words that occur in less than leave_tokens_out_with_less_than_occurence  documents,
        # or more than leave_tokens_out_with_ratio_of_occurence of the documents.
        dictionary.filter_extremes(no_below=leave_tokens_out_with_less_than_occurence,
                                   no_above=leave_tokens_out_with_ratio_of_occurence)
        self.keywords_stats.update(total_number_of_unique_terms_in_documents_after_exclusion = len(dictionary))
        # get bag-of-words representation of each quantlet
        corpus = {k: dictionary.doc2bow(list(v)) for k, v in docs_clean.items()}
        #sys.stdout = open(os.devnull, 'w')
        dictionary[0]
        #sys.stdout = sys.__stdout__
        return corpus, dictionary
    def get_document_term_matrix(self, corpus, dictionary):
        """Constructs the document term matrix from corpus and dictionary.

        corpus -- text corpus
        dictionary -- dictionary of token
        """
        df = np.zeros([len(corpus), len(dictionary.id2token)])
        dtype = float
        if not [True for i in corpus.values() for j in i if j[1] % 1 != 0]:
            dtype = int
        df = pd.DataFrame(df, index=list(corpus.keys()), columns=list(dictionary.id2token.values()), dtype=dtype)

        for k, v in corpus.items():
            for i in v:
                df.loc[k][i[0]] += i[1]
        return df
    def get_SVD_explained_variance_ratio(self, tdm, with_normalize=False):
        """Returns the explained variance ratios of the singular values in the singular value decomposition of the document term matrix.

        tdm -- (numpy matrix) document term matrix
        with_normalize -- (bool) normalisation of matrix.
        :return:
        """
        svd = TruncatedSVD(np.linalg.matrix_rank(tdm))
        normalizer = Normalizer(copy=False)
        if with_normalize:
            lsa = make_pipeline(svd, normalizer)
        else:
            lsa = make_pipeline(svd)
        X = lsa.fit_transform(tdm)
        return svd.explained_variance_ratio_
    def get_corpus_tfidf(self,corpus,dictionary):
        """ Returns TFIDF transformed corpus.

        Arguments:
        corpus -- corpus as contructed by QUANTLET.get_corpus_dictionary
        dictionary -- dictionary as contructed by QUANTLET.get_corpus_dictionary
        """
        model = TfIdfTransformer(dictionary=dictionary)
        c_tfidf = model.fit_transform([v for k, v in corpus.items()])
        c_tfidf = dict(zip(corpus.keys(), c_tfidf))
        return c_tfidf
    def lsa_model(self, corpus, dictionary, num_topics=10):
        """Computes the Latent Semantic Analysis model and returns it.

        corpus -- coprus
        dictionary -- dictionary
        num_topics -- number of topics modelled with LSA
        """
        corpus_list = [v for k, v in corpus.items()]
        model = LsiModel(corpus=corpus_list, id2word=dictionary.id2token, num_topics=num_topics)
        return model
    def get_lsa_matrix(self, lsa, corpus, dictionary):
        """Returns the document topic matrix for a corpus and the via LSA extraced topics.

        lsa -- LSA model, see lsa_model
        corpus -- corpus
        dictionary -- dictionary
        """
        corpus_list = [v for k, v in corpus.items()]
        V = corpus2dense(lsa[corpus_list], len(lsa.projection.s)).T / lsa.projection.s
        return pd.DataFrame(V, index=list(corpus.keys()))
    def cl_kmeans(self, X, n_clusters):
        """Kmeans clustering.

        X -- document topic matrix
        n_clusters -- (int) number of clusters
        """
        kmeans = KMeans(n_clusters=n_clusters)
        y_kmeans = kmeans.fit_predict(X)
        labels = dict(zip(X.index, y_kmeans))
        return labels,kmeans
    def cl_spectral(self,X,n_clusters,dist_metric='euclidean'):
        """Spectral clustering.

        X -- document topic matrix
        n_clusters -- (int) number of clusters
        dist_metric -- (str) name of distance metric, default 'euclidean'.
        """
        #From scikit - learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].These metrics support sparse matrix inputs.
        #From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’,
        # ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        # See the documentation for scipy.spatial.distance for details on these metrics.These metrics do not support sparse matrix inputs.
        dist = pairwise.pairwise_distances(X,metric=dist_metric)
        cl = SpectralClustering(n_clusters=n_clusters,affinity='precomputed').fit(dist.max() - dist)
        return dict(zip(X.index, cl.labels_)),cl
    def cl_agglomerative(self,X, n_clusters, dist_metric='euclidean', linkage= 'ward'):
        """Hierarchical clustering

        X -- document topic matrix
        n_clusters -- number of clusters
        dist_metric -- distance matric, default: 'euclidean'
        linkage -- linkage method, default 'ward'
        """
        cl = AgglomerativeClustering(n_clusters=n_clusters, affinity=dist_metric, linkage =linkage).fit(X)
        return dict(zip(X.index, cl.labels_)), cl
    def cl_dbscan_n_cluster(self,X,n_cluster,dist_metric='euclidean',maxIter = 100, verbose = False,lower=0,upper=40):
        """Dbscan clustering

        X -- document topic matrix
        n_cluster -- number of cluster
        dist_metric -- distance metric, default: 'euclidean'
        maxIter -- maximal number of iterations to find epsilon constant such that n_clusters appear
        verbose -- printing results if True
        lower -- lower search value for epsilon
        upper -- upper search value for epsilon
        """
        dist = pairwise.pairwise_distances(X, metric=dist_metric)
        db = DBSCAN(min_samples=1, metric='precomputed')

        def half_find(lower,upper,count=0,maxIter=maxIter,verbose=verbose):
            neweps = (lower + upper) / 2
            db.set_params(eps=neweps)
            labels = db.fit_predict(dist)
            l = len(set(labels)) - (1 if -1 in labels else 0)
            if verbose:
                print('%i: n_cluster = %i'%(count,l))
            if count >= maxIter-1:
                print('max iteration reached with %i clusters, %i clusters searched'%(l,n_cluster))
                return(neweps)
            if l == n_cluster:
                return (neweps)
            elif l > n_cluster:
                return half_find(neweps,upper,count+1,maxIter)
            else:
                return half_find(lower,neweps,count+1,maxIter)
        eps = half_find(lower=lower,upper=upper,maxIter=100)
        db.set_params(eps=eps)
        labels = db.fit_predict(dist)
        return dict(zip(X.index, labels)), db
    def cl_dbscan(self,X,eps,dist_metric='euclidean'):
        """dbscan clustering for epsilon

        X -- document topic matrix
        eps -- distance value
        dist_metric -- distance metric
        """
        dist = pairwise.pairwise_distances(X, metric=dist_metric)
        cl = DBSCAN(eps=eps,min_samples=1, metric='precomputed').fit(dist)
        return dict(zip(X.index, cl.labels_)), cl
    def cl_dbscan_grid(self, X, eps_grid, dist_metric='euclidean',n_cluster=None):
        """Dbscan for grid

        X -- document topic matrix
        eps_grid -- epsilon grid
        dist_metric -- distance metric
        n_cluster -- number of clusters
        """
        dist = pairwise.pairwise_distances(X, metric=dist_metric)
        db = DBSCAN(min_samples=1, metric='precomputed')
        arr = pd.DataFrame(np.zeros([len(eps_grid),3]))
        arr.columns = ['eps','n_cluster','abs_dist_to_wanted']
        if n_cluster is None:
            arr = arr.iloc[:,:2]
        for id,i in enumerate(eps_grid):
            arr.iloc[id,0] = eps_grid[id]
            db.set_params(eps=eps_grid[id])
            labels = db.fit_predict(dist)
            arr.iloc[id,1] = len(set(labels)) - (1 if -1 in labels else 0)
            if n_cluster is not None:
                arr.iloc[id,2] = abs(arr.iloc[id,1] - n_cluster)
        return arr
    def topic_labels(self,cl,document_topic_matrix,lsa,top_n=5,take_care_of_bigrams=True):
        """ Returns named cluster values. Names are the top_n most significant terms in the cluster

        Arguments:
        cl -- dicitonary with names of documents as keys and cluster as values
        document_topic_matrix -- document topic matrix, see QUANTLET.get_lsa_matrix
        lsa -- lsa model
        top_n -- number of most signifikant words for cluster to be returned
        take_care_of_bigrams -- if False, it is possible that for example 'loss function','loss' and 'function' are included (default: True)
        """

        topic_loadings = lsa.show_topics(num_topics=-1, num_words=len(lsa.id2word), formatted=False)
        topic_term_matrix = pd.DataFrame([dict(i[1]) for i in topic_loadings])
        document_term_matrix = np.dot(document_topic_matrix, topic_term_matrix)

        topics = dict()
        for i in sorted(set(cl.values())):
            idx_docs = [v == i for k, v in cl.items()]
            idx = document_term_matrix[idx_docs, :].mean(axis=0).argsort()[::-1]
            tops = topic_term_matrix.columns[idx]
            tops = list(tops)
            if take_care_of_bigrams:
                for l in reversed(tops):
                    idx = tops.index(l)
                    if [j for j in tops[:idx] if l in j]:
                        del tops[idx]
            topics.update({i: ', '.join(tops[:top_n])})
        named_cl = {k: topics[v] for k, v in cl.items()}
        return named_cl
    def clustering(self, n_clusters,top_n_words=5,tfidf=True,cluster_algo='kmeans',dist_metric='euclidean',linkage=None,directory='data/'):

        assert cluster_algo in ['kmeans','spectral','hierarchical']
        def file_ending(n,cluster_algo,dist_metric,linkage):
            res = ['_lsa']
            res += [str(n) + 'cl']
            res += [cluster_algo]
            res += [dist_metric]
            if cluster_algo == 'hierarchical':
                res += [linkage]
            return '_'.join(res).replace('_kmeans_euclidean','')

        c, d = self.get_corpus_dictionary(leave_tokens_out_with_less_than_occurence=5,
                                          leave_tokens_out_with_ratio_of_occurence=1,
                                          with_bigrams = True)
        d[0] # need to be executed such that d.id2token is computed
        if isinstance(n_clusters,int):
            n_clusters = [n_clusters]
        if isinstance(n_clusters,list):
            if 0 in n_clusters:
                named_cl = {k:self.quantlets[k].repo_name for k in c.keys()}
                self.save_qlet_repo_file(named_cl,'',directory=directory)
            for n in n_clusters:
                if n == 0:
                    continue
                if tfidf:
                    c = self.get_corpus_tfidf(c, d)
                lsa = self.lsa_model(corpus=c, dictionary=d, num_topics=n)
                X = self.get_lsa_matrix(lsa, corpus=c, dictionary=d)
                if cluster_algo == 'kmeans':
                    cl,_ = self.cl_kmeans(X=X, n_clusters=n)
                elif cluster_algo == 'spectral':
                    cl,_ = self.cl_spectral(X=X,n_clusters=n,dist_metric=dist_metric)
                elif cluster_algo == 'hierarchical':
                    self.cl_agglomerative(X=X,n_clusters=n_clusters,dist_metric=dist_metric,linkage=linkage)
                named_cl = self.topic_labels(cl,X,lsa,top_n_words)
                self.save_qlet_repo_file(named_cl,file_ending(n,cluster_algo,dist_metric,linkage),directory=directory)
    def create_datanames_file(self,directory=''):
        datanames = dict(datanames=[], names_select=[], description='hallo')
        files = [i for i in os.listdir('data') if 'qlets_github_ia.json' in i]
        files.extend([i for i in os.listdir('data') if 'cl.json' in i and 'qlets_github_ia' in i])
        files.extend([i for i in os.listdir('data') if
                      '.json' in i and 'cl.json' not in i and 'qlets_github_ia' in i and i != 'qlets_github_ia.json'])
        for f in files:
            if f == 'qlets_github_ia.json':
                datanames['names_select'].append('GitHub: clustered by repositories')
                datanames['datanames'].append(f.replace('.json', ''))
                continue
            f2 = f.lstrip('qlets_github_ia_').replace('.json', '')
            f2 = f2.split('_')
            linkage = ''
            if len(f2) == 1:
                algo = 'K-Means'
                dist_metric = ''
            else:
                algo = f2[1].title()
                dist_metric = ' ' + f2[2].title() + ' Dist., '
            if len(f2) > 3:
                linkage = f2[3].title()+', '
            datanames['names_select'].append(
                'GitHub: LSA, %s,%s%s %s clusters' % (algo, dist_metric, linkage, f2[0].replace('cl', '')))
            datanames['datanames'].append(f.replace('.json', ''))
        with open(directory+'/datanames.json', 'w')as f:
            json.dump(datanames, f)
    def save_qlet_repo_file(self, cluster_label,file_name_ending, directory=''):
        """Saves file to be read in by Quantlet.de

        cluster_label -- clustering labels for each Quantlet
        file_name_ending -- file name ending
        directory -- directory where to save file
        """
        correct = dict()  # TODO: Safe as JSON and read from file!!!
        correct.update({'SFE': ['Statistics of Financial Markets I', 'Statistics of Financial Markets']})
        correct.update({'MVA': ['Applied Multivariate Statistical Analysis']})
        correct.update({'XFG': ['Applied Quantitative Finance', 'Applied Quantitative Finance (3rd Edition)',
                                'XFG (3rd Edition)']})
        correct.update({'BCS': ['Basic Elements of Computational Statistics']})
        correct.update({'STF': ['Statistical Tools for Finance and Insurance']})
        correct.update({'SFS': ['Statistics of Financial Markets : Exercises and Solutions']})
        correct.update({'SPM': ['Nonparametric and Semiparametric Models']})
        correct.update({'ISP': ['An Introduction to Statistics with Python']})
        correct.update({'ARR': ['ARR - Academic Rankings Research']})
        correct.update({'SMS': ['Multivariate Statistics: Exercises and Solutions', 'SMS2']})
        correct.update({'MSE': ['Modern Mathematical Statistics : Exercises and Solutions']})
        correct.update({'SRM': ['SRM - Stochastische Risikomodellierung und statistische Methoden']})
        correct.update(
            {'TvHAC': ['Time-varying Hierarchical Archimedean Copulas Using Adaptively Simulated Critical Values']})
        correct.update({'SPA': ['SPA - Stochastic Population Analysis']})
        correct.update({'RJMCMC': ['Univariate Time Series']})
        correct_bog = []
        for k,v in correct.items():
            correct_bog.append(k)
            correct_bog.extend(v)


        def correct_book(pub):
            for k, v in correct.items():
                if pub in v:
                    return k
            return pub
        def __create_template(v,cluster,id):
            tmp = dict()
            tmp.update({'name': v.metainfo_debugged['name of quantlet']})

            tmp.update({'artist': cluster})

            txt = [v.metainfo_debugged['description']]
            if False:
                {k: str(v.metainfo_debugged['submitted']) for k, v in self.quantlets.items() if v.is_debuggable if
                 'submitted' in v.metainfo_debugged.keys() if v.metainfo_debugged['submitted'] is not None if
                 not isinstance(v.metainfo_debugged['submitted'], (str, list))}
                for i in ['author','submitted']: # TODO combine all
                    if not i in v.metainfo_debugged.keys(): continue
                    if v.metainfo_debugged[i] is None: continue
                    if isinstance(v.metainfo_debugged[i],list):
                        txt.extend(v.metainfo_debugged[i])
                    elif not isinstance(v.metainfo_debugged[i],datetime.date):
                        txt.append(v.metainfo_debugged[i])
                txt.append(v.directory)

            txt = [str(i) for i in list(v.metainfo_debugged.values()) if i is not None]
            tmp.update({'description': ' '.join(txt)})
            sw = v.software
            if sw is None:
                tmp.update({'software': ''})
            else:
                sw = sw.replace('ipynb', 'py').split(',')
                sw = [s for s in sw if s in ['py', 'm', 'r', 'sas', 'sh', 'cpp']]
                tmp.update({'software': ','.join(sw)})


            book = []
            if v.metainfo_debugged['published in'] is None:
                book.append(v.repo_name)
            elif v.metainfo_debugged['published in'] in correct_bog:
                book.append(correct_book(v.metainfo_debugged['published in']))
            else:
                if len(v.metainfo_debugged['published in']) > 0:
                    book.append(correct_book(v.metainfo_debugged['published in']))
                book.append(v.repo_name)
            tmp.update({'book': ' - '.join(book)})
            tmp.update({'id': id})
            tmp.update({'playcount': sys.getsizeof(v.metainfo_undebugged)})
            url = ['https://github.com/QuantLet']
            url.append(v.repo_name)
            url.append('tree/master')
            url.append(v.directory.lstrip(v.repo_name).lstrip('/'))
            url = '/'.join(url)
            tmp.update({'full_link': url})
            return tmp
        nodes = []
        id = 0
        for k, v in self.quantlets.items():
            if not v.is_debuggable:
                continue
            try:
                nodes.append(__create_template(v,cluster_label.get(k,' '),id))
            except:
                print(k)
                print(v.metainfo_undebugged)
                raise
            id += 1
        with open(directory+'qlets_github_ia'+file_name_ending+'.json', 'w') as outfile:
            json.dump(dict({'nodes': nodes, 'links': []}), outfile)
    def tsne(self,X,cluster_labels,n_iter=5000,dist_metric='euclidean',DPI=150, save_directory='',save_ending='',file_type='pdf'):
        """t-Stochastic Neighbour embedding is plotted.

        X -- document topic matrix
        cluster_labels -- cluster labels for each Quantlet
        n_iter -- number of iterations in t-SNE, default 5000
        dist_metric -- distance metric, default 'euclidean'
        DPI -- Dots per inch in graphic, default 150
        save_directory -- directory where images are saved
        save_ending -- filename ending for images
        """
        n = len(set(cluster_labels.values()))
        tsne = TSNE(n_components=2, random_state=1, n_iter=n_iter, metric=dist_metric)
        pos = tsne.fit_transform(X)

        cl_set = list(set(cluster_labels.values()))

        ss = 2
        plt.figure(figsize=(8 * ss, 6 * ss), dpi=DPI)
        plt.scatter(pos[:, 0], pos[:, 1], c=[cl_set.index(v) for k,v in cluster_labels.items()], cmap=plt.cm.get_cmap('jet', n), s=18)
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('off')
        fn = ''
        if len(save_directory)>0:
            fn += save_directory + '/'
        fn += 'cluster%02d_%s.%s' % (n, save_ending,file_type)
        plt.savefig(fn, bbox_inches='tight', transparent=True)

        plt.figure(figsize=(8 * ss, 6 * ss), dpi=DPI)
        cmap = plt.cm.get_cmap('jet', n)
        for id,i in enumerate(set(cluster_labels.values())):
            idx = [v==i for v in cluster_labels.values()]
            plt.scatter(pos[idx, 0], pos[idx, 1], c=cmap(id), s=18, label=i)
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('off')
        plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=(12 if not n == 50 else 8), bbox_to_anchor=(1, 1))
        fn = ''
        if len(save_directory)>0:
            fn += save_directory + '/'
        fn += 'cluster%02d_%s_with_legend.%s' % (n, save_ending,file_type)
        plt.savefig(fn, bbox_inches='tight', transparent=True)