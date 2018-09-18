# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:50:32 2018

@author: sterlinm.hub
"""


import itertools, yaml, datetime, re
import numpy as np
from nltk.corpus import stopwords
class METAFILE:
    """ This class contains the matainfo files informations. """
    keyReplacement = {
        'author': [],
        'input': ['inputs'],
        'datafiles': ['datafile'],
        'name of quantlet': ['iname of quantlet'],
        'subfunctions' : ['subfunction'],
        'submited' : ['submitte','submitted by']
    }
    keyReplacementValues = list(itertools.chain.from_iterable(keyReplacement.values()))
    def __init__(self, file, repo, content, commits):
        """Initializing function for the class METAFILE.
        file -- github.ContentFile.ContentFile of the metainfo file
        repo -- the repository in which the metainfo file is
        """
        self.metainfo_undebugged = file.decoded_content.decode()
        self.repo_name = repo.name
        self.path_in_repo = file.path
        self.path_of_metainfo = '/'.join([self.repo_name,file.path])
        self.directory = '/'.join([self.repo_name,file.path.replace(file.name,'')])
        if self.directory.endswith('/'):
            self.directory = self.directory[:-1]
        self.is_debuggable = False
        self.last_modified = file.last_modified
        self.commit_first = {k:v for k,v in commits[1].raw_data.items() if k in ['sha','commit']}
        self.commit_last = {k: v for k, v in commits[0].raw_data.items() if k in ['sha', 'commit']}
        self.sha = file.sha
        self.size = file.size
        try:
            try:
                self.metainfo_debugged = METAFILE.yaml_debugger(self.metainfo_undebugged)
            except yaml.scanner.ScannerError as e:
                self.metainfo_debugged = METAFILE.yaml_debugger(METAFILE.pre_clean(self.metainfo_undebugged))
            self.is_debuggable = True
            
            self.metainfo_debugged = {k.lower(): v for k, v in self.metainfo_debugged.items()}
            self.metainfo_debugged = METAFILE.clean_keys(self.metainfo_debugged)
            self.create_keyword_list()
            self.list_to_string()
            file_type = [c.name.split('.')[-1].lower() for c in content if c.name.split('.')[0].lower() == self.metainfo_debugged['name of quantlet'].lower()]
            file_type = [i for i in file_type if i in ['m','r','py','c','sh','ipynb','sas']]
            if file_type:    
                self.software = ','.join(file_type)
            else:
                self.software = None
            self.__grading(content)
        except:
            pass
#                [i for i in file_type if i not in ['png','jpg','jpeg','pdf']]
        
    def pre_clean(c):
        """Correcting non YAML debuggable files which contain ':' at the wrong location. """
        csplit = c.split(':')
        _tmp = ['\n\n' in b or '\n' in b for b in csplit]
        _tmp[0] = True
        _tmp[-1] = True
        if all(_tmp):
            return c
        idx = np.argwhere([not b for b in _tmp]).flatten()
        idx[::-1].sort()
        for i in idx:
            csplit[i] += ' -'+csplit[i+1]
            csplit = np.delete(csplit,i+1)
        return ':'.join(csplit)
    def yaml_debugger(x):
        """ YAML debugging the string, correcting for tabs """
        x = x.replace('\t', ' ')
        x = x.replace('\r', ' ')
        res = yaml.load(x, Loader = yaml.SafeLoader)
        return res
    def clean_keys(d):
        """ Renames keys of the dictary that were used falsely"""
        # import copy; d = copy.deepcopy(tmp)
        _contains_brackets = ['[' in i for i in list(d.keys())]
        if any(_contains_brackets):
            ret = {}
            for k,v in d.items():
                if '[' not in k:
                    ret.update({k:v})
            for k,v in d.items():# [{k:v} for k,v in d.items() if '[' in k]:
                if '[' in k:
                    _key = (k.split('[')[0]).rstrip()
                    if _key in ret.keys():
                        if isinstance(v,list):
                            if isinstance(ret[_key],str):
                                v = v.split(', ')
                            ret[_key].extend(v)
                        elif isinstance(v,str) and isinstance(ret[_key],list):
                            ret[_key].extend(v.split(', '))
                        elif isinstance(v,str) and isinstance(ret[_key],str):
                            ret[_key] += ', '+ v
                    else:
                        ret.update({_key: v})
            d = ret
           
        # TODO check with string distance which field is meant and combine
        if True:
            _d = np.concatenate(list(METAFILE.keyReplacement.values()))
            _b = np.isin(_d,np.array(list(d.keys())))
            tmp = {}
            if any(_b):
                for k,v in METAFILE.keyReplacement.items():
                    for v2 in [v2 for v2 in v if v2 in _d[_b]]:
                        tmp.update({k: v2})
            for k,v in tmp.items():
                d[k] = d.pop(v)
        return d
    def list_to_string(self):
        """ parsing lists to string, since yaml can contain list """
        if not self.is_debuggable:
            return
        for k,v in self.metainfo_debugged.items():
            if isinstance(v,list) and all([isinstance(i,str) for i in v]):
                sep = ', '
                if k == 'description':
                    sep = '.\n'
                elif k == 'author':
                    if [i for i in self.metainfo_debugged[k] if ',' in i]:
                        self.metainfo_debugged[k] = [' '.join(i.split(',',1)[::-1]) for i in self.metainfo_debugged[k]]
                self.metainfo_debugged[k] = sep.join(v)
    def create_keyword_list(self):
        """ Saves the keywords as a list """
        if isinstance(self.metainfo_debugged['keywords'],list):
            self.keyword_list = self.metainfo_debugged['keywords']
        else:
            self.keyword_list = self.metainfo_debugged['keywords'].split(',')
            self.keyword_list = [i.lstrip(' ') for i in self.keyword_list]
        self.keyword_list.sort()
    
    def __grading(self,content):
        self.grading_output = {
                'q_quali': ['A'],  # quality of metainfo file
                'keywords':None, # number of keywords
                'description_length':None, # number of words in discription
                'description_length_wo_stopwords':None, # number of words without stopwords in discription
                'comment':[], # indication why grade worse than A was given
                'pictures':None, # number of pictures in 
                'submitted_year':None # submission year
                }
        
        if self.is_debuggable:
            _fields = ['author', 'description','keywords','name of quantlet', 'published in']
            _missing_fields = [c for c in _fields if c not in self.metainfo_debugged.keys()]
            if len(_missing_fields)>0:
                _msg = ', '.join(_missing_fields)
                if len(_missing_fields)>1:
                    _msg  += ' is missing'
                else:
                    _msg  += ' are missing'
                self.grading_output['comment'].append(_msg)
                self.grading_output['q_quali'].append('D')
            try:
                self.grading_output['q_name'] = self.metainfo_debugged['name of quantlet']
                self.grading_output['keywords'] = len(self.keyword_list)
                self.grading_output['description_length'] = len(self.metainfo_debugged['description'].split())
                stop_words = set(stopwords.words('english'))
                self.grading_output['description_length_wo_stopwords'] = len([word for word in self.metainfo_debugged['description'].split() if word not in stop_words])
                
                
                if 'submitted' in self.metainfo_debugged.keys() and self.metainfo_debugged['submitted'] is not None:
                    try:
                        submission_year = re.findall(r"(\d{4})", self.metainfo_debugged['submitted'])
                        if submission_year:
                            self.grading_output['submitted_year']  = min(submission_year)
                        self.grading_output['submitted_year']
                    except: 
                        pass
                
                self.grading_output['submitted_year'] = datetime.datetime.strptime(self.last_modified,'%a, %d %b %Y %H:%M:%S GMT').strftime('%Y')
                
            except:
                self.grading_output['q_quali'].append('F')
                '! '.join(self.grading_output['comment'])
                return None
            if self.grading_output['keywords'] < 5:
                if self.grading_output['keywords'] > 0:
                    self.grading_output['q_quali'].append('B')
                    self.grading_output['comment'].append('less than 5 keywords')
                else:
                    self.grading_output['q_quali'].append('C')
                    self.grading_output['comment'].append('no keywords found')
            if self.grading_output['description_length'] < 10:
                if self.grading_output['description_length'] >0:
                    self.grading_output['comment'].append('less than 10 words in description')
                    self.grading_output['q_quali'].append('B')
                else:
                    self.grading_output['comment'].append('no description')
                    self.grading_output['q_quali'].append('C')
            content2 = [c.name.split('.') for c in content]
            if not any(['.'.join(c[:len(c)-1]).lower() == self.grading_output['q_name'].lower() for c in content2 if c[-1].lower() not in ['png','jpg','jpeg','pdf','md']]):
                self.grading_output['comment'].append('Q is not in folder or named differently')
                self.grading_output['q_quali'].append('D')
        else:
            self.grading_output['q_quali'] = 'F'
            self.grading_output['comment'] = 'YAML debug error'
            return None
        self.grading_output['pictures'] = sum([c.name.split('.')[-1].lower() in ['png','jpg','jpeg'] for c in content])
        pdfs = [c for c in content if 'pdf' == c.name.split('.')[-1].lower()]
        if len(pdfs)>0 and self.grading_output['pictures'] == 0:
            #_refs = [c.name.split('.')[0] for c in _contents if c.name.split('.')[1] in ['png','jpg','jpeg']]
            #_pdfs = [c for c in _pdfs if c.name.split('.')[0] in _refs]
            self.grading_output['comment'].append('only PDF picture in folder (?)')
            self.grading_output['q_quali'].append('B')
        self.grading_output['comment'] = '! '.join(self.grading_output['comment'])
        self.grading_output['q_quali'] = max(self.grading_output['q_quali'])
        self.grade = self.grading_output['q_quali']
