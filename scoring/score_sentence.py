# coding: utf-8
# Corded by Mikawa

from argparse import ArgumentParser
from logging import Logger
import kenlm
import MeCab
import numpy as np
import re

import scrapy

FILTERS = ['none',
           'total',
           'worst',
           'mean',
           'harmonic']

class ScoreSentence(object):
    def __init__(self, args:ArgumentParser, logger:Logger=None):
        self.args = args
        self.logger = logger
        
        self.filter_type = self.args.filter_type
        self.filter_threshold = self.args.filter_threshold
        
        model_path = self.args.used_ngram_model
        self.ngram = kenlm.LanguageModel(model_path)
        
        opt = '-d ./scripts/scoring/unidic-csj-3.0.1.1 -O "" -F "%m+%f[0]/%f[1] " -E ""'
        self.tagger = MeCab.Tagger(opt)
        
    def __call__(self, scores):
        _scores = scores.most_common(1)[0]
        scores = [score[0] for score in scores.most_common()]
        
        assert self.filter_type in FILTERS, \
            '"{}" is invalid filter type.'.format(self.filter_type)
        
        if self.filter_type == 'none':
            return _scores[0], _scores[1], None
        elif self.filter_type == 'total':
            scorer = self.total
        elif self.filter_type == 'worst':
            scorer = self.worst
        elif self.filter_type == 'mean':
            scorer = self.mean
        elif self.filter_type == 'harmonic':
            scorer = self.harmonic_mean
            
        if self.logger is not None:
            self.logger.info('Scoring by {0} ({1}-gram model)'.format(self.filter_type, self.ngram.order))
            
        results = []
        for sentence in scores:
            parsed_sentence = self.preprocess(sentence)
            
            words = ['<s>'] + parsed_sentence.split() + ['</s>']
            ngram_scores = self.ngram.full_scores(parsed_sentence)
            
            # Show scores and n-gram matches
            if self.logger is not None and self.args.display_ngram_score:
                self.logger.info('================================================')
                _ngram_scores = self.ngram.full_scores(parsed_sentence)
                for i, (prob, length, oov) in enumerate(_ngram_scores):
                    self.logger.info('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
                    if oov:
                        self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                self.logger.info('================================================')
                        
            results.append((sentence, scorer(ngram_scores)))
            
        _results = results
        removed_cands = []
        # 閾値以下を除去
        for i, result in enumerate(results):
            if result[1] < self.filter_threshold:
                removed_cands.append(results.pop(i))
        
        # 取り敢えず閾値以上の候補が見つからないときは最良のものを選択 -> もう少しいい手を考えたい
        if len(results)==0:
            results = _results
        
        if self.logger is not None:
            self.logger.info(results)
        
        ret_utt, ret_score = '', -1000000.0
        if self.args.ngram_reranking:
            for score in results:
                if score[1] > ret_score:
                    ret_utt, ret_score = score
        else:
            ret_utt, ret_score = results[0]
        
        return ret_utt, ret_score, (results + removed_cands)
    
    def preprocess(self, sentence):
        parsed_sentence = self.tagger.parse(sentence)
        parsed_sentence = re.sub('/ ', ' ', parsed_sentence)
        parsed_sentence = re.sub('、\+補助記号/読点 ', '<sp> ', parsed_sentence)

        # bccwj-csj-np.bin 向けの処理
        parsed_sentence = re.sub('。\+補助記号/句点 ', '<sp> ', parsed_sentence)
        parsed_sentence = parsed_sentence[:-5] if parsed_sentence[-5:] == '<sp> ' else parsed_sentence
        
        return parsed_sentence
    
    def total(self, ngram_scores):
        """N-gram による文全体の対数確率を返す
        """
        prob_sum = 0.0
        for prob, length, oov in ngram_scores:
            prob_sum += prob
            
        return prob_sum

    def worst(self, ngram_scores):
        """N-gram による対数確率の内，最小のものを返す．
        """
        prob_min = 0.0
        for prob, length, oov in ngram_scores:
            if prob_min > prob:
                prob_min = prob
                
        return prob_min

    def mean(self, ngram_scores):
        """N-gram による対数確率の相加平均を返す．
        """
        prob_sum = 0.0
        length = 0
        for prob, length, oov in ngram_scores:
            prob_sum += prob
            length += 1
        
        return prob_sum/length

    def harmonic_mean(self, ngram_scores):
        """N-gram による確率の調和平均を返す．
        """
        hprob_sum = 0.0
        length = 0
        for prob, length, oov in ngram_scores:
            hprob_sum += 1/prob
            length += 1
        
        return length/hprob_sum
