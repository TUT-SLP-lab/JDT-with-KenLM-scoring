# coding: utf-8
# Corded by Mikawa

from argparse import ArgumentParser
from logging import Logger
import math
import kenlm
import MeCab
import re

FILTERS = ['none',
           'total',
           'worst',
           'modified-worst',
           'geometric',
           'harmonic',
           'modified-harmonic',
           'depth-harmonic']

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
        sentences = [score[0] for score in scores.most_common()]
        
        assert self.filter_type in FILTERS, \
            '"{}" is invalid filter type.'.format(self.filter_type)
        
        if self.filter_type == 'none':
            return _scores[0], _scores[1], None
        elif self.filter_type == 'total':
            scorer = self.total
        elif self.filter_type == 'worst':
            scorer = self.worst
        elif self.filter_type == 'modified-worst':
            scorer = self.modified_worst
        elif self.filter_type == 'geometric':
            scorer = self.geometric_mean
        elif self.filter_type == 'harmonic':
            scorer = self.harmonic_mean
        elif self.filter_type == 'modified-harmonic':
            scorer = self.modified_harmonic
        elif self.filter_type == 'depth-harmonic':
            scorer = self.depth_harmonic
            
        if self.logger is not None:
            self.logger.info('Scoring by {0} ({1}-gram model)'.format(self.filter_type, self.ngram.order))
            
        results = []
        results = scorer(sentences)
        
        if self.logger is not None and self.args.display_ngram_score:
            for sentence, score in zip(scores, results):
                parsed_sentence = self.preprocess(sentence)
                
                words = ['<s>'] + parsed_sentence.split() + ['</s>']
                ngram_scores = self.ngram.full_scores(parsed_sentence)
                
                # Show scores and n-gram matches
                self.logger.info('================================================')
                for i, (prob, length, oov) in enumerate(ngram_scores):
                    self.logger.info('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
                    if oov:
                        if '数詞' not in words[i+1]:
                            self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                self.logger.info('sentence score: {0}'.format(score[1]))
                self.logger.info('================================================')
            
        removed_cands = []
        _results = []
        # 閾値以下を除去
        for result in results:
            if result[1] < self.filter_threshold:
                removed_cands.append(result)
            else:
                _results.append(result)
        results = _results
        
        _results = []
        # oov を含んだ候補文を除去
        if self.args.remove_contain_oov:
            for result in results:
                if result[2]:
                    removed_cands.append(result)
                else:
                    _results.append(result)
        results = _results
        
        # 取り敢えず閾値以上の候補が見つからないときは元のスコアの内最良のものを選択 -> もう少しいい手を考えたい
        if len(results)==0:
            if self.logger is not None:
                self.logger.info('all cands are filtered. threshold={}'.format(self.args.filter_threshold))
            results = [removed_cands.pop(0)]
        
        if self.logger is not None:
            self.logger.info(results)
        
        if self.args.ngram_reranking:
            ret_utt, ret_score = '', -1000000.0
            for score in results:
                if score[1] > ret_score:
                    ret_utt, ret_score = score
        else:
            ret_utt, ret_score, have_oov = results[0]
        
        return ret_utt, ret_score, (results + removed_cands)
    
    def preprocess(self, sentence):
        parsed_sentence = self.tagger.parse(sentence)
        parsed_sentence = re.sub('/ ', ' ', parsed_sentence)
        parsed_sentence = re.sub('、\+補助記号/読点 ', '<sp> ', parsed_sentence)
        parsed_sentence = re.sub('，\+補助記号/読点 ', '<sp> ', parsed_sentence)
        parsed_sentence = re.sub( ',\+補助記号/読点 ', '<sp> ', parsed_sentence)

        # bccwj-csj-np.bin 向けの処理 (一応，種類を分ける)
        parsed_sentence = re.sub('。\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        parsed_sentence = re.sub( '.\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        parsed_sentence = re.sub('．\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        parsed_sentence = re.sub('\!\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        parsed_sentence = re.sub('\?\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        parsed_sentence = re.sub('！\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        parsed_sentence = re.sub('？\+補助記号/句点 ', '<sp>  ', parsed_sentence)
        
        parsed_sentence = re.sub('ー\+助詞/終助詞 ', '\+助詞/終助詞', parsed_sentence)
        
        # 使用しているbccwj-csj-npは文末の句点がないため、除去する。
        while parsed_sentence[-5:] == '<sp> ' or parsed_sentence[-6:] == '<sp>  ':
            if parsed_sentence[-5:] == '<sp> ':
                parsed_sentence = parsed_sentence[:-5]
            elif parsed_sentence[-6:] == '<sp>  ':
                parsed_sentence = parsed_sentence[:-6]
        
        return parsed_sentence
    
    def total(self, sentences):
        """N-gram による文全体の対数確率を返す。
        """
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            prob_sum = 0.0
            exist_oov = False
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                    
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    prob_sum += prob
                        
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(prob_sum))
                self.logger.info('================================================')
            results.append((sentence, prob_sum, exist_oov))
            
        return results

    def worst(self, sentences):
        """N-gram による対数確率の内，最小のものを返す．局所的な破綻検出．
        """
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            prob_min = 0.0
            exist_oov = False
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                    
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    if prob_min > prob:
                        prob_min = prob
                        
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(prob_min))
                self.logger.info('================================================')
            results.append((sentence, prob_min, exist_oov))
                
        return results
    
    def modified_worst(self, sentences):
        """文頭単語のスコア悪化による影響を軽減した修正局所的破綻検出。
        また、主観的に問題がない場合でも、n-gramのlengthによってスコアが悪化することを修正する。
        """
        """N-gram による対数確率の内，最小のものを返す．局所的な破綻検出．
        """
        model_order = self.ngram.order
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            prob_min = 0.0
            exist_oov = False
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    
                    # modified
                    prob = (n_length / model_order)**2 * prob
                    ##########
                    
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                        
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    
                    if prob_min > prob:
                        prob_min = prob
                        
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(prob_min))
                self.logger.info('================================================')
                
            results.append((sentence, prob_min, exist_oov))
                
        return results

    def geometric_mean(self, sentences):
        """N-gram による確率の幾何平均を返す。
        """
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            prob_sum = 0.0
            exist_oov = False
            length = 0
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                    
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    prob_sum += prob
                    length += 1
                        
            score = prob_sum/length
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(score))
                self.logger.info('================================================')
            results.append((sentence, score, exist_oov))
                
        return results

    def harmonic_mean(self, sentences):
        """N-gram による確率の調和平均を返す．
        """
        model_order = self.ngram.order
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            hprob_sum = 0.0
            length = 0
            exist_oov = False
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                        
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    
                    prob = math.pow(2, prob) # KenLMはlog10だが，それだと値が非常に小さくなるため、2としている
                    hprob_sum += 1/prob
                    length += 1
                        
            score = math.log2(length/hprob_sum)
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(score))
                self.logger.info('================================================')
                    
            results.append((sentence, score, exist_oov))
                
        return results
    
    def modified_harmonic(self, sentences):
        """文頭単語のスコア悪化による影響を軽減した修正調和平均。
        また、主観的に問題がない場合でも、n-gramのlengthによってスコアが悪化することを修正する。
        """
        model_order = self.ngram.order
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            hprob_sum = 0.0
            length = 0
            exist_oov = False
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
                
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    
                    # modified
                    prob = (n_length / model_order)**2 * prob
                    ##########
                    
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                        
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    
                    prob = math.pow(2, prob) # KenLMはlog10だが，それだと値が非常に小さくなるため、2としている
                    hprob_sum += 1/prob
                    length += 1
                        
            score = math.log2(length/hprob_sum)
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(score))
                self.logger.info('================================================')
            
            results.append((sentence, score, exist_oov))
                
        return results
    
    def depth_harmonic(self, sentences):
        """修正調和平均をさらに入力時系列数（次元）に対して修正し、性質をworstに近づけた。
        ただし、worstとは違い、各入力が似ていても全く同じでなければユニークな値を返すことが可能。
        """
        model_order = self.ngram.order
        results = []
        for sentence in sentences:
            parsed_sentence = self.preprocess(sentence)
            # words = parsed_sentence.split()
            
            parsed_sentence = re.sub('<sp> ', '<sp>  ', parsed_sentence)
            parsed_sentences = parsed_sentence.split('<sp>  ')
            ngram_scores_list = [self.ngram.full_scores(prsd_sent) for prsd_sent in parsed_sentences]
            words_list = [prsd_sent.split() for prsd_sent in parsed_sentences]
            
            depth = sum([len(words) for words in words_list])
            power = math.sqrt(depth)
            
            hprob_sum = 0.0
            length = 0
            exist_oov = False
            
            if self.args.display_modified_ngram:
                self.logger.info('================================================')
                
            for ngram_scores, words in zip(ngram_scores_list, words_list):
                words = ['<s>'] + words + ['</s>']
                
                for i, (prob, n_length, oov) in enumerate(ngram_scores):
                    
                    # modified
                    prob = (n_length / model_order)**2 * prob
                    ##########
                    
                    if self.args.display_modified_ngram:
                        self.logger.info('{0} {1}: {2}'.format(prob, n_length, ' '.join(words[i+2-n_length:i+2])))
                        if oov:
                            if '数詞' not in words[i+1]:
                                self.logger.info('\t"{0}" is an OOV'.format(words[i+1]))
                        
                    exist_oov = ((oov and '数詞' not in words[i]) or exist_oov)
                    
                    prob = math.pow(10, prob * power) # KenLMはlog10だが，それだと値が非常に小さくなるため、2としている
                    hprob_sum += 1/prob
                    length += 1
                        
            score = math.log10(math.pow(length/hprob_sum, 1/power))
            if self.args.display_modified_ngram:
                self.logger.info('sentence score: {0}'.format(score))
                self.logger.info('================================================')
            
            results.append((sentence, score, exist_oov))
                
        return results
