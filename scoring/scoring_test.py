from argparse import ArgumentParser
from datetime import datetime
from logging import Logger, getLogger, StreamHandler, FileHandler, Formatter, DEBUG, WARN, INFO
import collections
import sys

from score_sentence import ScoreSentence, FILTERS

def set_logger(name, rootname="mylog/main.log"):
    dt_now = datetime.now()
    dt = dt_now.strftime('%Y%m%d_%H%M%S')
    fname = rootname + "." + dt
    logger = getLogger(name)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler2 = FileHandler(filename=fname)
    handler2.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler1.setLevel(INFO)
    handler2.setLevel(INFO)  #handler2はLevel.WARN以上
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(INFO)
    return logger

def add_local_args(parser):
    parser.add_argument('--filter-type', default='worst', type=str, help='application KenLM filter')
    parser.add_argument('--filter-threshold', default=-4.8, type=float, help='threshold of filter')
    parser.add_argument('--used-ngram-model', default='scoring/models/bccwj-csj-np.bin', type=str, help='n-gram model for KenLM scoring')
    parser.add_argument('--display-ngram-score', action='store_true', default=False, help='display n-gram score by KenLM')
    parser.add_argument('--ngram-reranking', action='store_true', default=False, help='re-ranking by n-gram score')
    parser.add_argument('--remove-contain-oov', action='store_true', default=False, help='remove sentence which hava oov word')
    parser.add_argument('--display-modified-ngram', action='store_true', default=False, help='display moified ngram analisys')
    
    return parser

parser = ArgumentParser('This program is tester for score_sentence.py.')
parser = add_local_args(parser)
args = parser.parse_args() # analize arguments
logger = set_logger("scoring", "mylog/scoring.log")

scorer = ScoreSentence(args, logger)

while True:
    text = input('>>')
    
    text_counter = collections.Counter()
    text_counter[text] = 0.0
    
    scorer(text_counter)