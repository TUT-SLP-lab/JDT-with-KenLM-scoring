# JDT-with-KenLM-scoring
Japanese-Dialog-Transformerの応答候補に対して、KenLMによるN-gram言語モデルでスコアリングし、フィルタリング若しくはリランキングを行う。スコアリングの性能だけ見たい場合はREADME.md最下部へ移動してください。

https://github.com/nttcslab/japanese-dialog-transformers

model(checkpoint)は上記のURLの通りにお願いします。

## KenLMのインストール
適当なディレクトリに移動後以下のコマンドを実行する。
```
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```

Anaconda環境で使用する際には、仮想環境を立ち上げた上でkenlm下にあるsetup.pyを以下のコマンドにて実行する必要がある。
```
python setup.py install
```

## Usage
ntt の Japapanese-Dialog-Transformer/scripts に内容物をコピー。
scripts内で以下のLinuxコマンドを実行する。

### MeCab + unidic-csj-3.0.1.1のダウンロード
```
apt-get install -y mecab libmecab-dev
pip install mecab-python3

cd scoring
wget https://clrd.ninjal.ac.jp/unidic_archive/csj/3.0.1.1/unidic-csj-3.0.1.1.zip
unzip unidic-csj-3.0.1.1.zip
```

### bccwj-csj-np.3g.kn.gzのダウンロード
```
mkdir models & cd models
pip install gdown
gdown --id 1P642KJRMUdXX3t2g3MQZnZ9BGovTLKiE
gunzip bccwj-csj-np.3g.kn.gz
mv bccwj-csj-np.3g.kn bccwj-csj-np.bin
```

## Run
Japanese-Dialog-Transformer ディレクトリ下に移動し以下を実行する。
```
python scripts/dialog.py data/sample/bin/  --path checkpoints/japanese-dialog-transformer-1.6B.pt  --beam 80  --min-len 10  --source-lang src  --target-lang dst  --tokenizer space  --bpe sentencepiece  --sentencepiece-model data/dicts/sp_oall_32k.model  --no-repeat-ngram-size 3  --nbest 80  --sampling  --sampling-topp 0.9  --temperature 1.0  --show-nbest 5  --filter-type depth-harmonic  --filter-threshold -4.8  --used-ngram-model scripts/scoring/models/bccwj-csj-np.bin  --display-ngram-score  --display-modified-ngram  --starting-phrase 松丸さんはお休みは何をしてますか？
```
もし、N-gram言語モデルによるリランキングを行う場合はオプションに以下を追加する。
```
--ngram-reranking
```
新たに三河が追加した引数は以下の通り。
```
  --filter-type FILTER_TYPE
                        application KenLM filter
  --filter-threshold FILTER_THRESHOLD
                        threshold of filter
  --used-ngram-model USED_NGRAM_MODEL
                        n-gram model for KenLM scoring
  --display-ngram-score
                        display n-gram score by KenLM
  --ngram-reranking     re-ranking by n-gram score
  --remove-contain-oov  remove sentence which hava oov word
  --display-modified-ngram
                        display moified ngram analisys
```

## Scoring-Test
Japanese-Dialog-Transformer ディレクトリ下に移動し以下を実行する。
```
python scripts/scoring/scoring_test.py --filter-type depth-harmonic --filter-threshold -1.5 --used-ngram-model scripts/scoring/models/bccwj-csj-np.bin --display-ngram-score --remove-contain-oov
```
このソースコードはscore_sentence.py内に存在する関数を試験するためのものである。
