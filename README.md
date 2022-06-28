# JDT-with-KenLM-scoring
Japanese-Dialog-Transformerの応答候補に対して、KenLMによるN-gram言語モデルでスコアリングを行う。

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
Japanese-Dialog-Transformer ディレクトリ下に移動。
```
python scripts/dialog.py data/sample/bin/  --path checkpoints/japanese-dialog-transformer-1.6B.pt  --beam 80  --min-len 10  --source-lang src  --target-lang dst  --tokenizer space  --bpe sentencepiece  --sentencepiece-model data/dicts/sp_oall_32k.model  --no-repeat-ngram-size 3  --nbest 80  --sampling  --sampling-topp 0.9  --temperature 1.0  --show-nbest 5  --filter-type worst  --filter-threshold -4.8  --used-ngram-model scripts/scoring/models/bccwj-csj-np.bin  --display-ngram-score
```
もし、N-gram言語モデルによるリランキングを行う場合はオプションに以下を追加。
```
--ngram-reranking
```
