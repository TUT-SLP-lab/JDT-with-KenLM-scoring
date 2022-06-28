# JDT-with-KenLM-scoring
Japanese-Dialog-Transformerの応答候補に対して、KenLMによるN-gram言語モデルでスコアリングを行う。

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
