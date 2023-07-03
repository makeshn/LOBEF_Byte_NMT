Code Accompanying [Local Byte Fusion for Machine Translation](https://arxiv.org/abs/2205.11490)

Adapted from https://github.com/UriSha/EmbeddinglessNMT/.
## Requirements

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

## Setting up environment
```bash
git clone https://github.com/makeshn/LOBEF_Byte_NMT.git
cd LOBEF_Byte_NMT
pip install -e .
mkdir results
cd examples/translation
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
cd ../..
```
**Edit examples/translation/mosesdecoder/scripts/training/clean-corpus-n.perl to run on bytes**:

From:
```bash
sub word_count {
 my ($line) = @_;
 if ($ignore_xml) {
  $line =~ s/<\S[^>]*\S>/ /g;
  $line =~ s/\s+/ /g;
  $line =~ s/^ //g;
  $line =~ s/ $//g;
 }
 my @w = split(/ /,$line);
 return scalar @w;
}
```
To:
```bash
sub word_count {
 use bytes;
 my ($line) = @_;
 if ($ignore_xml) {
  $line =~ s/<\S[^>]*\S>/ /g;
  $line =~ s/\s+/ /g;
  $line =~ s/^ //g;
  $line =~ s/ $//g;
 }
 return length($line);
}
```
## Download raw and pre-processed data from:
* Data can be downloaded from this [link](https://drive.google.com/drive/folders/1VkUxFjgVWcElZXZ4_bow_vfVYEpiT3A6?usp=sharing)
* Data can be found in the following folders
    * OPUS data - opus.tar.gz for raw data, byte-bin for preprocessed data
    * Cross domain data - cross_domain_adaptation.zip
    * Cross lingual transfer - cross_lingual_transfer.zip
* Move the folders to the data/ folder and update the paths in the scripts.
## Train and evaluate:
```
bash embeddingless_scripts/train_byte_ncf.sh
bash embeddingless_scripts/train_byte_wsf.sh
```
