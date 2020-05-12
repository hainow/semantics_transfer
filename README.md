# Language Style Transfer
This repo contains the code and data of the following paper:

<i> "Style Transfer from Non-Parallel Text by Cross-Alignment". Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. NIPS 2017. [arXiv](https://arxiv.org/abs/1705.09655)</i>

The method learns to perform style transfer between two non-parallel corpora. For example, given positive and negative reviews as two corpora, the model can learn to reverse the sentiment of a sentence.
<p align="center"><img width=800 src="img/example_sentiment.png"></p>

<br>


## Data Format
Please name the corpora of two styles by "formal.train" and "formal.test" 
respectively with parallel sentences.  Similarly for test segments. 

The <code>data/full_data/</code> directory contains train and test 
data. 

The <code>semantics/</code> directory contains the code for semantics feature 
extraction and the <code>semantics_vector/</code> directory is the output of 
that extraction.  

<br>

## Quick Start
- To train a model, first create a <code>tmp/</code> folder (where the model 
and results will be saved), then go to the <code>code/</code> folder and run 
the following command with `C` being the factor for KL loss, and `epsilon` 
being the factor for cycle loss:
```bash
python seq2seq_cycle_vae.py --load_model false --model ../gyafc_tfd_vae_cyc/ --vocab ../gyafc_tfd_vae_cyc/vocab --max_seq_length 25 --output ../gyafc_tfd_vae_cyc/ --C 0.25 --epsilon 1.  --train ../data/GYAFC_Corpus/Entertainment_Music/train/
```

- To test the model, run the following command:
```bash
python seq2seq_cycle_vae.py --load_model true  --model ../gyafc_baseline/  --vocab ../gyafc_baseline/vocab --max_epochs 20 --test ../data/GYAFC_Corpus/Entertainment_Music/test/ --C 0 --epsilon 1
```

<br>

## Classification of formality 
- To train the classifider: 
```bash
python classifier.py --vocab ../gyafc_baseline_cyc/vocab --max_epochs 20 --model ../classifier_model/
```

- To test the classifider (you need to change the stored output 
from training accordingly): 
```bash
python classifier.py --test  --test_transfer ../gyafc_baseline/ --model ../classifier_model/ --load_model true --vocab ../gyafc_baseline/vocab
```

## Dependencies
Python >= 3.5, TensorFlow 1.15
