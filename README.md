# DocOIE: A Document-level Context-Aware Dataset for OpenIE

DocOIE: **Doc**ument-level context-aware **O**pen **I**nformation **E**xtraction

## Introduction
DocOIE is a document-level context-aware dataset for Open Information Extraction.
DocIE is a context-aware Open Information Extraction Model.
The details of this work are elaborated in our paper published in [Findings of ACL 2021](https://arxiv.org/abs/2105.04271).

## DocOIE dataset
DocOIE consists of evaluation and training dataset.

**Evaluation dataset** ``\DocOIE\data_evaluation\`` contains 800 expert-annotated sentences, sampled from 80 documents in two domains (healthcare and transportation).


**Training dataset**  ``\DocOIE\data_train\`` contains 2,400 documents from the two domains (healthcare and transportation); 1,200 documents in each domain. All sentences from these documents are used to bootstrap pseudo labels for neural model training.  
**Note**: Only document IDs are included in DocOIE Training dataset, for document collection at [PatFT](http://patft.uspto.gov/).


## DocIE model
### Installation Instructions
Credits given to [IMoJIE](https://github.com/dair-iitd/imojie), we build our DocIE model based on it.

Use a python-3.7 environment and install the dependencies using,
```
pip install -r requirements.txt
```
This will install custom versions of allennlp and pytorch_transformers based on the code in the folder.

### Training dataset prepration
Collect training dataset according to document IDs. Use [ReVerb](https://github.com/knowitall/reverb) and [OpenIE4](https://github.com/knowitall/openie) to extract pseudo labels for model training.
Place the training files to 
- ``data/train/healthcare/train_extractions_reverb_oie4.json`` for healthcare domain
- ``data/train/transport/train_extractions_oie4_reverb.json`` for transportation domain
### Running the code

```
python allennlp_script.py --param_path DocIE/configs/docie_healthcare.json --s trained_models/doc_healthcare --beam_size 3 --max_length 500 --max_sent_length 40 --context_window 5 --top_encoder true
```
Arguments:
- param_path: file containing all the parameters for the model
- s:  path of the directory where the model will be saved
- context_window: window size of the surrounding sentences
- max_length: maximum length of source sentence and surrounding sentences (note that the max length for BERT is 512).
- max_sent_length: maximum length of each surrounding sentence.
- top_encoder: whether or not to use additonal transformer layers as top enconder.

## Citing
If you use this code in your research, please cite:

```
@inproceedings{dong-etal-2021-docoie,
    title = "{D}oc{OIE}: A Document-level Context-Aware Dataset for {O}pen{IE}",
    author = "Dong, Kuicai  and
      Yilin, Zhao  and
      Sun, Aixin  and
      Kim, Jung-Jae  and
      Li, Xiaoli",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.210",
    doi = "10.18653/v1/2021.findings-acl.210",
    pages = "2377--2389",
}

```


## Contact
In case of any issues, please send a mail to
```kuicai001 (at) e (dot) ntu (dot) edu (dot) sg```


