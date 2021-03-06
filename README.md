# Integrating Representation and Interaction for Context-aware Document Ranking

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Abstract
This repository contains the source code for the TOIS paper [Integrating Representation and Interaction for Context-aware Document Ranking](https://dl.acm.org/doi/10.1145/3529955) by Chen et al. <br>

Recent studies show that historical behaviors (such as queries and their clicks) contained in a search session can benefit the ranking performance of subsequent queries in the session. Existing neural context-aware ranking models usually rank documents based on either latent representations of user search behaviors, or the word-level interactions between the candidate document and each historical behavior in the search session. However, these two kinds of models both have their own drawbacks. Representation based models neglect fine-grained information of word-level interactions, whereas interaction based models suffer from the length restriction of session sequence because of the large cost on word-level interactions. To complement the limitations of these two kinds of models, we propose a unified context-aware document ranking model which takes full advantage of both representation and interaction. Specifically, instead of matching a candidate document with every single historical query in a session, we encode the session history into a latent representation and use this representation to enhance the current query and the candidate document. We then just match the enhanced query and candidate document with several matching components to capture the fine-grained information of word-level interactions. Rich experiments on two public query logs prove the effectiveness and the efficiency of our model for leveraging representation and interaction.

Authors: Haonan Chen, Zhicheng Dou, Qiannan Zhu, Xiaochen Zuo, and Ji-rong Wen

## Requirements
- Python 3.6.2 <br>
- Pytorch 1.4.0 (with GPU support) <br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5  

## Usage
- Obtain the data
  - AOL: Please reach to the author of [CARS](https://arxiv.org/pdf/1906.02329.pdf)
  - Tiangong-ST: [Download here](http://www.thuir.cn/tiangong-st/)

## Citations
If you use the code, please cite the following paper:  
```
@article{CDZZW2022TOIS,
    author = {Chen, Haonan and Dou*, Zhicheng and Zhu, Qiannan and Zuo, Xiaochen and Wen, Ji-Rong},
    title = {Integrating Representation and Interaction for Context-Aware Document Ranking},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1046-8188},
    url = {https://doi.org/10.1145/3529955},
    doi = {10.1145/3529955},
    journal = {ACM Trans. Inf. Syst.}
}
```