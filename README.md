# ![image-captioning](https://dl.dropboxusercontent.com/u/16169065/header.png)

#### Introduction

This repository contains sample code for [A Distributed Representation Based Query Expansion Approach for Image Captioning](http://www.semihyagcioglu.com/projects/image-captioning).

For more details, please visit the [project page](http://www.semihyagcioglu.com/projects/image-captioning).

#### Requirements

- Python 2.7 (Packages such as numpy, scipy etc)

You can install these packages with **pip**. To install all dependencies at once, just run the following command. 

		pip install -r requirements.txt

#### Installation

- You should download and extract pre-trained visual features to the relevant folders. For example from https://github.com/mjhucla/mRNN-CR

#### Notes

- We provide pre-trained word vectors but you can train your word vectors or use another pre-trained word vectors. 

#### Demo

- A sample image captioning task is implemented in **demo.py**

#### Citing

If you find this software useful in your research, please consider citing:

		@inproceedings{yagcioglu2015captioning,
		author 	= {Yagcioglu, Semih and Erdem, Erkut and Erdem, Aykut and Çakıcı Ruket},
		title 	= {A Distributed Representation Based Query Expansion Approach for
		Image Captioning},
		booktitle = {Proceedings of the 53rd Annual Meeting of the Association for
		Computational Linguistics and The 7th International Joint Conference of
		the Asian Federation of Natural Language Processing},
		year 	= {2015},
		organization 	= {ACL}}

#### Licence

**MIT** Licence.