# Fast Click-Through Rate Estimation using Data Aggregates
This repository contains code required to reproduce experiments for this paper. If used please cite:
```
@inproceedings{wiatr2021fast,
  title={Fast Click-Through Rate Estimation Using Data Aggregates},
  author={Wiatr, Roman and S{\l}ota, Renata G and Kitowski, Jacek},
  booktitle={International Conference on Computational Science},
  pages={685--698},
  year={2021},
  organization={Springer}
}
```
## Abstract.
Click-Through Rate estimation is a crucial prediction task in Real-Time Bidding environments prevalent in display advertising. The estimation provides information on how to trade user visits in various systems. Logistic Regression is a popular choice as the model for this task. Due to the amount, dimensionality and sparsity of data, it is challenging to train and evaluate the model. One of the techniques to reduce the training and evaluation cost is dimensionality reduction. In this work, we present Aggregate Encoding, a technique for dimensionality reduction using data aggregates. Our approach is to build aggregate-based estimators and use them as an ensemble of models weighted by logistic regression. The novelty of our work is the separation of feature values according to the value frequency, to better utilise regularization. For our experiments, we use the iPinYou data set, but this approach is universal and can be applied to other problems requiring dimensionality reduction of sparse categorical data.
## Setup
0. Clone this repository :)
1. Install required libraries using [setup.py](setup.py).
2. Download [make-ipinyou-data](https://github.com/wnzhang/https://github.com/wnzhang/make-ipinyou-data) and follow the instructions.
3. Set [IPINYOU_DATA_DIR in CONST.py](src/experiment/ipinyou/CONST.py) to point to the directory containing ```make-ipinyou-data```
4. Run [agge/run.py](src/experiment/ipinyou/agge/run.py) for a demo of Aggregate Encoding.
5. Run [hash/run.py](src/experiment/ipinyou/agge/run.py) for a demo of Hashing Trick

Both demos execute one (or more) configurable advertiser(s).
Example output:
```json
{
	"auc_{advertiser}_1_None_f={features}_b={bins}_bt={type}": ["{AUC for each repeated experiment}"]
}
```
Where:
* ```{advertiser}``` is the advertiser identifier
* ```{features}``` is the amount of features produced by the encoding
* ```{bins}``` amount of bins for Aggregate Encoding or -1
* ```{type}``` type of encoding for Aggregate Encoding or -1

Example output of Aggregate Encoding:
```json
{
	"delta_3476_1_None_f=42_b=1_bt=qcut": [4.247216701507568, 6.185106039047241, 4.1242570877075195],
	"auc_3476_1_None_f=42_b=1_bt=qcut": [59.98832507043696, 59.95801864913126, 59.900236640710794],
	"delta_3476_1_None_f=89_b=5_bt=qcut": [15.858128309249878, 16.339589834213257, 17.3368878364563],
	"auc_3476_1_None_f=89_b=5_bt=qcut": [62.46377849211999, 62.49185137460679, 62.568329792396106],
	"delta_3476_1_None_f=132_b=10_bt=qcut": [23.532950162887573, 28.411821842193604, 23.64673900604248],
	"auc_3476_1_None_f=132_b=10_bt=qcut": [62.81933286284017, 62.85405514603353, 62.97970800051742]
}
```
Example output of Hashing Trick:
```json
{
	"auc_3476_1_None_f=20_b=-1_bt=-1": [59.80883914141225, 59.844038329707786],
	"delta_3476_1_None_f=20_b=-1_bt=-1": [1.9405508041381836, 1.9867839813232422],
	"auc_3476_1_None_f=10_b=-1_bt=-1": [58.24100796427529]
}
```

In case of problems and/or questions contact Roman Wiatr ```rwiatr [at] gmail.com```
