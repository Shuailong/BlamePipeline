BlamePipeline
=============

Implementation of the AAAI 2019 paper: Who Blames Whom in a Crisis? Detecting Blame Ties from News Articles Using Neural Networks

### Task
Given a news article, extract blame ties (who blames whom) between entities in the article.

### Example
<p align="left"><img src="pic/example.png" width="450" alt="CCG with LM"></p>
*An example sentence from our dataset containing a blame tie. The red/bold words are entities involved in a blame tie, and the blue/italic words are supporting evidence that the blame tie exists.*

### Dataset

• Source: New York Times/Wall Street Journal/USA Today

• Time period: 2007/10– 2010/06

|            | USA  | NYT  | WSJ  |
| ---------- | ---- | ---- | ---- |
| days       | 310  | 736  | 648  |
| articles   | 132  | 429  | 438  |
| blame ties | 353  | 787  | 754  |

| Number        | value | Ratio             | value |
| :------------ | ----- | ----------------- | ----- |
| # of articles | 998   | Average -/+ ratio | 2.19  |
| # of samples  | 8562  | Total -/+ ratio   | 3.61  |

### Models

Context Model
![Context Model](pic/contextmodel.png)

### Results

| Model    | Dev F1 | Test F1 |
| -------- | ------ | ------- |
| Entity   | 61.07  | 60.06   |
| Context  | 73.16  | 66.35   |
| Combined | 76.13  | 69.92   |

