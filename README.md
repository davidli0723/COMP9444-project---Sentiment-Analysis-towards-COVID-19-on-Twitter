# COMP9444-project---Sentiment-Analysis-towards-COVID-19-on-Twitter

Area of Research: Natural Language Processing (NLP). 

Problem Statement: Sentiment analysis of Tweets is one of the most popular tasks in the NLPspace; it has many practical applications including stock prediction, box office results estimation, survey of popular opinions and content recommendation. This project involves estimating the sentiment of tweets about the COVID-19 pandemic to track people’s opinions on major events surrounding it.

Methodology: Experience and compare different language models' performance when doing the sentiment analysis

| Models                     | Accuracy | F1-score on Neu | F1-score on Neg | F1-score on Pos |
|-----------------------------|-----------|-----------------|-----------------|-----------------|
| Word2Vec/BiLSTM             | 0.85      | 0.91            | 0.75            | 0.60            |
| Word2Vec/BiLSTM (original paper) | 0.749     | -               | -               | -               |
| GloVe/BiLSTM                | 0.82      | 0.88            | 0.63            | 0.52            |
| GloVe/BiLSTM (original paper) | 0.743     | -               | -               | -               |
| BERT/BiLSTM                 | 0.74      | 0.85            | 0.16            | 0.00            |
| BERT/CNN                    | 0.78      | 0.87            | 0.42            | 0.14            |
| BERT                        | 0.96      | 0.94            | 0.97            | 0.98            |
| BERT (original paper)       | 0.932     | -               | -               | -               |
