# Stack Overflow-Question-Status-Prediction

Predicts the status of the stack Overflow question - <b>open</b>, <b>not a real question</b>, <b>not constructive</b>, <b>off topic</b>, <b>too localized</b>

## Dataset
<a href="https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data?select=train-sample.csv">Predict Closed Questions on Stack Overflow</a>

## Bidirectional Gated Recurrent Unit Neural Network
- processes sequences with two GRUs
- one of the GRUs takes input in forward direction (past to future) while the other in backward (future to past)
- has just two gates 'input' and 'forget'
- better than LSTM - does not need memory units; easy to modify; faster to train; computationally efficient 

## Data (Text) Visualizations
- WordCloud
- Barchart grid
- Treemap
- Circle Packing

## Performance Metrics
- Accuracy: 97.1%
- Precision: 96.4%
- Recall: 97.3%
- F1 score: 96.8%
