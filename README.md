# StockPredict

### v0.1 Predict the moving direction of stock index
#### 1. Data Preparing

* Using close price everyday for stock index in china

#### 2. Model Training

* model: LSTM
* target: up/down/stay
* measure: accuracy of prediction

#### 3. Testing

* Sliding window, using next day's close price minus today' to evaluate the correctness of prediction.
