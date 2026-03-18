# Project Scope

## Project Title
Sentiment-Augmented Financial Prediction

## Research Question
Do sentiment-based features from financial news and SEC filings improve short-term stock price movement prediction beyond financial statement and market-based features?

## Project Type
Supervised learning classification project.

## Prediction Target
Binary classification:
- Label = 1 if 5-day forward return > 0
- Label = 0 otherwise

## Unit of Observation
One row = one stock on one date.

## Initial Universe
30-40 Large cap S&P 500 companies from a single sector. 
Why? focused sector subset allows for us better comparability of finance rations, keeps it manageable, and preserves the strong market coverage for later sentiment modeling. 

## Time Horizon
2015-2024

## Feature Layers
### Layer 1
Financial statement features only
Examples:
- Current ratio
- Debt-to-equity
- Net margin
- ROA
- ROE
- Revenue growth
- Earnings growth

### Layer 2
Add market data features
Examples:
- 5-day return
- 21-day return
- rolling volatility
- volume ratio
- RSI

### Layer 3
Add sentiment features
Examples:
- headline sentiment score
- sentiment momentum
- news volume
- MD&A sentiment delta

## Models to Compare
### In-class
- Logistic Regression
- Random Forest
- Bagging
- Gradient Boosting / AdaBoost

### Outside-class
- Lasso / Ridge if used appropriately
- XGBoost

## Validation Strategy
Time-series split / walk-forward validation.
No random split.
Final holdout period reserved for out-of-sample testing.

## Evaluation Metrics
- AUC-ROC
- F1-score
- Precision
- Recall
- Log loss
- Confusion matrix

## Deliverables
- Clean merged modeling dataset
- Baseline results for Layer 1
- Comparison across Layers 1, 2, and 3
- Tables/plots for presentation
- Final report and slide deck

## In Scope
- Short-term stock direction prediction
- Comparison of feature layers
- Model evaluation
- Interpretation of feature importance

## Out of Scope
- Live trading
- production deployment
- perfect backtesting engine
- portfolio optimization
- real-time streaming pipeline

## MVP Definition
The project is successful at minimum if:
1. A clean dataset is built
2. At least one baseline classifier is trained
3. Results are evaluated correctly with time-aware validation
4. Layer comparison is shown clearly