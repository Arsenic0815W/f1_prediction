# F1 Race Outcome Predictor - Bahrain GP 2025  
Created by Shubham

## Description  
This project predicts the finishing order of the 2025 Bahrain Grand Prix using a machine learning model trained on historical F1 race data from 2022 to 2024. It utilizes the FastF1 library to retrieve past race statistics and builds a Random Forest Regressor to simulate race outcomes based on team and driver performance factors.

## Technologies Used  
- Python  
- FastF1  
- pandas  
- numpy  
- scikit-learn  
- VS Code  

## Features  
- Fetches real race data from FastF1 for Rounds 4 of 2022, 2023, and 2024 seasons  
- Processes and cleans lap times and positions to calculate average driver pace  
- Applies custom performance multipliers for drivers and teams  
- Trains a Random Forest model to predict driver finishing positions  
- Prints predicted 2025 race standings in a well-formatted table  
- Displays model performance metrics including Mean Absolute Error and R² Score  

## How It Works  
Each driver's average lap pace and a performance score (based on team and driver factors) are used as input features. The model learns from these features and historical results to estimate probable positions. For 2025 predictions, the script simulates performance using known 2025 driver-team pairings.

## Usage  
1. Clone the repository  
2. Install required dependencies  
3. Run the script using any Python environment  
4. Check the terminal output for predicted positions and model performance  

## Notes  
All predictions are experimental and intended for fun. The real world of F1 is far more dynamic and unpredictable.
