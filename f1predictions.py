import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

def fetch_f1_race_data(year, round_number):
    try:
        race = fastf1.get_session(year, round_number, 'R')
        race.load()
        results = race.results[['DriverNumber', 'FullName', 'TeamName', 'Position', 'Points']]
        results = results.rename(columns={'FullName': 'Driver'})
        laps = race.laps
        avg_pace = laps.groupby('Driver')['LapTime'].apply(
            lambda x: x.mean().total_seconds() if not x.empty else None
        ).reset_index(name='AvgPace_sec')
        results = pd.merge(results, avg_pace, on='Driver', how='left')
        return results
    except Exception as e:
        print(f"Error fetching race data for {year} Round {round_number}: {e}")
        return None

def clean_race_data(df):
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    return df

def apply_race_factors(df):
    team_factors = {
        'Red Bull Racing': 0.98, 'Ferrari': 0.99, 'McLaren': 1.00,
        'Mercedes': 1.01, 'Aston Martin': 1.02, 'Williams': 1.03,
        'RB': 1.04, 'Haas F1 Team': 1.05, 'Kick Sauber': 1.05,
        'Alpine': 1.06, 'Sauber': 1.05, 'AlphaTauri': 1.04
    }
    driver_factors = {
        'Max Verstappen': 0.97, 'Lewis Hamilton': 0.98, 'Charles Leclerc': 0.99,
        'Lando Norris': 1.00, 'Carlos Sainz': 1.00, 'George Russell': 1.01,
        'Fernando Alonso': 1.02, 'Oscar Piastri': 1.02, 'Sergio Perez': 1.03,
        'Valtteri Bottas': 1.03, 'Esteban Ocon': 1.04, 'Pierre Gasly': 1.04,
        'Alex Albon': 1.03, 'Yuki Tsunoda': 1.04, 'Daniel Ricciardo': 1.03,
        'Kevin Magnussen': 1.05, 'Nico Hulkenberg': 1.05, 'Zhou Guanyu': 1.06,
        'Logan Sargeant': 1.07, 'Liam Lawson': 1.05, 'Oliver Bearman': 1.06
    }
    df['PerformanceScore'] = 1.0
    for idx, row in df.iterrows():
        team_factor = team_factors.get(row['TeamName'], 1.05)
        driver_factor = driver_factors.get(row['Driver'], 1.02)
        df.loc[idx, 'PerformanceScore'] = team_factor * driver_factor
    return df

def predict_race_outcome(model, latest_data):
    driver_teams = {
        'Max Verstappen': 'Red Bull Racing', 'Yuki Tsunoda': 'RB',
        'Charles Leclerc': 'Ferrari', 'Lewis Hamilton': 'Mercedes',
        'Lando Norris': 'McLaren', 'Oscar Piastri': 'McLaren',
        'George Russell': 'Mercedes', 'Andrea Kimi Antonelli': 'Mercedes',
        'Fernando Alonso': 'Aston Martin', 'Lance Stroll': 'Aston Martin',
        'Pierre Gasly': 'Alpine', 'Jack Doohan': 'Alpine',
        'Esteban Ocon': 'Haas F1 Team', 'Oliver Bearman': 'Haas F1 Team',
        'Alexander Albon': 'Williams', 'Carlos Sainz': 'Williams',
        'Nico Hulkenberg': 'Kick Sauber', 'Gabriel Bortoleto': 'Kick Sauber',
        'Isack Hadjar': 'RB', 'Liam Lawson': 'RB'
    }
    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'TeamName'])
    results_df = apply_race_factors(results_df)
    results_df['PredictedPace'] = 95.0 * results_df['PerformanceScore'] + np.random.normal(0, 0.2, len(results_df))
    results_df = results_df.sort_values('PredictedPace')
    
    # Adjusting the print formatting
    print("\nBahrain GP 2025 Race Predictions:")
    print("=" * 100)
    print(f"{'Position':<10}{'Driver':<25}{'Team':<25}{'Predicted Pace (s)':<20}")
    print("-" * 100)
    
    for idx, row in results_df.iterrows():
        print(f"{idx+1:<10}{row['Driver']:<25}{row['TeamName']:<25}{row['PredictedPace']:.3f}")
    print("=" * 100)

if __name__ == "__main__":
    print("Initializing F1 race prediction model...")
    race_data = []
    for year in [2022, 2023, 2024]:
        df = fetch_f1_race_data(year, 4)
        if df is not None:
            df['Year'] = year
            race_data.append(df)
    if race_data:
        combined_df = pd.concat(race_data, ignore_index=True)
        combined_df = clean_race_data(combined_df)
        combined_df = apply_race_factors(combined_df)
        features = combined_df[['AvgPace_sec', 'PerformanceScore']]
        target = combined_df['Position']
        imputer = SimpleImputer(strategy='median')
        features_clean = imputer.fit_transform(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, target)
        predict_race_outcome(model, combined_df)
        y_pred = model.predict(features_scaled)
        mae = mean_absolute_error(target, y_pred)
        r2 = r2_score(target, y_pred)
        print("\nModel Performance Metrics:")
        print(f'Mean Absolute Error: {mae:.2f} positions')
        print(f'R^2 Score: {r2:.2f}')
    else:
        print("Failed to fetch race data.")