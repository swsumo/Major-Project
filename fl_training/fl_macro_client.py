"""
FL Client — Macro Recommendation Model
Run 3 instances AFTER fl_macro_server.py is running.

Usage:
    python fl_macro_client.py --client_id 1
    python fl_macro_client.py --client_id 2
    python fl_macro_client.py --client_id 3
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import io
import flwr as fl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def prepare_macro_data(df):
    features, targets = [], []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        successful = user_data[user_data['adherence_score'] > 0.7]
        if len(successful) < 2:
            continue
        for _, row in successful.iterrows():
            features.append([
                row['age'],
                1 if row['gender'] == 'M' else 0,
                row['weight'],
                row['start_weight'],
                1 if row['goal'] == 'weight_loss' else 0,
                1 if row['goal'] == 'muscle_gain' else 0,
                1 if row['goal'] == 'maintenance' else 0,
                row['gym_days'],
                1 if row['activity_level'] == 'sedentary' else 0,
                1 if row['activity_level'] == 'light' else 0,
                1 if row['activity_level'] == 'moderate' else 0,
                1 if row['activity_level'] == 'active' else 0,
                row['calorie_target']
            ])
            targets.append(row['protein_target'])
    return np.array(features), np.array(targets)


def get_client_data(client_id):
    df = pd.read_csv('fl_training_data.csv')
    unique_users = list(df['user_id'].unique())
    np.random.seed(42)
    np.random.shuffle(unique_users)
    users_per_client = len(unique_users) // 3
    start = (client_id - 1) * users_per_client
    end   = None if client_id == 3 else start + users_per_client
    client_df = df[df['user_id'].isin(unique_users[start:end])]
    X, y = prepare_macro_data(client_df)
    print(f"   Client {client_id}: {len(X)} samples")
    return X, y


def model_to_params(model):
    buf = io.BytesIO()
    pickle.dump(model, buf)
    buf.seek(0)
    return [np.frombuffer(buf.read(), dtype=np.uint8)]


def params_to_model(params):
    buf = io.BytesIO(bytes(params[0].astype(np.uint8)))
    return pickle.load(buf)


class MacroClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        X, y = get_client_data(client_id)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42 + client_id
        )
        self.model.fit(self.X_train, self.y_train)
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        print(f"\n🏋️  Macro Client {client_id} initialized — MAE: {mae:.4f}g protein")

    def get_parameters(self, config):
        return model_to_params(self.model)

    def fit(self, parameters, config):
        server_round = config.get("server_round", 0)
        global_model = params_to_model(parameters)

        global_preds = global_model.predict(self.X_train)
        blended_y    = 0.7 * self.y_train + 0.3 * global_preds

        local_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42 + self.client_id
        )
        local_model.fit(self.X_train, blended_y)
        self.model = local_model

        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        print(f"   ✅ Macro Client {self.client_id} Round {server_round} — MAE: {mae:.4f}g")

        with open(f'models/trained_models/macro_client_{self.client_id}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        return model_to_params(self.model), len(self.X_train), {"mae": float(mae)}

    def evaluate(self, parameters, config):
        global_model = params_to_model(parameters)
        mae = mean_absolute_error(self.y_test, global_model.predict(self.X_test))
        return float(mae), len(self.X_test), {"mae": float(mae)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    print("=" * 60)
    print(f"🏋️  MACRO CLIENT {args.client_id}")
    print("=" * 60)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8082",
        client=MacroClient(client_id=args.client_id),
    )


if __name__ == "__main__":
    main()