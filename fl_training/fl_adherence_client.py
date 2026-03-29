"""
FL Client — Adherence Prediction (Binary Classification)
Predicts: 0 = Low adherence (<0.6), 1 = High adherence (>=0.6)

Usage:
    python fl_adherence_client.py --client_id 1
    python fl_adherence_client.py --client_id 2
    python fl_adherence_client.py --client_id 3
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import io
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prepare_adherence_data(df):
    features, targets = [], []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        if len(user_data) < 3:
            continue
        for _, row in user_data.iterrows():
            calorie_deficit_pct = (row['tdee'] - row['calorie_target']) / row['tdee']
            features.append([
                row['age'],
                1 if row['gender'] == 'M' else 0,
                row['start_weight'],
                row['gym_days_target'],
                calorie_deficit_pct,
                1 if row['goal'] == 'weight_loss' else 0,
                1 if row['goal'] == 'muscle_gain' else 0,
                1 if row['activity_level'] == 'sedentary' else 0,
                1 if row['activity_level'] == 'active' else 0,
                row['week']
            ])
            # Binary: 0 = Low (<0.6), 1 = High (>=0.6)
            targets.append(1 if row['adherence_score'] >= 0.6 else 0)
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
    X, y = prepare_adherence_data(client_df)
    print(f"   Client {client_id}: {len(X)} samples | Low={( y==0).sum()} High={(y==1).sum()}")
    return X, y


def model_to_params(model):
    buf = io.BytesIO()
    pickle.dump(model, buf)
    buf.seek(0)
    return [np.frombuffer(buf.read(), dtype=np.uint8)]


def params_to_model(params):
    buf = io.BytesIO(bytes(params[0].astype(np.uint8)))
    return pickle.load(buf)


class AdherenceClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        X, y = get_client_data(client_id)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=4, min_samples_leaf=2,
            class_weight='balanced', random_state=42 + client_id
        )
        self.model.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_test, self.model.predict(self.X_test)) * 100
        print(f"\n   Client {client_id} initialized — Accuracy: {acc:.1f}%")

    def get_parameters(self, config):
        return model_to_params(self.model)

    def fit(self, parameters, config):
        server_round = config.get("server_round", 0)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=4, min_samples_leaf=2,
            class_weight='balanced',
            random_state=42 + self.client_id + server_round
        )
        model.fit(self.X_train, self.y_train)
        self.model = model
        acc  = accuracy_score(self.y_test, self.model.predict(self.X_test)) * 100
        loss = 1.0 - (acc / 100)
        print(f"   Client {self.client_id} Round {server_round} — Accuracy: {acc:.1f}%")
        with open(f'models/trained_models/adherence_client_{self.client_id}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        return model_to_params(self.model), len(self.X_train), {"mae": float(loss), "accuracy": float(acc)}

    def evaluate(self, parameters, config):
        global_model = params_to_model(parameters)
        acc  = accuracy_score(self.y_test, global_model.predict(self.X_test)) * 100
        loss = 1.0 - (acc / 100)
        return float(loss), len(self.X_test), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    print("=" * 60)
    print(f"ADHERENCE CLIENT {args.client_id} (Binary Classification)")
    print("=" * 60)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=AdherenceClient(client_id=args.client_id),
    )

if __name__ == "__main__":
    main()