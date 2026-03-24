import argparse
import numpy as np
import pandas as pd
import pickle
import flwr as fl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os

def prepare_weight_data(df):
    """Extract features and targets for weight prediction."""
    features, targets, user_ids = [], [], []

    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        for i in range(len(user_data) - 1):
            curr = user_data.iloc[i]
            nxt  = user_data.iloc[i + 1]
            features.append([
                curr['weight'],
                curr['avg_daily_calories'],
                curr['avg_daily_protein'],
                curr['gym_days'],
                curr['age'],
                1 if curr['gender'] == 'M' else 0,
                curr['tdee']
            ])
            targets.append(nxt['weight'])
            user_ids.append(user_id)
    return np.array(features), np.array(targets), user_ids


def get_client_data(client_id: int):
    """Load data and return only this client's portion."""
    df = pd.read_csv('fl_training_data.csv')
    X, y, user_ids = prepare_weight_data(df)
    # Fit scaler on full data (same as before — ensures consistent scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split users into 3 clients (same seed as original)
    unique_users = list(set(user_ids))
    np.random.seed(42)
    np.random.shuffle(unique_users)
    users_per_client = len(unique_users) // 3
    start = (client_id - 1) * users_per_client
    end   = None if client_id == 3 else start + users_per_client
    client_users = unique_users[start:end]
    mask = np.isin(user_ids, client_users)
    X_client = X_scaled[mask]
    y_client = np.array(y)[mask]

    # Save scaler from client 1 (used by everyone)
    if client_id == 1:
        os.makedirs('models/trained_models/3client_pfl', exist_ok=True)
        with open('models/trained_models/3client_pfl/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f" Scaler saved by Client 1")
    print(f" Client {client_id}: {len(X_client)} samples from {len(client_users)} users")
    print(f" Users: {sorted(client_users)}")
    return X_client, y_client, scaler, df, client_users

def model_to_params(model: GradientBoostingRegressor):
    """Serialize model to flwr Parameters using pickle bytes."""
    import io
    buf = io.BytesIO()
    pickle.dump(model, buf)
    buf.seek(0)
    model_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
    return [model_bytes]


def params_to_model(params):
    """Deserialize flwr Parameters back to sklearn model."""
    import io
    model_bytes = bytes(params[0].astype(np.uint8))
    buf = io.BytesIO(model_bytes)
    return pickle.load(buf)

class GymClient(fl.client.NumPyClient):
    """
    Flower client representing one gym.
    Each round:
      1. Receives global model params from server
      2. Fine-tunes on local data
      3. Sends updated params back to server
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.X, self.y, self.scaler, self.df, self.users = get_client_data(client_id)
        # Train/test split for local evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        # Initialize local model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42 + client_id
        )
        # Initial fit before federation starts
        self.model.fit(self.X_train, self.y_train)
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        print(f"\n Client {client_id} initialized — Local MAE: {mae:.4f} kg")

    def get_parameters(self, config):
        """Send local model params to server."""
        return model_to_params(self.model)

    def fit(self, parameters, config):
        """
        Receive global model, fine-tune on local data, return updated model.
        This is the core FL step.
        """
        server_round = config.get("server_round", 0)
        print(f"\n Client {self.client_id} — Round {server_round} training...")
        # Load global model from server
        global_model = params_to_model(parameters)
        # Fine-tune on local data (warm start from global model)
        # We create a new model but initialize from global predictions as pseudo-labels
        # blended with real labels — this simulates fine-tuning
        global_preds = global_model.predict(self.X_train)
        
        # Blend: 70% real labels + 30% global predictions for fine-tuning
        blended_y = 0.7 * self.y_train + 0.3 * global_preds

        local_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42 + self.client_id
        )
        local_model.fit(self.X_train, blended_y)
        self.model = local_model
        # Evaluate locally
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        print(f" Client {self.client_id} — After round {server_round} MAE: {mae:.4f} kg")
        # Save this client's trained model
        with open(f'models/trained_models/3client_pfl/client_{self.client_id}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        return model_to_params(self.model), len(self.X_train), {"mae": float(mae)}

    def evaluate(self, parameters, config):
        """Evaluate global model on local test data."""
        global_model = params_to_model(parameters)
        mae  = mean_absolute_error(self.y_test, global_model.predict(self.X_test))
        r2   = r2_score(self.y_test, global_model.predict(self.X_test))
        print(f" Client {self.client_id} eval — MAE: {mae:.4f} kg | R²: {r2:.4f}")
        return float(mae), len(self.X_test), {"mae": float(mae), "r2": float(r2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True, choices=[1, 2, 3],
                        help='Client ID (1, 2, or 3)')
    args = parser.parse_args()
    print(f" GYM CLIENT {args.client_id} — Connecting to FL Server")
    client = GymClient(client_id=args.client_id)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

if __name__ == "__main__":
    main()