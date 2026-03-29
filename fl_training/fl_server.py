import flwr as fl
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg that saves aggregated model params after each round."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_params = None
        self.round_metrics = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        print(f"\n Round {server_round} — received {len(results)} client models")

        # Pick client with lowest MAE
        best_params = None
        best_mae = float('inf')

        for client_proxy, fit_res in results:
            mae = fit_res.metrics.get('mae', float('inf'))
            print(f"   Client MAE: {mae:.4f} kg")
            if mae < best_mae:
                best_mae = mae
                best_params = fit_res.parameters

        self.best_params = best_params

        # Save params
        os.makedirs('models/trained_models/3client_pfl', exist_ok=True)
        params_list = fl.common.parameters_to_ndarrays(best_params)

        with open('models/trained_models/3client_pfl/fl_global_params.pkl', 'wb') as f:
            pickle.dump(params_list, f)

        print(f"Best model saved (round {server_round}, MAE: {best_mae:.4f})")

        return best_params, {"best_mae": best_mae}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        losses = [r.loss for _, r in results]
        avg_loss = float(np.mean(losses))

        self.round_metrics.append({
            'round': server_round,
            'avg_mae': avg_loss
        })

        print(f"Round {server_round} — Average MAE across clients: {avg_loss:.4f} kg")

        return avg_loss, {"avg_mae": avg_loss}


def main():
    
    print("FLOWER FL SERVER — Weight Prediction Model")
    print("Waiting for 3 clients to connect...")
    print("Start clients in separate terminals:")
    print(" python fl_client.py --client_id 1")
    print(" python fl_client.py --client_id 2")
    print(" python fl_client.py --client_id 3")
    strategy = SaveModelStrategy(
        min_fit_clients=3,          # Wait for all 3 clients
        min_evaluate_clients=3,     # Evaluate on all 3
        min_available_clients=3,    # Don't start until all 3 connected
        fraction_fit=1.0,           # Use 100% of clients each round
        fraction_evaluate=1.0,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),  # 5 FL rounds
        strategy=strategy,
    )
    print("\n FL Training Complete!")


if __name__ == "__main__":
    main()


    