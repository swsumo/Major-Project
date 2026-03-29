"""
FL Server — Macro Recommendation Model
Run FIRST, then start 3 clients.

Usage: python fl_macro_server.py
"""

import flwr as fl
import numpy as np
import pickle
import os
from flwr.server.client_proxy import ClientProxy


class MacroSaveStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_params = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        print(f"\n✅ Round {server_round} — received {len(results)} client models")
        best_params = None
        best_mae = float("inf")

        for _, fit_res in results:
            mae = fit_res.metrics.get("mae", float("inf"))
            print(f"   Client MAE: {mae:.4f}g protein")
            if mae < best_mae:
                best_mae = mae
                best_params = fit_res.parameters

        self.best_params = best_params
        os.makedirs("models/trained_models", exist_ok=True)
        params_list = fl.common.parameters_to_ndarrays(best_params)
        with open("models/trained_models/macro_fl_params.pkl", "wb") as f:
            pickle.dump(params_list, f)
        print(f"   💾 Best macro model saved (MAE: {best_mae:.4f}g)")
        return best_params, {"best_mae": best_mae}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        losses = [r.loss for _, r in results]
        avg = float(np.mean(losses))
        print(f"   📊 Round {server_round} avg MAE: {avg:.4f}g")
        return avg, {"avg_mae": avg}


if __name__ == "__main__":
    print("=" * 60)
    print("🌸 FL SERVER — Macro Recommendation Model")
    print("=" * 60)
    print("Waiting for 3 clients...")
    print("  python fl_macro_client.py --client_id 1")
    print("  python fl_macro_client.py --client_id 2")
    print("  python fl_macro_client.py --client_id 3")
    print("=" * 60)

    strategy = MacroSaveStrategy(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8082",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    print("\n🎉 Done! Now run: python fl_macro_finalize.py")