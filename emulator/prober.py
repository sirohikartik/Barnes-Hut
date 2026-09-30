import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


class PhysicalProbeSuite:
    """
    Suite for probing internal neural representations h_l of the diffusion emulator
    to decode physical invariants and quantities.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @staticmethod
    def extract_target_arrays(physics_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Flattens and batches physical metadata into target numpy arrays.
        """
        targets = {
            "total_energy": [],
            "kinetic_energy": [],
            "potential_energy": [],
            "angular_momentum_norm": [],
            "angular_momentum_vec": [],
            "linear_momentum_norm": [],
            "linear_momentum_vec": [],
            "positions": [],
            "velocities": [],
        }

        for item in physics_list:
            targets["total_energy"].append(item["total_energy"])
            targets["kinetic_energy"].append(item["kinetic_energy"])
            targets["potential_energy"].append(item["potential_energy"])
            targets["angular_momentum_norm"].append(item["angular_momentum_norm"])
            targets["angular_momentum_vec"].append(item["angular_momentum"])
            targets["linear_momentum_norm"].append(item["linear_momentum_norm"])
            targets["linear_momentum_vec"].append(item["linear_momentum"])
            targets["positions"].append(item["positions"])
            targets["velocities"].append(item["velocities"])

        return {k: np.array(v, dtype=np.float32) for k, v in targets.items()}

    def evaluate_representation(
        self,
        train_h: np.ndarray,
        test_h: np.ndarray,
        train_targets: Dict[str, np.ndarray],
        test_targets: Dict[str, np.ndarray],
        probe_type: str = "linear",
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains a probe mapping representation h -> physical_quantity
        and evaluates R^2, RMSE, and correlation on held-out test data.
        """
        results = {}

        # Standardize representation
        h_mean = np.mean(train_h, axis=0)
        h_std = np.std(train_h, axis=0) + 1e-7
        norm_train_h = (train_h - h_mean) / h_std
        norm_test_h = (test_h - h_mean) / h_std

        for target_name in train_targets.keys():
            y_train = train_targets[target_name]
            y_test = test_targets[target_name]

            # If 1D target shape (N, 1), flatten
            if y_train.ndim == 2 and y_train.shape[1] == 1:
                y_train_flat = y_train.ravel()
                y_test_flat = y_test.ravel()
            else:
                y_train_flat = y_train
                y_test_flat = y_test

            # Standardize targets for stable fitting
            y_mean = np.mean(y_train_flat, axis=0)
            y_std = np.std(y_train_flat, axis=0) + 1e-7
            norm_y_train = (y_train_flat - y_mean) / y_std
            norm_y_test = (y_test_flat - y_mean) / y_std

            if probe_type == "linear":
                from sklearn.linear_model import RidgeCV
                probe = RidgeCV(alphas=[1e-4, 1e-2, 0.1, 1.0, 10.0])
                probe.fit(norm_train_h, norm_y_train)
                norm_pred = probe.predict(norm_test_h)
            elif probe_type == "mlp":
                probe = MLPRegressor(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    max_iter=300,
                    alpha=1e-3,
                    random_state=42,
                )
                probe.fit(norm_train_h, norm_y_train)
                norm_pred = probe.predict(norm_test_h)
            else:
                raise ValueError(f"Unknown probe type: {probe_type}")

            # Transform predictions back to original physical units
            y_pred = norm_pred * y_std + y_mean

            # Calculate metrics
            r2 = float(r2_score(y_test_flat, y_pred))
            mse = float(mean_squared_error(y_test_flat, y_pred))
            rmse = float(np.sqrt(mse))
            nrmse = float(rmse / (np.std(y_test_flat) + 1e-7))

            # Pearson correlation (for 1D quantities)
            corr = 0.0
            if y_test_flat.ndim == 1 or y_test_flat.shape[1] == 1:
                flat_t = y_test_flat.ravel()
                flat_p = y_pred.ravel()
                if np.std(flat_p) > 1e-7 and np.std(flat_t) > 1e-7:
                    corr = float(pearsonr(flat_t, flat_p)[0])
            else:
                corr = float(np.mean([
                    pearsonr(y_test_flat[:, j], y_pred[:, j])[0]
                    for j in range(y_test_flat.shape[1])
                    if np.std(y_pred[:, j]) > 1e-7 and np.std(y_test_flat[:, j]) > 1e-7
                ]))

            results[target_name] = {
                "r2": r2,
                "rmse": rmse,
                "nrmse": nrmse,
                "corr": corr,
            }

        return results
