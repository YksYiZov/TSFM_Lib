from selector.baseline_select import Baseline_Select_Model
import numpy as np
import math
import importlib
import traceback
import torch # 即使不直接用，处理tensor转换也可能需要
from gluonts.dataset.common import ListDataset

class MyFancySelectModel(Baseline_Select_Model):

    def _get_select_strategy(self, dataset):
        """
        Ranking-based Model Selection with optional Quality & Diversity
        - Always returns a FULL ordering of all models
        - Downstream is responsible for truncation (Top-K)
        """

        training_data_iter = dataset.validation_dataset.dataset
        required_pred_len = dataset.prediction_length

        all_series = list(training_data_iter)
        total_sequences = len(all_series)

        subset_size = min(256, max(1, total_sequences // 2))
        num_subsets = 2

        # ==========================
        # 消融控制
        # ==========================
        use_quality = True
        use_diversity = True
        alpha = 0.7

        def select_strategy(dataset_name=None):

            model_quality = {}      # model_id -> MAE
            model_residuals = {}    # model_id -> residual vector

            # ======================================================
            # 1. Model Evaluation (完全沿用你原有逻辑)
            # ======================================================
            for family, size_dict in self.Model_sizes.items():
                for size_name, model_info in size_dict.items():
                    model_id = model_info["id"]

                    try:
                        module = importlib.import_module(model_info["model_module"])
                        ModelClass = getattr(module, model_info["model_class"])

                        model_instance = ModelClass(
                            self.args,
                            module_name=model_info["module_name"],
                            model_name="My_Fancy_Select",
                            model_local_path=model_info["model_local_path"],
                        )

                        model_instance.model_size = f"{family}_{size_name}".split("_")[-1]

                        subset_metrics = []
                        residual_buffer = []

                        for _ in range(num_subsets):

                            indices = np.random.choice(total_sequences, subset_size, replace=False)

                            backtest_input_list = []
                            backtest_ground_truths = []

                            for idx in indices:
                                entry = all_series[idx]
                                full_target = entry["target"]

                                if len(full_target) > required_pred_len:
                                    backtest_input_list.append({
                                        "start": entry["start"],
                                        "target": full_target[:-required_pred_len],
                                        "feat_static_cat": entry.get("feat_static_cat"),
                                        "feat_dynamic_real": entry.get("feat_dynamic_real"),
                                        "item_id": entry.get("item_id", f"sample_{idx}"),
                                        "past_feat_dynamic_real": entry.get("past_feat_dynamic_real"),
                                    })
                                    backtest_ground_truths.append(full_target[-required_pred_len:])

                            if not backtest_input_list:
                                subset_metrics.append(math.inf)
                                continue

                            predictor = model_instance.get_predictor(dataset, 128)
                            forecasts = predictor.predict(backtest_input_list)

                            errors = []

                            for i, fc in enumerate(forecasts):
                                truth = backtest_ground_truths[i]

                                if hasattr(fc, "mean"):
                                    pred = fc.mean
                                elif hasattr(fc, "quantile"):
                                    pred = fc.quantile(0.5)
                                else:
                                    pred = np.mean(fc.samples, axis=0)

                                if pred.ndim != truth.ndim:
                                    pred = pred.reshape(truth.shape)

                                errors.append(np.mean(np.abs(pred - truth)))
                                residual_buffer.append((pred - truth).reshape(-1))

                            subset_metrics.append(np.mean(errors))

                        model_quality[model_id] = np.mean(subset_metrics)
                        model_residuals[model_id] = (
                            np.concatenate(residual_buffer)
                            if residual_buffer else None
                        )

                    except Exception as e:
                        print(f"Error evaluating model {model_id}: {e}")
                        traceback.print_exc()
                        model_quality[model_id] = math.inf
                        model_residuals[model_id] = None

            # ======================================================
            # 2. Greedy Ranking (关键修改点)
            # ======================================================
            remaining_models = list(model_quality.keys())
            ordered_models = []

            while remaining_models:

                best_model = None
                best_score = (-math.inf, -math.inf)  # (main_score, fallback_quality)

                for m in remaining_models:

                    fallback_quality = -model_quality[m]  # MAE 越小越好
                    score = 0.0

                    if use_quality:
                        score += alpha * fallback_quality

                    if use_diversity and ordered_models:
                        corrs = []
                        r1 = model_residuals.get(m)

                        for s in ordered_models:
                            r2 = model_residuals.get(s)
                            if r1 is None or r2 is None:
                                continue
                            if np.std(r1) == 0 or np.std(r2) == 0:
                                continue
                            corrs.append(np.corrcoef(r1, r2)[0, 1])

                        if corrs:
                            score += (1 - alpha) * (1 - np.mean(corrs))

                    candidate_score = (score, fallback_quality)

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_model = m

                if best_model is None:
                    best_model = remaining_models[0]

                ordered_models.append(best_model)
                remaining_models.remove(best_model)

            # 与原接口保持一致
            return ordered_models, self.args.ensemble_size

        return select_strategy

