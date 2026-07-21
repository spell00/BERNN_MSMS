"""Compatibility wrapper for Ax optimization APIs.

Provides an `optimize` function with the historical managed_loop signature,
implemented on top of `ax.api.client.Client`.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from ax.api.client import Client
    from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

    AX_AVAILABLE = True
except Exception:
    AX_AVAILABLE = False

    def _fallback_value(param: dict[str, Any]) -> Any:
        if "value" in param:
            return param["value"]
        if param.get("type") == "choice":
            return list(param.get("values", [None]))[0]
        if param.get("type") == "range":
            lo, hi = param.get("bounds", [0, 1])
            if isinstance(lo, int) and isinstance(hi, int):
                return int(round((lo + hi) / 2))
            return float((lo + hi) / 2)
        return None

    def optimize(
        parameters: list[dict[str, Any]],
        evaluation_function: Callable[[dict[str, Any]], Any],
        objective_name: str,
        minimize: bool,
        total_trials: int,
        random_seed: int | None = None,
        **_: Any,
    ):
        """Small deterministic fallback when ax-platform is unavailable.

        This keeps CLI smoke tests and minimal one-shot training usable in
        lightweight environments. It evaluates one midpoint/default parameter
        set and returns the historical managed-loop tuple shape.
        """
        parameterization = {p["name"]: _fallback_value(p) for p in parameters}
        raw = evaluation_function(parameterization)
        if isinstance(raw, dict):
            val = raw.get(objective_name, 0.0)
            mean = float(val[0] if isinstance(val, tuple) else val)
        else:
            mean = float(raw)
        values = ({objective_name: {"mean": mean, "sem": 0.0}},)
        return parameterization, values, None, None


if AX_AVAILABLE:
    def _to_param_config(param: dict[str, Any]):
        ptype = param.get("type")
        name = param["name"]
        if ptype == "range":
            lo, hi = param["bounds"]
            is_int = isinstance(lo, int) and isinstance(hi, int)
            return RangeParameterConfig(
                name=name,
                bounds=(lo, hi),
                parameter_type="int" if is_int else "float",
                scaling="log" if param.get("log_scale", False) else "linear",
            )
        if ptype == "choice":
            values = list(param["values"])
            first = values[0]
            if isinstance(first, bool):
                parameter_type = "bool"
            elif isinstance(first, int):
                parameter_type = "int"
            elif isinstance(first, float):
                parameter_type = "float"
            else:
                parameter_type = "str"
            return ChoiceParameterConfig(
                name=name,
                values=values,
                parameter_type=parameter_type,
                is_ordered=param.get("is_ordered"),
            )
        raise ValueError(f"Unsupported Ax parameter type: {ptype}")


    def optimize(
        parameters: list[dict[str, Any]],
        evaluation_function: Callable[[dict[str, Any]], Any],
        objective_name: str,
        minimize: bool,
        total_trials: int,
        random_seed: int | None = None,
        **_: Any,
    ):
        """Managed-loop-like optimize API implemented with Ax Client.

        Returns a tuple compatible with the historical API:
        `(best_parameters, values, experiment, model)`.
        """
        client = Client(random_seed=random_seed)
        client.configure_experiment(parameters=[_to_param_config(p) for p in parameters])
        objective_expr = objective_name if not minimize else f"-{objective_name}"
        client.configure_optimization(objective=objective_expr)

        for _i in range(total_trials):
            for trial_index, parameterization in client.get_next_trials(max_trials=1).items():
                try:
                    raw = evaluation_function(parameterization)
                    if isinstance(raw, dict):
                        val = raw.get(objective_name, None)
                        if isinstance(val, tuple):
                            mean, sem = float(val[0]), float(val[1])
                        else:
                            mean, sem = float(val), 0.0
                    else:
                        mean, sem = float(raw), 0.0
                    client.complete_trial(
                        trial_index=trial_index,
                        raw_data={objective_name: (mean, sem)},
                    )
                except Exception as exc:
                    print(f"[ax_compat] Trial {trial_index} failed: {exc!r}")
                    client.mark_trial_failed(trial_index=trial_index, failed_reason=str(exc))

        best_parameters, best_metrics, _trial_index, _arm_name = client.get_best_parameterization(
            use_model_predictions=False
        )
        best_entry = best_metrics.get(objective_name, (float("nan"), 0.0))
        if isinstance(best_entry, tuple):
            best_mean, best_sem = float(best_entry[0]), float(best_entry[1])
        else:
            best_mean, best_sem = float(best_entry), 0.0

        values = ({objective_name: {"mean": best_mean, "sem": best_sem}},)
        experiment = getattr(client, "_experiment", None)
        model = None
        return best_parameters, values, experiment, model
