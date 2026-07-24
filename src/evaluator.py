import json
from collections import defaultdict
from typing import Dict, List, Any
from .schemas import PilotExample, ModelOutput, OperationType

class Evaluator:
    def evaluate_example(self, example: PilotExample, output: ModelOutput) -> Dict[str, Any]:
        result = {
            "example_id": example.example_id,
            "strategy": output.strategy,
            "domain": example.domain.value,
            "operation_type": example.operation_type.value,
            "tool_accuracy": 0,
            "arg_exact_match": 0,
            "field_accuracy": 0.0,
            "stale_reuse": 0,
            "accidental_deletion": 0,
            "cancellation_failure": 0,
            "rollback_accuracy": 0,
            "clarification_needed": int(example.expected_clarification_behavior is not None),
            "clarification_predicted": int(output.clarification is not None),
            "prompt_tokens": output.prompt_tokens,
            "latency_s": output.latency_s
        }
        
        def norm(val):
            if isinstance(val, str):
                return val.lower().strip().replace(".", "").replace(",", "")
            return val
            
        args_pred = {k: norm(v) for k, v in output.arguments.items()}
        args_gold = {k: norm(v) for k, v in example.expected_normalized_tool_arguments.items()}
        
        if output.tool == example.expected_tool_name:
            result["tool_accuracy"] = 1
            
        if args_pred == args_gold:
            result["arg_exact_match"] = 1
            
        if args_gold:
            matched = sum(1 for k, v in args_gold.items() if args_pred.get(k) == v)
            result["field_accuracy"] = matched / len(args_gold)
        elif not args_gold and not args_pred:
            result["field_accuracy"] = 1.0
            
        stale = False
        for field, vals in example.superseded_values.items():
            for v in vals:
                if norm(v) in args_pred.values():
                    stale = True
                    break
            if stale: break
        result["stale_reuse"] = 1 if stale else 0
        
        acc_del = False
        if example.expected_tool_name is not None:
            for k in example.final_active_state.keys():
                if k not in output.arguments:
                    acc_del = True
                    break
            if result["clarification_needed"] == 0: 
                result["accidental_deletion"] = 1 if acc_del else 0
            
        if example.operation_type == OperationType.CANCELLATION:
            if output.tool is not None:
                result["cancellation_failure"] = 1
                
        if example.operation_type == OperationType.ROLLBACK:
            if result["arg_exact_match"] == 1 and result["tool_accuracy"] == 1:
                result["rollback_accuracy"] = 1
                
        return result

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        agg = defaultdict(lambda: {"sum": 0, "count": 0})
        
        for r in results:
            strat = r["strategy"]
            for m in ["tool_accuracy", "arg_exact_match", "field_accuracy", "stale_reuse", 
                     "accidental_deletion", "cancellation_failure", "rollback_accuracy",
                     "prompt_tokens", "latency_s"]:
                
                if m == "cancellation_failure" and r["operation_type"] != "cancellation": continue
                if m == "rollback_accuracy" and r["operation_type"] != "rollback": continue
                
                agg[f"{strat}_{m}"]["sum"] += r[m]
                agg[f"{strat}_{m}"]["count"] += 1
                
                op = r["operation_type"]
                agg[f"{strat}_{op}_{m}"]["sum"] += r[m]
                agg[f"{strat}_{op}_{m}"]["count"] += 1
                
                dom = r["domain"]
                agg[f"{strat}_{dom}_{m}"]["sum"] += r[m]
                agg[f"{strat}_{dom}_{m}"]["count"] += 1

        final = {}
        for k, v in agg.items():
            if v["count"] > 0:
                final[k] = v["sum"] / v["count"]
            else:
                final[k] = 0.0
                
        for strat in set(r["strategy"] for r in results):
            tp = sum(1 for r in results if r["strategy"] == strat and r["clarification_needed"] and r["clarification_predicted"])
            fp = sum(1 for r in results if r["strategy"] == strat and not r["clarification_needed"] and r["clarification_predicted"])
            fn = sum(1 for r in results if r["strategy"] == strat and r["clarification_needed"] and not r["clarification_predicted"])
            
            final[f"{strat}_clarification_precision"] = tp / (tp + fp) if tp + fp > 0 else 0.0
            final[f"{strat}_clarification_recall"] = tp / (tp + fn) if tp + fn > 0 else 0.0

        return final
