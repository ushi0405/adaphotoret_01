import json
import os
from pathlib import Path
from AdaphotoRet_run import search_photos

def normalize_path(path: str) -> str:
    """将路径标准化：统一正斜杠、小写、去除扩展名"""
    path = path.replace('\\', '/').lower()
    base, _ = os.path.splitext(path)
    return base

def precision_at_k(retrieved_paths, relevant_paths, k):
    retrieved_norm = [normalize_path(p) for p in retrieved_paths[:k]]
    relevant_norm = [normalize_path(p) for p in relevant_paths]
    hits = sum(1 for p in retrieved_norm if p in relevant_norm)
    return hits / k

def reciprocal_rank(retrieved_paths, relevant_paths):
    retrieved_norm = [normalize_path(p) for p in retrieved_paths]
    relevant_norm = [normalize_path(p) for p in relevant_paths]
    for i, p in enumerate(retrieved_norm, 1):
        if p in relevant_norm:
            return 1 / i
    return 0.0

def evaluate_all(test_queries_dict, top_k):
    results = {}
    for query, relevant in test_queries_dict.items():
        print(f"正在评估查询: {query} (P@{top_k})")
        top1, top2, top3, report, table, top_results = search_photos(query, include_explanation=False, allow_environment=False)
        retrieved = [r["img_path"] for r in top_results[:top_k]]
        p_k = precision_at_k(retrieved, relevant, top_k)
        mrr = reciprocal_rank(retrieved, relevant)
        results[query] = {
            f"precision_at_{top_k}": p_k,
            "mrr": mrr,
            "retrieved": retrieved,
            "relevant": relevant
        }
        print(f"  P@{top_k}={p_k:.2f}, MRR={mrr:.2f}")
        print(f"  检索到的前{top_k}张: {retrieved}")
        print()
    if not results:
        return {}
    avg_p = sum(v[f"precision_at_{top_k}"] for v in results.values()) / len(results)
    avg_mrr = sum(v["mrr"] for v in results.values()) / len(results)
    print(f"平均 P@{top_k}: {avg_p:.2f}")
    print(f"平均 MRR: {avg_mrr:.2f}")
    return results

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    query_file = base / "test_queries.json"
    if not query_file.is_file():
        raise SystemExit("缺少 test_queries.json；请准备本地查询与相关图片标注后再评估。")
    with query_file.open("r", encoding="utf-8") as f:
        test_queries = json.load(f)

    # 计算 P@3 并保存为 eval_results.json
    results_p3 = evaluate_all(test_queries, top_k=3)
    with open(base / "eval_results.json", "w", encoding="utf-8") as f:
        output_p3 = {q: {
            "precision_at_3": res["precision_at_3"],
            "mrr": res["mrr"],
            "retrieved": res["retrieved"],
            "relevant": res["relevant"]
        } for q, res in results_p3.items()}
        json.dump(output_p3, f, indent=2, ensure_ascii=False)
    print("评估结果已保存到 eval_results.json (P@3)")

    # 计算 P@1 并保存为 eval_results_p1.json
    results_p1 = evaluate_all(test_queries, top_k=1)
    with open(base / "eval_results_p1.json", "w", encoding="utf-8") as f:
        output_p1 = {q: {
            "precision_at_1": res["precision_at_1"],
            "mrr": res["mrr"],
            "retrieved": res["retrieved"],
            "relevant": res["relevant"]
        } for q, res in results_p1.items()}
        json.dump(output_p1, f, indent=2, ensure_ascii=False)
    print("评估结果已保存到 eval_results_p1.json (P@1)")