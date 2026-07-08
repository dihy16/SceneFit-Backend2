"""
User study scoring algorithm.
Canonical location: app/services/study/scorer.py
(Backward-compat shim remains at app/services/method_scorer.py)

Computes per-method scores from participant responses using a two-stage metric:
  Stage 1 — MRR: mean reciprocal rank of per-method first-stage outfit selection.
  Stage 2 — WinRate: fraction of participants who chose this method as overall winner.
  Final score: alpha * MRR + (1 - alpha) * WinRate
"""
from typing import Any, Dict, List


def score_methods(
    methods: List[str],
    participant_responses: List[Dict[str, Any]],
    *,
    alpha: float = 0.6,
    num_outfits: int = 5,
) -> Dict[str, Any]:
    """
    Compute per-method scores from collected participant responses.

    Args:
        methods: Ordered list of method ids/names.
        participant_responses: List of response dicts in Unity or legacy format.
        alpha: Weight for Stage-1 MRR vs Stage-2 WinRate (0-1).
        num_outfits: Number of outfit slots shown per method.

    Returns:
        {
          "methods": { method_id: { scores... } },
          "summary": { "total_participants": int, "ranked_methods": [str, ...] }
        }
    """
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    total_participants = len(participant_responses)
    per_method: Dict[str, Dict] = {}
    for m in methods:
        per_method[m] = {
            'first_stage_counts': {},
            'view_counts': {i: 0 for i in range(num_outfits)},
            'final_choice_count': 0,
            'mrr_sum': 0.0,
        }

    for resp in participant_responses:
        method_choices = None
        views_map: Dict[str, List[int]] = {}

        if isinstance(resp, dict) and isinstance(resp.get('responses'), (list, tuple)):
            method_choices = {}
            for r in resp.get('responses', []):
                if not isinstance(r, dict):
                    continue
                mid = (
                    r.get('methodName') or r.get('method_name')
                    or r.get('methodId') or r.get('method_id')
                )
                if mid is None:
                    continue

                sel = None
                img_url = r.get('selectedURL') or r.get('imgURL') or r.get('img_url')
                response_img_urls = r.get('imgURLs') or r.get('imgPaths') or r.get('img_paths')
                if img_url and isinstance(response_img_urls, list) and img_url in response_img_urls:
                    sel = response_img_urls.index(img_url)

                if sel is None and img_url:
                    top_img_paths = resp.get('imgPaths')
                    if isinstance(top_img_paths, dict) and str(mid) in top_img_paths:
                        url_list = top_img_paths[str(mid)]
                        if isinstance(url_list, list) and img_url in url_list:
                            sel = url_list.index(img_url)

                if sel is None:
                    sel = (
                        r.get('selectedIndex') or r.get('selected_index')
                        or r.get('selectedRank') or r.get('selected_rank')
                        or r.get('selected')
                    )

                vcounts = r.get('viewCounts') or r.get('view_counts')
                if sel is None:
                    continue
                try:
                    sr = int(sel)
                except Exception:
                    continue

                if 0 <= sr < num_outfits:
                    method_choices[str(mid)] = sr
                    if isinstance(vcounts, (list, tuple)):
                        vals: List[int] = []
                        for x in list(vcounts)[:num_outfits]:
                            try:
                                vals.append(max(0, int(x)))
                            except Exception:
                                vals.append(0)
                        while len(vals) < num_outfits:
                            vals.append(0)
                        views_map[str(mid)] = vals

        final_choice_method = None
        if isinstance(resp, dict):
            final_choice_method = (
                resp.get('winnerMethodName') or resp.get('winner_method_name')
                or resp.get('finalWinnerMethodName') or resp.get('final_winner_method_name')
                or resp.get('finalWinnerMethodId') or resp.get('final_winner_method_id')
                or resp.get('final_choice_method') or resp.get('final_method') or resp.get('final_choice')
            )

        if method_choices is None:
            method_choices = resp.get('method_choices')

        choices_map: Dict[str, int] = {}
        if isinstance(method_choices, dict):
            for k, v in method_choices.items():
                try:
                    choices_map[str(k)] = int(v)
                except Exception:
                    continue
        elif isinstance(method_choices, (list, tuple)):
            for idx, v in enumerate(method_choices):
                if idx >= len(methods):
                    break
                try:
                    choices_map[methods[idx]] = int(v)
                except Exception:
                    continue
        else:
            continue

        for m, chosen_idx in choices_map.items():
            if m not in per_method:
                continue
            counts = per_method[m]['first_stage_counts']
            counts[chosen_idx] = counts.get(chosen_idx, 0) + 1
            if m in views_map:
                for i, c in enumerate(views_map[m][:num_outfits]):
                    per_method[m]['view_counts'][i] = per_method[m]['view_counts'].get(i, 0) + int(c)
            try:
                if 0 <= int(chosen_idx) < num_outfits:
                    per_method[m]['mrr_sum'] += 1.0 / (int(chosen_idx) + 1)
            except Exception:
                pass

        if final_choice_method and final_choice_method in per_method:
            per_method[final_choice_method]['final_choice_count'] += 1

    out: Dict[str, Any] = {'methods': {}, 'summary': {'total_participants': total_participants}}
    ranked = []
    for m in methods:
        entry = per_method[m]
        first_counts = entry['first_stage_counts']
        first_props = {
            str(k): (v / total_participants if total_participants > 0 else 0.0)
            for k, v in first_counts.items()
        }
        final_k = entry['final_choice_count']
        winrate = final_k / total_participants if total_participants > 0 else 0.0
        mrr = entry.get('mrr_sum', 0.0) / total_participants if total_participants > 0 else 0.0
        final_score = alpha * mrr + (1.0 - alpha) * winrate
        out['methods'][m] = {
            'first_stage_counts': {str(k): v for k, v in sorted(first_counts.items())},
            'first_stage_proportions': first_props,
            'view_counts': {str(k): v for k, v in sorted(entry['view_counts'].items())},
            'view_rate': (sum(entry['view_counts'].values()) / total_participants) if total_participants > 0 else 0.0,
            'stage1_mrr': mrr,
            'final_choice_count': final_k,
            'stage2_winrate': winrate,
            'final_score': final_score,
            'alpha': alpha,
        }
        ranked.append((m, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    out['summary']['ranked_methods'] = [r[0] for r in ranked]
    return out
