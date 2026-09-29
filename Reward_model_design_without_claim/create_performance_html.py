"""Build a self-contained colored performance report from span predictions."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from retracted_reward.data import load_judged_data, split_by_answer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="claims_and_answers_judged_large.json")
    parser.add_argument("--source", default="Sample_parent_data/judge_answers_LQA.json")
    parser.add_argument("--predictions", default="runs_large/closed_book_predictions.json")
    parser.add_argument("--output", default="runs_large/closed_book_performance.html")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def marked_text(text: str, gold: list[dict], predicted: list[dict]) -> str:
    boundaries = {0, len(text)}
    for span in gold + predicted:
        boundaries.update((int(span["start"]), int(span["end"])))
    points = sorted(boundaries)
    parts = []
    for start, end in zip(points, points[1:]):
        segment = html.escape(text[start:end])
        in_gold = any(int(s["start"]) <= start and end <= int(s["end"]) for s in gold)
        in_pred = any(int(s["start"]) <= start and end <= int(s["end"]) for s in predicted)
        css = "overlap" if in_gold and in_pred else "missed" if in_gold else "extra" if in_pred else ""
        parts.append(f'<mark class="{css}">{segment}</mark>' if css else segment)
    return "".join(parts)


def load_context(gold_path: str | Path, source_path: str | Path) -> dict[str, dict[str, str]]:
    judged = json.loads(Path(gold_path).read_text(encoding="utf-8-sig"))
    source = Path(source_path)
    source_by_id = {}
    if source.exists():
        source_root = json.loads(source.read_text(encoding="utf-8-sig"))
        for record in source_root.get("records", []):
            record_id = str(record.get("anchor_csv_data", {}).get("Record ID", ""))
            if record_id:
                source_by_id[record_id] = record

    context = {}
    for row in judged.get("samples", []):
        paper_id = str(row.get("record_id", ""))
        source_record = source_by_id.get(paper_id, {})
        question = source_record.get("generated_question") or row.get("generated_question") or ""
        parent_claim = (
            source_record.get("question_target_claim")
            or row.get("question_target_claim")
            or (row.get("anchor_claims") or [{}])[0].get("text", "")
        )
        for answer_id in row.get("answers", {}):
            context[f"{paper_id}:{answer_id}"] = {
                "question": str(question),
                "parent_claim": str(parent_claim),
            }
    return context


def main() -> None:
    args = arguments()
    _, answers = load_judged_data(args.gold)
    gold_answers = split_by_answer(answers, args.seed)[args.split]
    def paragraph_id(answer) -> str:
        # Negative-augmented datasets can reuse an answer ID across sample types.
        # Their prediction files disambiguate those rows with the sample type.
        sample_type = getattr(answer, "sample_type", None)
        middle = f":{sample_type}" if sample_type else ""
        return f"{answer.paper_id}{middle}:{answer.answer_id}"

    gold = {paragraph_id(a): a for a in gold_answers}
    context = load_context(args.gold, args.source)
    root = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    predictions = {str(row["paragraph_id"]): row for row in root["predictions"]}

    rows = []
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    exact_tp = exact_fp = exact_fn = 0
    char_tp = char_fp = char_fn = 0
    for paragraph_id, answer in gold.items():
        prediction = predictions.get(paragraph_id, {"spans": []})
        gold_spans = list(answer.spans)
        predicted_spans = prediction.get("spans", [])
        gold_positive, predicted_positive = bool(gold_spans), bool(predicted_spans)
        status = "tp" if gold_positive and predicted_positive else "fp" if predicted_positive else "fn" if gold_positive else "tn"
        counts[status] += 1

        gold_exact = {(int(s["start"]), int(s["end"])) for s in gold_spans}
        pred_exact = {(int(s["start"]), int(s["end"])) for s in predicted_spans}
        exact_tp += len(gold_exact & pred_exact)
        exact_fp += len(pred_exact - gold_exact)
        exact_fn += len(gold_exact - pred_exact)
        gold_chars = {i for s in gold_spans for i in range(int(s["start"]), int(s["end"]))}
        pred_chars = {i for s in predicted_spans for i in range(int(s["start"]), int(s["end"]))}
        char_tp += len(gold_chars & pred_chars)
        char_fp += len(pred_chars - gold_chars)
        char_fn += len(gold_chars - pred_chars)
        rows.append({
            "id": paragraph_id,
            "status": status,
            "gold_count": len(gold_spans),
            "predicted_count": len(predicted_spans),
            "question": context.get(paragraph_id, context.get(f"{answer.paper_id}:{answer.answer_id}", {})).get("question", ""),
            "parent_claim": context.get(paragraph_id, context.get(f"{answer.paper_id}:{answer.answer_id}", {})).get("parent_claim", ""),
            "text": marked_text(answer.text, gold_spans, predicted_spans),
        })

    def f1(tp: int, fp: int, fn: int) -> float:
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    binary_f1 = f1(counts["tp"], counts["fp"], counts["fn"])
    exact_f1 = f1(exact_tp, exact_fp, exact_fn)
    char_f1 = f1(char_tp, char_fp, char_fn)
    cards = "\n".join(
        f'''<article class="example" data-status="{row['status']}" data-search="{html.escape((row['id'] + ' ' + row['question'] + ' ' + row['parent_claim']).lower(), quote=True)}">
  <header><strong>{html.escape(row['id'])}</strong><span class="status {row['status']}">{row['status'].upper()}</span></header>
  <div class="counts">Gold spans: {row['gold_count']} &middot; Predicted spans: {row['predicted_count']}</div>
  <dl class="context">
    <dt>Question</dt><dd>{html.escape(row['question'])}</dd>
    <dt>Parent claim</dt><dd>{html.escape(row['parent_claim'])}</dd>
  </dl>
  <p class="paragraph">{row['text']}</p>
</article>''' for row in rows)

    document = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-book detection performance</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8fa; --surface:#fff; --text:#17202a; --muted:#65717e; --line:#d9dee5; --green:#d5f5df; --green-text:#145a32; --red:#ffd9d6; --red-text:#8a1c15; --amber:#ffe9b8; --amber-text:#714b00; --blue:#dcecff; --blue-text:#164b82; }}
@media (prefers-color-scheme:dark) {{ :root{{--bg:#12161b;--surface:#1b2128;--text:#edf2f7;--muted:#aab4bf;--line:#39434d;--green:#164c32;--green-text:#c6f7d8;--red:#602621;--red-text:#ffd9d6;--amber:#5d4615;--amber-text:#ffedbd;--blue:#183e66;--blue-text:#d8eaff}} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ width:min(1120px,calc(100% - 32px)); margin:28px auto 60px }} h1 {{ margin:0; font-size:clamp(1.5rem,3vw,2.15rem); letter-spacing:0 }}
.subhead {{ color:var(--muted); margin:4px 0 20px }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; margin-bottom:18px }}
.metric,.example {{ background:var(--surface); border:1px solid var(--line); border-radius:8px }} .metric {{ padding:12px 14px }} .metric b {{ display:block; font-size:1.4rem }} .metric span,.counts {{ color:var(--muted); font-size:.88rem }}
.legend,.controls {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; margin:14px 0 }} .swatch {{ display:inline-flex; align-items:center; gap:6px }} .swatch i {{ width:18px; height:12px; border-radius:2px }}
.controls label {{ color:var(--muted) }} select,input {{ min-height:38px; border:1px solid var(--line); border-radius:6px; padding:6px 9px; background:var(--surface); color:var(--text); font:inherit }} input {{ flex:1; min-width:210px }}
#shown {{ margin-left:auto; color:var(--muted) }} .examples {{ display:grid; gap:12px }} .example {{ padding:14px 16px }} .example header {{ display:flex; justify-content:space-between; gap:12px; align-items:center }}
.status {{ border-radius:4px; padding:1px 7px; font-size:.78rem; font-weight:700 }} .status.tp {{ background:var(--green);color:var(--green-text) }} .status.tn {{ background:var(--blue);color:var(--blue-text) }} .status.fp {{ background:var(--amber);color:var(--amber-text) }} .status.fn {{ background:var(--red);color:var(--red-text) }}
.context {{ display:grid; grid-template-columns:minmax(84px,max-content) 1fr; gap:4px 12px; margin:10px 0 0; padding-top:10px; border-top:1px solid var(--line) }} .context dt {{ color:var(--muted); font-weight:700 }} .context dd {{ margin:0; overflow-wrap:anywhere }}
.paragraph {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:12px 0 0 }} mark {{ border-radius:2px; padding:1px 0 }} mark.overlap {{ background:var(--green);color:var(--green-text) }} mark.missed {{ background:var(--red);color:var(--red-text); text-decoration:underline }} mark.extra {{ background:var(--amber);color:var(--amber-text); text-decoration:underline wavy }}
.empty {{ display:none; padding:30px 0; text-align:center; color:var(--muted) }}
@media(max-width:560px) {{ main{{width:min(100% - 20px,1120px);margin-top:16px}} .controls{{align-items:stretch}} .controls label{{width:100%}} #shown{{margin-left:0;width:100%}} .example header{{align-items:flex-start}} }}
</style>
</head>
<body><main>
<h1>Closed-book detection performance</h1>
<p class="subhead">Test split &middot; {len(rows)} paragraphs &middot; seed {args.seed}</p>
<section class="metrics" aria-label="Performance summary">
  <div class="metric"><b>{binary_f1:.3f}</b><span>Detection F1</span></div>
  <div class="metric"><b>{exact_f1:.3f}</b><span>Exact-span F1</span></div>
  <div class="metric"><b>{char_f1:.3f}</b><span>Character F1</span></div>
  <div class="metric"><b>{counts['tp']} / {counts['tn']}</b><span>True positive / negative</span></div>
  <div class="metric"><b>{counts['fp']} / {counts['fn']}</b><span>False positive / negative</span></div>
</section>
<div class="legend" aria-label="Text highlight legend">
  <span class="swatch"><i style="background:var(--green)"></i>Gold + predicted overlap</span>
  <span class="swatch"><i style="background:var(--red)"></i>Missed gold text</span>
  <span class="swatch"><i style="background:var(--amber)"></i>Extra predicted text</span>
</div>
<div class="controls">
  <label for="status">Result</label><select id="status"><option value="all">All results</option><option value="tp">True positives</option><option value="tn">True negatives</option><option value="fp">False positives</option><option value="fn">False negatives</option></select>
  <label for="search">Search</label><input id="search" type="search" placeholder="ID, question, or claim">
  <span id="shown" aria-live="polite"></span>
</div>
<section class="examples" id="examples">{cards}</section><p class="empty" id="empty">No matching examples.</p>
</main>
<script>
const statusFilter=document.getElementById('status'), search=document.getElementById('search'), examples=[...document.querySelectorAll('.example')], shown=document.getElementById('shown'), empty=document.getElementById('empty');
function filter(){{const state=statusFilter.value,query=search.value.trim().toLowerCase();let visible=0;examples.forEach(el=>{{const match=(state==='all'||el.dataset.status===state)&&el.dataset.search.includes(query);el.hidden=!match;if(match)visible++;}});shown.textContent=`${{visible}} of ${{examples.length}} shown`;empty.style.display=visible?'none':'block';}}
statusFilter.addEventListener('change',filter);search.addEventListener('input',filter);filter();
</script></body></html>'''
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
