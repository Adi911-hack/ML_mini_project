import csv
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'recording_evaluation.csv').open(encoding='utf-8', newline='')))
texts = [r['assumed_transcript'] for r in rows]
reference = np.array([float(r['reference_duration_s']) for r in rows])
baseline = np.array([float(r['synthesized_duration_s']) for r in rows])
lengths = np.array([len(t) for t in texts], dtype=float)
predicted = []
for i in range(len(rows)):
    train = np.arange(len(rows)) != i
    model = Ridge(alpha=1000)
    model.fit(lengths[train].reshape(-1, 1), reference[train])
    predicted.append(float(model.predict([[lengths[i]]])[0]))
predicted = np.array(predicted)
report = []
for i, r in enumerate(rows):
    report.append({'id':r['id'], 'transcript':texts[i], 'reference_seconds':round(reference[i],3), 'baseline_seconds':round(baseline[i],3), 'calibrated_loo_seconds':round(predicted[i],3), 'baseline_abs_error_seconds':round(abs(baseline[i]-reference[i]),3), 'calibrated_abs_error_seconds':round(abs(predicted[i]-reference[i]),3)})
with (ROOT/'duration_comparison.csv').open('w', newline='', encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=report[0].keys());writer.writeheader();writer.writerows(report)
summary={'count':len(rows),'evaluation':'leave-one-out duration prediction (four recordings used to train, one held out each fold)','baseline_mae_seconds':round(float(np.mean(abs(baseline-reference))),3),'calibrated_mae_seconds':round(float(np.mean(abs(predicted-reference))),3),'relative_mae_reduction_percent':round(float(100*(1-np.mean(abs(predicted-reference))/np.mean(abs(baseline-reference)))),1),'limitations':'Only output duration improved. Speech intelligibility, content accuracy, MOS, and published Seq2Seq reproduction are not established. The duration estimator uses a tiny custom dataset.'}
(ROOT/'duration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
