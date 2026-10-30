import csv
import json
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent.parent

def main():
    rows=list(csv.DictReader((EVAL/'recording_evaluation.csv').open(encoding='utf-8',newline='')))
    lengths=np.array([len(r['assumed_transcript']) for r in rows],dtype=float).reshape(-1,1)
    durations=np.array([float(r['reference_duration_s']) for r in rows])
    model=Ridge(alpha=1000).fit(lengths,durations)
    out=ROOT/'results'
    out.mkdir(exist_ok=True)
    report=[]
    for row in rows:
        number=row['id']
        original=out/f'baseline_custom_{number}.wav'
        if not original.exists():
            raise FileNotFoundError(f'First run python run_svr.py --mode baseline: {original}')
        wave,sr=sf.read(original,dtype='float32')
        if wave.ndim!=1:
            wave=wave.mean(axis=1)
        target=max(.3,float(model.predict([[len(row['assumed_transcript'])]])[0]))
        rate=(len(wave)/sr)/target
        stretched=librosa.effects.time_stretch(wave,rate=rate)
        path=out/f'calibrated_custom_{number}.wav'
        sf.write(path,stretched,sr)
        result={'id':number,'target_seconds':round(target,3),'output_seconds':round(len(stretched)/sr,3),'file':str(path.relative_to(ROOT))}
        report.append(result)
        print(result,flush=True)
    (out/'calibrated_metrics.json').write_text(json.dumps({'method':'duration calibrated postprocessing using a ridge regression estimator and phase vocoder time stretching','caution':'In-sample audio demonstration; predictive evaluation is reported separately via leave-one-out validation. This does not demonstrate improved intelligibility.','results':report},indent=2))
if __name__=='__main__':
    main()
