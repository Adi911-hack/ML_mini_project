import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'colatron_original' / 'submit_code' / 'training'

def texts_and_specs():
    entries = []
    for line in (SOURCE / 'training.txt').read_text(encoding='utf-8').splitlines():
        parts = line.split('|', 3)
        if len(parts) != 4:
            continue
        mel_path = SOURCE / parts[1]
        if mel_path.exists():
            entries.append((parts[3], np.load(mel_path).astype(np.float32)))
    return entries

def encode(text, width):
    arr = np.array([ord(c) / 127 for c in text.lower()[:width]], dtype=np.float32)
    return np.pad(arr, (0, width - len(arr)))

def run(mode, stride, examples, iterations):
    start = time.perf_counter()
    entries = texts_and_specs()
    if len(entries) < 2:
        raise RuntimeError('Need two source mel spectrograms')
    width = max(len(t) for t, _ in entries)
    X = np.stack([encode(t, width) for t, _ in entries])
    max_frames = max(s.shape[0] for _, s in entries)
    n_steps = (max_frames + stride - 1) // stride
    targets = np.stack([np.pad(s[::stride], ((0, n_steps-len(s[::stride])),(0,0))) for _,s in entries])
    models=[]
    for i in range(n_steps):
        m=MultiOutputRegressor(SVR(kernel='poly', gamma='scale', C=1.0, epsilon=0.2))
        m.fit(X, targets[:,i,:])
        models.append(m)
    train_time=time.perf_counter()-start
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    with open(ROOT/'custom_sentences.csv',newline='',encoding='utf-8') as f:
        custom=list(csv.DictReader(f))
    if not custom:
        raise ValueError('Custom dataset contains no text')
    test = [('original_1', entries[0][0]), ('original_2', entries[1][0])]
    test += [(f'custom_{i+1}',row['text']) for i,row in enumerate(custom)]
    records=[]
    for name,text in test:
        t0=time.perf_counter()
        feat=encode(text,width).reshape(1,-1)
        pred=np.stack([m.predict(feat)[0] for m in models]).astype(np.float32)
        if mode=='trim':
            length = min(entries,key=lambda e:abs(len(e[0])-len(text)))[1].shape[0]
            pred=pred[:max(1, (length+stride-1)//stride)]
        mel=np.clip(pred.T,0,1)
        wave=librosa.feature.inverse.mel_to_audio(np.maximum(1e-6, librosa.db_to_power(mel*100-100+20)), sr=20000, n_fft=2048, hop_length=250, win_length=1000, n_iter=iterations, fmin=0, fmax=10000)
        peak=np.max(np.abs(wave))
        if peak>0: wave=0.85*wave/peak
        path=out/f'{mode}_{name}.wav'
        sf.write(path,wave,20000)
        records.append({'name':name,'mode':mode,'text':text,'mel_frames':int(mel.shape[1]),'audio_seconds':round(len(wave)/20000,3),'processing_seconds':round(time.perf_counter()-t0,3),'wav':str(path.relative_to(ROOT)),'predicted_mel_mean':round(float(np.mean(mel)),5)})
    result={'method':'adapted per-frame multioutput polynomial SVR using 2 supplied mel examples','mode':mode,'training_samples':len(entries),'mel_bins':80,'stride':stride,'train_seconds':round(train_time,3),'results':records,'caveat':'No original checkpoint or complete linear spectrograms; this is not a reproduction of published speech quality. No independent paired-audio accuracy evaluation.'}
    (out/f'{mode}_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['baseline','trim'],default='baseline')
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--iterations',type=int,default=12)
    a=p.parse_args()
    run(a.mode,a.stride,None,a.iterations)
