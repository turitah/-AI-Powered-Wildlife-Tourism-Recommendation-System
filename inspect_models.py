import pathlib
import pickle
import joblib
import numpy as np
import sklearn.preprocessing as sp

sp.dtype = np.dtype

p = pathlib.Path('Notebook/models')
for name in ['feature_columns.pkl', 'label_encoder.pkl', 'wildlife_detection_random_forest.pkl']:
    path = p / name
    print('===', name, '===')
    print('exists', path.exists(), 'size', path.stat().st_size if path.exists() else None)
    with open(path, 'rb') as f:
        header = f.read(32)
    print('header', header)
    for loader_name, loader in [('pickle', pickle.load), ('joblib', joblib.load)]:
        try:
            with open(path, 'rb') as f:
                obj = loader(f) if loader_name == 'pickle' else loader(path)
            print(loader_name, 'ok', type(obj))
            print('repr', repr(obj)[:500])
            if hasattr(obj, 'classes_'):
                print('classes', list(obj.classes_)[:10])
            if hasattr(obj, 'feature_importances_'):
                print('feature_importances', obj.feature_importances_[:10])
            if isinstance(obj, list):
                print('list len', len(obj), obj)
        except Exception as e:
            print(loader_name, 'err', type(e).__name__, e)
    print()
