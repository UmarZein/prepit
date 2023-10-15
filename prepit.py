__version__ = '0.1.0'
import time
import random 
import importlib.util

HAS_NUMPY = importlib.util.find_spec('numpy')
HAS_PANDAS = importlib.util.find_spec('pandas')
HAS_SKLEARN = importlib.util.find_spec('sklearn')
HAS_LGBM = importlib.util.find_spec('lightgbm')
HAS_TQDM = importlib.util.find_spec('tqdm')
GPU_SUPPORT=False

class Score:
    def __init__(self, score, duration):
        self.score=score 
        self.duration=duration

class BRANCH:
    def __init__(self):
        pass
    def record(self, active):
        return []
    def record_name(self,prefix=''):
        return ['__']

def get_record(something, active):
    if isinstance(something,BRANCH):
        return something.record(active)
    else:
        return [int(active)]

def get_name(something, prefix=''):
    if isinstance(something,BRANCH):
        return something.record_name(prefix)
    else:
        return [prefix+something.__name__]

class Sequence(BRANCH):
    def __init__(self, funcs, prefix=""):
        self.funcs=funcs
        self.prefix=prefix
    def __call__(self, x):
        for f in self.funcs:
            x=f(x)
        return x
    def record(self, active):
        r=[]
        for f in self.funcs:
            r+=get_record(f,active)# if self.selected is not None else get_record(f,False)
        return r
    def record_name(self,prefix=''):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_name(f,prefix=prefix+type(self).__name__+f'-{i}__')# if self.selected is not None else get_record(f,False)
        return r

class Maybe(BRANCH):
    def __init__(self, func, p=.5, prefix=""):
        self.func=func
        self.p=p
        self.yes=False
        self.prefix=prefix
    def __call__(self, x):
        self.yes = random.random()<self.p
        if self.yes:
            x=self.func(x)
        return x
    def record(self, active):
        return get_record(self.func,active and self.yes)
    def record_name(self,prefix=''):
        return get_name(self.func,prefix=prefix+type(self).__name__+'__')

class OneOf(BRANCH):
    def __init__(self, funcs, prefix=""):
        self.funcs=funcs
        self.selected=None
        self.prefix=prefix
    def __call__(self, x):
        self.selected=random.choice(range(len(self.funcs)))
        return self.funcs[self.selected](x)
    def record(self, active):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_record(f,i==self.selected if active and self.selected is not None else False)
        return r
    def record_name(self,prefix=''):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_name(f,prefix=prefix+type(self).__name__+f'-{i}__')
        return r

class Until(BRANCH):
    def __init__(self, funcs, prefix=""):
        self.funcs=funcs
        self.selected=None
        self.prefix=prefix
    def __call__(self, x):
        self.selected=random.choice(range(len(self.funcs)))
        for f in self.funcs[:self.selected+1]:
            x=f(x)
        return x
    def record(self, active):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_record(f,i<=self.selected if active and self.selected is not None else False) 
        return r
    def record_name(self,prefix=''):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_name(f,prefix=prefix+type(self).__name__+f'-{i}__')
        return r

class SomeProb(BRANCH):
    def __init__(self, funcs, p=0.5, prefix=""):
        self.funcs=funcs
        self.p=p
        self.selected=None
        self.prefix=prefix
    def __call__(self, x):
        self.selected=[random.random() for _ in range(len(self.funcs))]
        for i,f in enumerate(self.funcs):
            if self.selected[i]<self.p:
                x=f(x)
        return x
    def record(self, active):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_record(f,self.selected[i]<self.p if active and self.selected is not None else False)# if self.selected is not None else get_record(f,False)
        return r
    def record_name(self,prefix=''):
        r=[]
        for i,f in enumerate(self.funcs):
            r+=get_name(f,prefix=prefix+type(self).__name__+f'-{i}__')# if self.selected is not None else get_record(f,False)
        return r

if HAS_LGBM:
    #you ought to have numpy at this point
    import numpy as np#pyright: ignore[reportMissingImports] 
    import lightgbm #pyright: ignore[reportMissingImports]

    try:
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        train_data = lightgbm.Dataset(data, label=label)
        params = {'num_iterations': 1, 'device': 'gpu', 'verbose':-1,}
        lightgbm.train(params, train_set=train_data)
        GPU_SUPPORT=True
    except Exception as _:
        GPU_SUPPORT=False

if HAS_PANDAS and HAS_SKLEARN:
     
    import numpy as np#pyright: ignore[reportMissingImports]
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier#pyright: ignore[reportMissingImports]
    from sklearn.metrics import f1_score, mean_squared_error#pyright: ignore[reportMissingImports]
    import pandas as pd#pyright: ignore[reportMissingImports]
    def default_evaluator(df: pd.DataFrame):
        global yv, yt
        global preds
        train_index=df.index<int(len(df)*.8)
        train_index2=df.index>int(len(df)*.2)
        Xt=df.iloc[train_index,:-1]
        yt=df.iloc[train_index,-1]
        Xt2=df.iloc[train_index2,:-1]
        yt2=df.iloc[train_index2,-1]
        Xv=df.iloc[~train_index,:-1]
        yv=df.iloc[~train_index,-1]
        Xv2=df.iloc[~train_index2,:-1]
        yv2=df.iloc[~train_index2,-1]
        categorical=False
        if yt.nunique()<len(yt)**.5:
            categorical=True
        model=RandomForestClassifier() if categorical else\
              RandomForestRegressor()
        model2=RandomForestClassifier() if categorical else\
              RandomForestRegressor()
        if HAS_LGBM:
            from lightgbm import LGBMClassifier,LGBMRegressor #pyright: ignore[reportMissingImports]

            if GPU_SUPPORT:
                model=LGBMClassifier(device='gpu', verbose=-1) if categorical else\
                      LGBMRegressor(device='gpu', verbose=-1)
                model2=LGBMClassifier(device='gpu', verbose=-1) if categorical else\
                      LGBMRegressor(device='gpu', verbose=-1)
            else:
                model=LGBMClassifier(verbose=-1) if categorical else\
                      LGBMRegressor(verbose=-1)
                model2=LGBMClassifier(verbose=-1) if categorical else\
                      LGBMRegressor(verbose=-1)

        start=time.time()
        model.fit(Xt,yt)
        preds=model.predict(Xv)
        duration=time.time()-start
        model2.fit(Xt2,yt2)
        preds=model.predict(Xv)
        preds2=model2.predict(Xv2)
        score=f1_score(preds,yv,average='micro') if categorical else mean_squared_error(preds,yv)
        score2=f1_score(preds2,yv2,average='micro') if categorical else mean_squared_error(preds2,yv2)
        return Score(
            score=(score*score2)**.5,
            duration=duration
        )
    def preprocessing_scores(data: pd.DataFrame, pipeline, evaluator=default_evaluator, min_samples=0):
        records=[].copy()
        feature_records=[].copy()
        iterator=range(min_samples)
        if HAS_TQDM:
            import tqdm#pyright: ignore[reportMissingModuleSource]
            iterator = tqdm.tqdm(iterator)
        for _ in iterator:
            record=[].copy()
            df=pipeline(data.copy())
            record=pipeline.record(True)
            if record in feature_records:
                continue
            feature_records.append(record.copy())
            score=evaluator(df)
            record.append(score.duration)
            record.append(score.score)
            records.append(record)
        return pd.DataFrame(np.array(records), columns=get_name(pipeline)+['duration','score'])




