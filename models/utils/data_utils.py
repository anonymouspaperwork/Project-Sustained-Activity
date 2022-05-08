from statistics import mean, median
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import orjson as json
import multiprocessing as mp
import math

namelist=['coreallcmt','coreallpr','corealliss','corepronum','coreactpronum','coreoveryearnum','coreover2yearnum','corefoll',
            'corecmtmean','corepr','coreiss','coreisscommenter','coreprcommenter','corecmtcommenter','coreissevter','corefolling','corestarpronum','coreorgnum','corehascomp',
            'contributorallcmt','contributorallpr','contributoralliss','contributorpronum','contributoractpronum','contributoroveryearnum', 'contributorover2yearnum','contributorfoll',
            'interesterallcmt','interesterallpr','interesteralliss','interesterpronum','interesteractpronum', 'interesteroveryearnum','interesterover2yearnum','interesterfoll',
            'contributorcmt','contributorpr','contributoriss','contributorisscommenter','contributorprcommenter','contributorcmtcommenter','contributorissevter','contributorfolling','contributorstarpronum','contributororgnum','contributorhascomp',
            'interesteriss','interesterisscommenter','interesterprcommenter','interestercmtcommenter','interesterissevter','interesterfolling','interesterstarpronum','interesterorgnum','interesterhascomp',
            
            'cmtpeople','majorcmt','outcmt',
            'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall','mergedprnum','proclspr','propr','crtclsissnum','numissue', 'isscommenternum', 'prcommenternum','cmtcommenternum','issevternum',
            'corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','daycmtnum','cmtratio','eachdmean','eachdmedian','eachdstd','merget',
            'fork','numwat','type',
            'openiss','openissratio','labelnum','bugnum', 'enhancementnum', 'helpwantednum','gfinum'
        ]

pd.set_option('display.max_columns',1000)
def unflod(df,s):
    df=df[s].values
    lst=[]
    for i in df:
        dic={}
        for j in range(len(i)):
            dic[j]=i[j]
        lst.append(dic)
    return pd.DataFrame(lst)

def getmean(l):
    return np.mean(l)

def getmedian(l):
    return np.median(l)

def getvar(l):
    return np.var(l)

def getstd(l):
    return np.std(l)

def avelist(alist):
    lsnum=len(alist)
    if lsnum==0:
        return [0,0,0]
    else:
        c = []
        for i in range(len(alist[0])):
            a=0
            for t in range(lsnum):
                a+=alist[t][i]
            c.append(a/lsnum)
        return c

def addcore(proid,ind):
    global revisedf
    newdf=revisedf[revisedf['project_id']==proid]
    corelist=newdf.iloc[0][1]
    corelist=avelist(corelist)
    return [corelist,ind]

def addctr(proid,ind):
    global revisedf
    newdf=revisedf[revisedf['project_id']==proid]
    ctrlist=newdf.iloc[0][2]
    ctrlist=avelist(ctrlist)
    return [ctrlist,ind]

def changecorecmt(proid,ind):
    global revisedf
    newdf=revisedf[revisedf['project_id']==proid]
    corelist=newdf.iloc[0][1]
    corecmt=sum([i[0] for i in corelist])
    return [corecmt,ind]

def changeleftcmt(proid,ind):
    global revisedf
    newdf=revisedf[revisedf['project_id']==proid]
    ctrlist=newdf.iloc[0][2]
    leftcmt=sum([i[0] for i in ctrlist])
    return [leftcmt,ind]

def checkact(cmtt,rtn):
    event_observed=1
    collectT=1000*datetime(2021, 3, 6, 23, 59, 59, 999999).timestamp()
    if cmtt!=0:
        if (collectT-min(cmtt))>730*86400000:
            month=24
        else:
            month=math.ceil((collectT-min(cmtt))/(86400000*30.42))
        eachmo=[]
        onecmt=[d for d in cmtt if (collectT-d)/86400000<730]
        for m in range(month):
            cmtnum=0
            for ind in range(len(onecmt)):
                if (collectT-onecmt[ind])/86400000>=m*30.42 and (collectT-onecmt[ind])/86400000<m*30.42+30.42:
                    cmtnum+=1
                    if cmtnum==3:
                        break
            eachmo.append(cmtnum)
        if median(eachmo)>=1:
            event_observed=0
            
    return [event_observed,rtn]

def checklife(a,b,c,rtn):
    add=0
    if a!=0:
        onecmt=a
        startt=b
        onecmt=[d for d in onecmt if (d-startt)/86400000<=730]
        if onecmt!=0 and c>730:
            eachmo=[]
            onecmt=[d for d in onecmt if (d-startt)/86400000<=730]
            for m in range(24):
                cmtnum=0
                for ind in range(len(onecmt)):
                    det=onecmt[ind]-startt
                    if det/86400000>=m*30.42 and det/86400000<m*30.42+30.42:
                        cmtnum+=1
                        if cmtnum==2:
                            break
                eachmo.append(cmtnum)
            if median(eachmo)>=1:
                add=1
    return [add,rtn]

def takelast(elem):
    return elem[-1]

def evtob(df):
    cmttlist=df["cmtt"].values
    checkdata=[[cmttlist[i],i] for i in range(len(cmttlist))]
    with mp.Pool(mp.cpu_count() // 2) as pool:
        res = pool.starmap(checkact, checkdata)
        res.sort(key=takelast)
        res=[i[0] for i in res]
    return pd.DataFrame(res)

def strictlife(df):
    cmtt=df["cmtt"].values
    lifeday=df["lifeday"].values
    startT=df["startT"].values
    checkdata=[[cmtt[i],startT[i],lifeday[i],i] for i in range(len(cmtt))]
    with mp.Pool(mp.cpu_count() // 2) as pool:
        res = pool.starmap(checklife, checkdata)
        res.sort(key=takelast)
        res=[i[0] for i in res]
    print(sum(res),len(res))
    return pd.DataFrame(res)

def load_raw_data(file_path,dataset_index):
    lst=[]
    for path in file_path:
        with open(path) as f:
            js = json.loads(f.read())
        data=js[str(dataset_index)]
        for i in range(len(data)):
            lst.append(data[str(i)])

    lst_ndarray = np.array(lst)
    rs=np.load('/data/projectdata/randlist.npy', allow_pickle=True)
    rs=rs.tolist()
    lst=list(lst_ndarray[rs])


    df=pd.DataFrame(lst)
    df=df[df['lifeday']>=90]
    
    
    df.insert(0,'oneyear',strictlife(df))

    twoyear=1000*datetime(2021, 3, 6, 23, 59, 59, 999999).timestamp()-730*86400000
    df=df[df['startT']<twoyear]
    year2012=1000*datetime(2011, 12, 31, 23, 59, 59, 999999).timestamp()
    df=df[df['startT']>year2012]
    year2022=1000*datetime(2022, 1, 1, 0, 0, 0, 0).timestamp()
    df=df[df['endT']<year2022]
    print(len(df))

    df=df.sort_values(by=['startT'],ascending=True)
    print(df['oneyear'].value_counts())
    return df
            
         

def load_train_test_data(df,fold,m):
    global revisedf
    lst=[]
    with open('/data/projectdata/ght_revise.json') as f:
        js = json.loads(f.read())
    data=js['0']
    for i in range(len(data)):
        lst.append(data[str(i)])
    revisedf=pd.DataFrame(lst)

    proidlist=df['project_id'].values.tolist()
    projectidlist=[[proidlist[i],i] for i in range(len(proidlist))]
    df.reset_index(drop=True,inplace=True)
    with mp.Pool(mp.cpu_count() // 2) as pool:
        res=pool.starmap(changecorecmt, projectidlist)
        res.sort(key=takelast)
        res=[i[0] for i in res]
        df['corecmt']=pd.DataFrame(res)
    df.reset_index(drop=True,inplace=True)
    with mp.Pool(mp.cpu_count() // 2) as pool:
        res=pool.starmap(changeleftcmt, projectidlist)
        res.sort(key=takelast)
        res=[i[0] for i in res]
        df['leftcmt']=pd.DataFrame(res)


    if m in ["Logistic_regression","Random_forest","Complete_XGB"]:
        pro=df[['corecap',
        'corewill',
        "contributorcap","interestercap",
        'contributorwill','interesterwill',

        'cmtpeople','clsisst','majorcmt','outcmt',#cap
        'type','fork','numwat',#opp
        'gfinum','bugnum','enhancementnum','helpwantednum','labelnum','openiss','openissratio',
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall', 'cmtratio',  'numissue','proclspr','propr','eachdaycmt','daycmtnum','cmtcommenternum','crtclsissnum','isscommenternum','issevternum','prcommenternum','corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','merget','mergedprnum',#will
        "oneyear",'project_id'
    ]]
    elif m=="only_corecap":
        pro=df[['corecap',"oneyear",'project_id'#owner
        ]]
    elif m=="only_corewill":
        pro=df[['corewill',"oneyear",'project_id'
        ]]
    elif m=="only_other_cap":
        pro=df[["contributorcap","interestercap","oneyear",'project_id']]
    elif m=="only_other_will":
        pro=df[['contributorwill','interesterwill',"oneyear",'project_id']]
    
    elif m=="only_pro_contri":
        pro=df[['cmtpeople','majorcmt','outcmt',
            'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall','mergedprnum','proclspr','propr','crtclsissnum','numissue', 'isscommenternum', 'prcommenternum','cmtcommenternum','issevternum',
            'corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','daycmtnum','cmtratio','eachdaycmt','merget',"oneyear",'project_id']]
    elif m=="only_pro_new":
        pro=df[['fork','numwat','type',
            'openiss','openissratio','labelnum','bugnum', 'enhancementnum', 'helpwantednum','gfinum',"oneyear",'project_id']]

        
    elif m=="ab_corecap":
        pro=df[[
        'corewill',
        "contributorcap","interestercap",
        'gfinum','bugnum','enhancementnum','helpwantednum','labelnum','openiss','openissratio',
        'contributorwill','interesterwill',

        'cmtpeople','clsisst','majorcmt','outcmt',#cap
        'type','fork','numwat',#opp
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall', 'cmtratio',  'numissue','proclspr','propr','eachdaycmt','daycmtnum','cmtcommenternum','crtclsissnum','isscommenternum','issevternum','prcommenternum','corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','merget','mergedprnum',#will
        "oneyear",'project_id'
        ]]
    elif m=="ab_corewill":
        pro=df[['corecap',
        
        "contributorcap","interestercap",
        'gfinum','bugnum','enhancementnum','helpwantednum','labelnum','openiss','openissratio',
        'contributorwill','interesterwill',

        'cmtpeople','clsisst','majorcmt','outcmt',#cap
        'type','fork','numwat',#opp
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall', 'cmtratio',  'numissue','proclspr','propr','eachdaycmt','daycmtnum','cmtcommenternum','crtclsissnum','isscommenternum','issevternum','prcommenternum','corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','merget','mergedprnum',#will
        "oneyear",'project_id'
        ]]
    elif m=="ab_other_cap":
        pro=df[['corecap',
        'corewill',
        
        'gfinum','bugnum','enhancementnum','helpwantednum','labelnum','openiss','openissratio',
        'contributorwill','interesterwill',

        'cmtpeople','clsisst','majorcmt','outcmt',#cap
        'type','fork','numwat',#opp
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall', 'cmtratio',  'numissue','proclspr','propr','eachdaycmt','daycmtnum','cmtcommenternum','crtclsissnum','isscommenternum','issevternum','prcommenternum','corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','merget','mergedprnum',#will
        "oneyear",'project_id']]
    elif m=="ab_other_will":
        pro=df[['corecap',
        'corewill',
        "contributorcap","interestercap",
        'gfinum','bugnum','enhancementnum','helpwantednum','labelnum','openiss','openissratio',
        

        'cmtpeople','clsisst','majorcmt','outcmt',#cap
        'type','fork','numwat',#opp
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall', 'cmtratio',  'numissue','proclspr','propr','eachdaycmt','daycmtnum','cmtcommenternum','crtclsissnum','isscommenternum','issevternum','prcommenternum','corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','merget','mergedprnum',#will
        "oneyear",'project_id']]
    
    elif m=="ab_all_cap":
        pro=df[['corecap',
        'corewill',
        "contributorcap","interestercap",
        'gfinum','bugnum','enhancementnum','helpwantednum','labelnum','openiss','openissratio',
        'contributorwill','interesterwill',

        
        'type','fork','numwat',#opp
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall', 'cmtratio',  'numissue','proclspr','propr','eachdaycmt','daycmtnum','cmtcommenternum','crtclsissnum','isscommenternum','issevternum','prcommenternum','corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','merget','mergedprnum',#will
        "oneyear",'project_id']]
    elif m=="ab_pro_contri":
        pro=df[['corecap',
        'corewill',
        "contributorcap","interestercap",
        'contributorwill','interesterwill',
        'fork','numwat','type',
        'openiss','openissratio','labelnum','bugnum', 'enhancementnum', 'helpwantednum','gfinum',
        "oneyear",'project_id']]
    elif m=="ab_pro_new":
        pro=df[['corecap',
        'corewill',
        "contributorcap","interestercap",
        'contributorwill','interesterwill',
        'cmtpeople','majorcmt','outcmt',
        'nrpter','nisscommter','nissevter','nprcrter','nprcommter','ncmter','ncmtcommter','cmtnumall','mergedprnum','proclspr','propr','crtclsissnum','numissue', 'isscommenternum', 'prcommenternum','cmtcommenternum','issevternum',
        'corecmt','leftcmt','actday','frontcmt','endcmt','ratiocmt','daycmtnum','cmtratio','eachdaycmt','merget',
        "oneyear",'project_id']]

    c=list(pro.columns)
    pro.reset_index(drop=True,inplace=True)
    if 'eachdaycmt' in c:
        pro.insert(0,'eachdmean',pro["eachdaycmt"].apply(getmean))
        pro.insert(0,'eachdmedian',pro["eachdaycmt"].apply(getmedian))
        pro.insert(0,'eachdstd',pro["eachdaycmt"].apply(getstd))
        del pro['eachdaycmt']
    T=pd.concat([pro],axis=1)
    c=list(pro.columns)
    T.columns=c
    T.reset_index(drop=True,inplace=True)
    if "corecap" in c:
        corecap=unflod(T,"corecap")
        del T['corecap']
        c=list(T.columns)
        T=pd.concat([T,corecap],axis=1)
        c+=['corepronum','coreallcmt','corefoll','corealliss','coreallpr','coreactpronum','coreoveryearnum','coreover2yearnum']
        T.columns=c

    if "corewill" in c:
        corewill=unflod(T,"corewill")
        del T['corewill']
        c=list(T.columns)
        T=pd.concat([T,corewill],axis=1)
        c+=['corecmtmean','corepr','coreiss','corefolling','corehascomp','corestarpronum','coreorgnum','coreisscommenter','coreissevter','coreprcommenter','corecmtcommenter']
        T.columns=c
        with mp.Pool(mp.cpu_count() // 2) as pool:
            res=pool.starmap(addcore, projectidlist)
            res.sort(key=takelast)
            res=[i[0] for i in res]

            T["corecmtmean"]=pd.DataFrame([i[0] for i in res])
            T["corepr"]=pd.DataFrame([i[1] for i in res])
            T["coreiss"]=pd.DataFrame([i[2] for i in res])

    if "contributorcap" in c:
        contributorcap=unflod(T,"contributorcap")
        interestercap=unflod(T,"interestercap")
        del T['contributorcap']
        del T['interestercap']
        c=list(T.columns)
        T=pd.concat([T,contributorcap],axis=1)
        T=pd.concat([T,interestercap],axis=1)
        c+=['contributorpronum','contributorallcmt','contributorfoll','contributoralliss','contributorallpr','contributoractpronum','contributoroveryearnum','contributorover2yearnum']
        c+=['interesterpronum','interesterallcmt','interesterfoll','interesteralliss','interesterallpr','interesteractpronum','interesteroveryearnum','interesterover2yearnum']
        T.columns=c

    if "contributorwill" in c:
        contributorwill=unflod(T,"contributorwill")
        interesterwill=unflod(T,"interesterwill")
        del T['contributorwill']
        del T['interesterwill']
        c=list(T.columns)
        T=pd.concat([T,contributorwill],axis=1)
        T=pd.concat([T,interesterwill],axis=1)
        c+=['contributorcmt','contributorpr','contributoriss','contributorfolling','contributorhascomp','contributorstarpronum','contributororgnum','contributorisscommenter','contributorissevter','contributorprcommenter','contributorcmtcommenter']
        c+=['interesteriss','interesterfolling','interesterhascomp','interesterstarpronum','interesterorgnum','interesterisscommenter','interesterissevter','interesterprcommenter','interestercmtcommenter']
        T.columns=c
        with mp.Pool(mp.cpu_count() // 2) as pool:
            res=pool.starmap(addctr, projectidlist)
            res.sort(key=takelast)
            res=[i[0] for i in res]
        
            T["contributorcmt"]=pd.DataFrame([i[0] for i in res])
            T["contributorpr"]=pd.DataFrame([i[1] for i in res])
            T["contributoriss"]=pd.DataFrame([i[2] for i in res])
        
    p_train_split1=int((fold/10)*T.shape[0])
    p_train_split2=int((fold/10+0.1)*T.shape[0])
    train_data1=T.iloc[:p_train_split1]
    train_data2=T.iloc[p_train_split2:]
    train_data=pd.concat([train_data1,train_data2],axis=0)
    test_data=T.iloc[p_train_split1:p_train_split2]

    
    p_train = train_data[train_data.oneyear == 1]

    n_train = train_data[train_data.oneyear == 0]

    train_data=pd.concat([p_train,n_train],ignore_index=True)
    
    y_train=train_data['oneyear']
    y_test=test_data['oneyear']

    del train_data['oneyear']
    del test_data['oneyear']
    X_train=train_data
    X_test=test_data

    return X_train, X_test, y_train, y_test