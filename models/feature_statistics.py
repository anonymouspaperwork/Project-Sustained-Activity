from statistics import mean, median
from datetime import datetime
import orjson as json
import numpy as np
import pandas as pd
import multiprocessing as mp
from decimal import Decimal, ROUND_HALF_UP
import scipy.stats as stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
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

featurelist=[
    ###core
    #capacity
    '\#cmt\_all\_core','\#pr\_all\_core','\#issue\_all\_core','\#pro\_core','\#pro\_act\_core','\#pro\_oneyear\_core','\#pro\_twoyear\_core','\#follower\_core',
    #willingness
    '\#cmt\_core','\#pr\_core','\#issue\_core','\#iss\_comment\_core','\#pr\_comment\_core','\#cmt\_comment\_core','\#iss\_event\_core','\#following\_core','\#star\_pro\_core','\#org\_core','show\_comp\_core',

    ###Peripheral
    #contributor capacity
    '\#cmt\_all\_ctr','\#pr\_all\_ctr','\#issue\_all\_ctr','\#pro\_ctr','\#pro\_act\_ctr','\#pro\_oneyear\_ctr','\#pro\_twoyear\_ctr','\#follower\_ctr',
    #interester capacity
    '\#cmt\_all\_inq','\#pr\_all\_inq','\#issue\_all\_inq','\#pro\_inq','\#pro\_act\_inq','\#pro\_oneyear\_inq','\#pro\_twoyear\_inq','\#follower\_inq',

    #contributor willingness
    '\#cmt\_ctr','\#pr\_ctr','\#issue\_ctr','\#iss\_comment\_ctr','\#pr\_comment\_ctr','\#cmt\_comment\_ctr','\#iss\_event\_ctr','\#following\_ctr','\#star\_pro\_ctr','\#org\_ctr','show\_comp\_ctr',
    #interester willingness
    '\#issue\_inq','\#iss\_comment\_inq','\#pr\_comment\_inq','\#cmt\_comment\_inq','\#iss\_event\_inq','\#following\_inq','\#star\_pro\_inq','\#org\_inq','show\_comp\_inq',

    ###Steady contribution supply
    #Number of contributors
    '\#cmter','\#core','\#peripheral',
    #Contribution Types
    '\#reporter','\#iss\_commenter','\#iss\_eventer','\#pr\_opener','\#pr\_commenter','\#committer','\#cmt\_commenter','\#cmt\_pro','\#pr\_merged','\#pr\_closed','\#pr','\#iss\_cls','\#iss\_pro','\#iss\_comment\_pro','\#pr\_comment\_pro','\#cmt\_comment\_pro','\#iss\_event\_pro',
    #Contribution Patterns
    '\#cmt\_pro\_core','\#cmt\_pro\_peripheral','\#cmt\_actday','\#cmt\_front','\#cmt\_end','cmt\_ratio','\#cmt\_first','cmt\_first\_ratio','\#cmt\_mean','\#cmt\_median','cmt\_std','merge\_t',

    ###Influx of newcomers
    #Attention of newcomers
    '\#fork','\#star','type',
    #Opportunity of newcomers
    '\#iss\_open','iss\_open\_ratio','\#label','\#bug','\#enhancement','\#help-wanted','\#GFI',

]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

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

def strictlife(df):
    cmtt=df["cmtt"].values
    lifeday=df["lifeday"].values
    startT=df["startT"].values
    checkdata=[[cmtt[i],startT[i],lifeday[i],i] for i in range(len(cmtt))]
    with mp.Pool(mp.cpu_count() // 2) as pool:
        res = pool.starmap(checklife, checkdata)
        res.sort(key=takelast)
        res=[i[0] for i in res]
    return pd.DataFrame(res)

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

def load_raw_data(file_path,dataset_index):
    lst=[]
    for path in file_path:
        with open(path) as f:
            js = json.loads(f.read())
        data=js[str(dataset_index)]
        for i in range(len(data)):
            lst.append(data[str(i)])
            
    lst_ndarray = np.array(lst)
    rs=np.load('/data/LIMEdata/randlist.npy', allow_pickle=True)
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
    return df

def load_train_test_data(df):
    global revisedf
    lst=[]
    with open('/data/LIMEdata/ght_revise.json') as f:
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
    return T

def get_lime_list(n):
    lilst=[[],[]]
    for i in range(limedf.shape[0]):
        for m in limedf["lime_lst"].iloc[i]:
            ftstr=m[0]
            ftstr=ftstr.split( )
            for fts in ftstr:
                if fts=='>' or fts=='<' or fts=='<=' or is_number(fts):
                    continue
                else:
                    limeftname=fts
            if namelist[n]==limeftname:
                if m[1]>0:
                    lilst[1].append(i)
                if m[1]<=0:
                    lilst[0].append(i)
                break
    return [lilst,n]

df=pd.read_pickle("/data/projectdata/X_test.pkl")
y_test=pd.read_pickle("/data/projectdata/y_test.pkl")
df.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
df['oneyear']=y_test

df0=df[df['oneyear']==0]
df1=df[df['oneyear']==1]
data0=pd.read_pickle("/data/LIMEdata/limeres0.pkl")
data1=pd.read_pickle("/data/LIMEdata/limeres1.pkl")
data2=pd.read_pickle("/data/LIMEdata/limeres2.pkl")
data3=pd.read_pickle("/data/LIMEdata/limeres3.pkl")
data4=pd.read_pickle("/data/LIMEdata/limeres4.pkl")
data5=pd.read_pickle("/data/LIMEdata/limeres5.pkl")
data6=pd.read_pickle("/data/LIMEdata/limeres6.pkl")
data7=pd.read_pickle("/data/LIMEdata/limeres7.pkl")
data8=pd.read_pickle("/data/LIMEdata/limeres8.pkl")
data9=pd.read_pickle("/data/LIMEdata/limeres9.pkl")
data10=pd.read_pickle("/data/LIMEdata/limeres10.pkl")
data11=pd.read_pickle("/data/LIMEdata/limeres11.pkl")
data12=pd.read_pickle("/data/LIMEdata/limeres12.pkl")
data13=pd.read_pickle("/data/LIMEdata/limeres13.pkl")
data14=pd.read_pickle("/data/LIMEdata/limeres14.pkl")
data15=pd.read_pickle("/data/LIMEdata/limeres15.pkl")
data16=pd.read_pickle("/data/LIMEdata/limeres16.pkl")
data17=pd.read_pickle("/data/LIMEdata/limeres17.pkl")
data18=pd.read_pickle("/data/LIMEdata/limeres18.pkl")
data19=pd.read_pickle("/data/LIMEdata/limeres19.pkl")
limedf=pd.concat([data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19], axis=0)


namelistind=[[i] for i in range(len(namelist))]
with mp.Pool(mp.cpu_count() // 2) as pool:
    res=pool.starmap(get_lime_list, namelistind)
    res.sort(key=takelast)
    lime_list=[i[0] for i in res]

m=0
for i in range(len(namelist)):
    if featurelist[i]=='\#cmt\_all\_core':
        print('\\midrule')
        print('\\multicolumn{2}{l}{\\textbf{Core\_Developer}}\\\\')
        print('Capacity\\\\')
    if featurelist[i]=='\#cmt\_core':
        print('Willingness\\\\')
    if featurelist[i]=='\#cmt\_all\_ctr':
        print('\\midrule')
        print('\\multicolumn{2}{l}{\\textbf{Peripheral\_Developers}}\\\\')
        print('\\multicolumn{2}{l}{Contributor\_Capacity}\\\\')
    if featurelist[i]=='\#cmt\_ctr':
        print('\\multicolumn{2}{l}{Contributor\_Willingness}\\\\')
    if featurelist[i]=='\#cmt\_all\_inq':
        print('\\multicolumn{2}{l}{Inquisitive\_Capacity}\\\\')
    if featurelist[i]=='\#issue\_inq':
        print('\\multicolumn{2}{l}{Inquisitive\_Willingness}\\\\')
    if featurelist[i]=='\#cmter':
        print(' ')
        print('\\midrule')
        print('\\multicolumn{2}{l}{\\textbf{Steady\_Contribution\_Supply}}\\\\')
        print('\\multicolumn{2}{l}{Number\_of\_Contributors}\\\\')
    if featurelist[i]=='\#reporter':
        print('\\multicolumn{2}{l}{Contribution\_Types}\\\\')
    if featurelist[i]=='\#cmt\_actday':
        print('\\multicolumn{2}{l}{Contribution\_Patterns}\\\\')
    if featurelist[i]=='\#fork':
        print('\\midrule')
        print('\\multicolumn{2}{l}{\\textbf{Influx\_of\_Newcomers}}\\\\')
        print('\\multicolumn{2}{l}{Attention\_of\_Newcomers}\\\\')
    if featurelist[i]=='\#iss\_open':
        print('\\multicolumn{2}{l}{Opportunity\_of\_Newcomers}\\\\')
    data0=df0[namelist[i]].values.tolist()
    data1=df1[namelist[i]].values.tolist()
    signlist=df[namelist[i]].values.tolist()
    signlist=np.array(signlist)
    neglist=list(signlist[lime_list[i][0]])
    poslist=list(signlist[lime_list[i][1]])

    if featurelist[i] in ['\#pr\_ctr','\#pr\_core']:
        decimalstr="0.00"#0.00
    else:
        decimalstr="0.00"#0.0
    if stats.mannwhitneyu(data0,data1,alternative='two-sided')[1]<0.0005 and stats.mannwhitneyu(neglist,poslist,alternative='two-sided')[1]<0.0005:
        m+=1
        print("\SmallCode{"+featurelist[i]+"}&"+str(Decimal(float(median(data0))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(data0))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"&"+str(Decimal(float(median(data1))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(data1))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"$^*$&"
        +str(Decimal(float(median(neglist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(neglist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"&"+str(Decimal(float(median(poslist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(poslist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"$^*$\\\\"
        )
    if stats.mannwhitneyu(data0,data1,alternative='two-sided')[1]<0.0005 and stats.mannwhitneyu(neglist,poslist,alternative='two-sided')[1]>=0.0005:
        m+=1
        print("\SmallCode{"+featurelist[i]+"}&"+str(Decimal(float(median(data0))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(data0))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"&"+str(Decimal(float(median(data1))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(data1))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"$^*$&"
        +str(Decimal(float(median(neglist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(neglist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"&"+str(Decimal(float(median(poslist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(poslist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"\hspace{1.09mm} "+"\\\\"
        )
    if stats.mannwhitneyu(data0,data1,alternative='two-sided')[1]>=0.0005 and stats.mannwhitneyu(neglist,poslist,alternative='two-sided')[1]<0.0005:
        m+=1
        print("\SmallCode{"+featurelist[i]+"}&"+str(Decimal(float(median(data0))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(data0))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"&"+str(Decimal(float(median(data1))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(data1))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"\hspace{1.09mm} "+"&"
        +str(Decimal(float(median(neglist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(neglist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"&"+str(Decimal(float(median(poslist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"/"+str(Decimal(float(mean(poslist))).quantize(Decimal(decimalstr), rounding=ROUND_HALF_UP))+"$^*$\\\\"
        )
print(m)