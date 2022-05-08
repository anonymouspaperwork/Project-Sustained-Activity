import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp


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
    '\#cmt\_pro\_core','\#cmt\_pro\_peripheral','\#cmt\_actday','\#cmt\_front','\#cmt\_end','cmt\_ratio','\#cmt\_first','cmt\_first\_ratio','\#cmt\_mean','\#cmt\_median','cmt\_std','\#merge\_t',

    ###Influx of newcomers
    #Attention of newcomers
    '\#fork','\#star','type',
    #Opportunity of newcomers
    '\#iss\_open','iss\_open\_ratio','\#label','\#bug','\#enhancement','\#help-wanted','\#GFI',

]

def takelast(elem):
    return elem[-1]
    
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
                if m[1]<0:
                    lilst[0].append(i)
                break
    return [lilst,n]

if __name__=="__main__":
    df=pd.read_pickle("/data/projectdata/X_test.pkl")
    y_test=pd.read_pickle("/data/projectdata/y_test.pkl")
    df.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)
    df['oneyear']=y_test

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
    ind=0
    ftlist=['\#pr\_all\_core','\#pro\_oneyear\_core','\#peripheral','\#fork']
    ftlists=['#pr_all_core','#pro_oneyear_core','#peripheral','#fork']



    for i in range(len(namelist)):
        if featurelist[i]!=ftlist[ind]:
            continue
        signlist=df[namelist[i]].values.tolist()
        typelist=df['type'].values.tolist()
        #typelist=df['cmtpeople'].values.tolist()

        signlist=np.array(signlist)
        typelist=np.array(typelist)
        neglist=list(signlist[lime_list[i][0]])
        poslist=list(signlist[lime_list[i][1]])
        typeneglist=list(typelist[lime_list[i][0]])
        typeposlist=list(typelist[lime_list[i][1]])
        a0,a1,a2,a3=[],[],[],[]
        for d in range(len(typeneglist)):
            if typeneglist[d]==0:
            #if typeneglist[d]==1:
                a0.append(neglist[d])
            else:
                a1.append(neglist[d])
        for d in range(len(typeposlist)):
            if typeposlist[d]==0:
            #if typeposlist[d]==1:
                a2.append(poslist[d])
            else:
                a3.append(poslist[d])
        pos=0
        alist=[[a0,a2],[a1,a3]]
        plt.figure(ind,figsize=(6, 6.5))
        position=[]
        for t in [0,1]:
            list0=alist[t][0]
            list1=alist[t][1]
            bplot=plt.boxplot([list0,list1],widths=0.8,
            positions=[pos,pos+1], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
            position.append(pos+0.5)
            pos+=3
            [[item.set_color('k') for item in bplot['medians']]]
            colors = ['cornflowerblue','coral']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
        plt.xlim((-1, 3*2-1))
        plt.ylabel(ftlists[ind],{'size':25})
        plt.xticks(position,["organization","user"])
        #plt.xticks(position,["#cmter=1","#cmter>1"])
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax = plt.gca()
        bwith = 2
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        #plt.xticks(rotation=270)
        plt.grid(alpha=0.6)
        plt.tight_layout()
        plt.savefig('comvartype'+str(ind)+'.png')
        #plt.savefig('comvarcmter'+str(ind)+'.png')
        ind+=1
        if ind==len(ftlist):
            break
