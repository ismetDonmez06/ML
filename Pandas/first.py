import pandas as pd
serieas =pd.Series([1,2,3,4])
print(serieas)
#tip sorugus
print(type(serieas))
#index bilgisi
print(serieas.axes)
#boyut
print(serieas.ndim)
#uzunluk
print(serieas.size)
# Array formatında erişme
print(serieas.values)
#ilk 3 deger
print(serieas.head(3))
#son 3 eleman
print(serieas.tail)
#index isimlendirme
Seri =pd.Series([99,88,77,44],index=[1,3,4,7])
#eleman erisme
print(Seri[0:2])
#sozluk üzerinden sözluk olusturma
sozluk = pd.Series({"reg":10,"log":11,"cart":12})
print(sozluk)
#iki seriyi birleştirerek seri olusur
print(pd.concat([Seri,Seri]))

#ELEMAN İSLEMLERİ
import numpy as np
A = np.array([1,3,44,78,444])
seri =pd.Series(A)
print(seri)

#eleman seçme işlemleri
print(seri[0])
print(seri[0:3])

#indeks özellilkleri

seri = pd.Series([121,45,56,98,78],index=["reg","lof","sec","pic","kic"])
print(seri.index)
print(seri.keys())
print(seri.items())
print(seri.values)

#eleman sorgulama
print("reg" in seri)

#fancy eleman secme
print(seri[["reg","lof"]])
seri["reg"] = 150
print(seri["reg":"pic"])

#Pandas Dataframe
l = [1,2,45,86,78]
df =pd.DataFrame(l,columns=["degisken_isim"])
print(df)
m = np.arange(1,10).reshape((3,3))
df = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df)

#df isimlendirme
df.columns = ("deg1","deg2","deg3")
print(df)
print(type(df))
print(df.axes)
print(df.shape)
print(df.ndim)
print(df.size)
print(df.values)
print(df.head(3))
print(df.tail(3))

#eleman işlemleri

s1 = np.random.randint(10,size = 5)
s2 = np.random.randint(10,size = 5)
s3 = np.random.randint(10,size = 5)

soz = {"var1 ":s1,"var2":s2,"var3":s3 }

df = pd.DataFrame(soz)
print(df)
print(df[0:1])
df.index=["a","b","c","d","e"]
print(df["c":"e"])
#silme
df=df.drop("a",axis=0)
print(df)

#fancy
l=["c","e"]
df =df.drop(l,axis=0)
print(df)


#degiskenler icin
soz = {"var1":s1,"var2":s2,"var3":s3 }
df = pd.DataFrame(soz)
print("var1" in df)
l = ["var1","var4","var2"]
for i in l:
    print(i in df)
df["var4"] =df["var1"]*df["var3"]
print(df)
#değişken silme
df = df.drop("var4",axis=1)
print(df)

#Gozlem ve Değişken seçimi
m =np.random.randint(1,30,size=(10,3))
df = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df)

#loc tanımlandığıo şekli ile seçim yapmak için kullanılır
print(df.loc[0:3])
print(df.loc[:3,"var3"])

#iloc alışık olduğumuz indeksleme mantığı ile seçim yapılır
print(df.iloc[0:3])
print(df.iloc[:3,:2])
#print(df.loc[:3,"var3"]) hata verir
print(df.iloc[:3]["var3"])

# Koşullu eleman işlemleri

print(df["var1"])
print(df["var1"][0:2])
print(df[:2][["var1","var2"]])

print(df[df.var1>15])

print(df[(df.var1>15) & (df.var3<5)])
print(df[(df.var1>15)][["var1","var2"]])


#join işlemli

m =np.random.randint(1,30,size=(5,3))
df = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df)

m =np.random.randint(1,30,size=(5,3))
df2 = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df2)

print(pd.concat([df,df2],ignore_index=True))
print(df.columns)

#farklı değişkenler varsa birleşecek verilerde
df2.columns = ["var1","var2","deg3"]
print(df2)

print(pd.concat([df,df2],join="inner"))#kesişimlere göre birleştirdi


#ileri birleştirme işlemleri

df2 = pd.DataFrame({'calisanlar':["ayse","ali","veli","fatma"],
                    "ilk_giris ":[2010,2009,2014,2019]})
df1 = pd.DataFrame({'calisanlar':["ali","veli","ayse","fatma"],
                    "grup ":["muhasebe","muhendis","muhendis","ik"]})

print(pd.merge(df1,df2))
print(pd.merge(df1,df2,on ='calisanlar'))

#coktan teke

df3 =pd.merge(df1,df2)

df4 = pd.DataFrame({"grup":["Muhasebe","muhendsilik","ik"],
                    "mudur":["caner","mustafa","berkcan"]})

#print(pd.merge(df3,df4))


#toplullaştırma ve gruplaştırma

import seaborn as sns

df =sns.load_dataset("planets")
print(df.head())

print(df.shape)
print(df.mean())
print(df["mass"].mean())
print(df["mass"].count())
print(df["mass"].min())
print(df["mass"].max())
print(df["mass"].std())
print(df["mass"].var())
print(df.describe().T)
#eksilk verileri sildi
print(df.dropna())


#Gruplama
df = pd.DataFrame({"gruplar" : ["a","b","c","a","b","c"],
                   "veri" :[10,11,52,23,43,55]},columns=["gruplar","veri"]
                  )
print(df)
print(df.groupby("gruplar").mean())
print(df.groupby("gruplar").sum())


df =sns.load_dataset("planets")
print(df.head())

print(df.groupby("method")["orbital_period"].mean())

print(df.groupby("method")["mass"].mean())


print(df.groupby("method")["orbital_period"].describe())


#ileri toplulaştırma işlemleri

df = pd.DataFrame({"gruplar" : ["a","b","c","a","b","c"],
                   "degisken1" :[10,11,52,23,43,55],
                   "degisken2" :[1,13,24,28,42,70]
                   },columns=["gruplar","degisken1","degisken2"]
                )

print(df)

#aggregate istedğimiz değerleri hesaplama
print(df.groupby("gruplar").mean())
agg=df.groupby("gruplar").aggregate([min,np.median,max])
print(agg)
agg=df.groupby("gruplar").aggregate({"degisken1":min ,"degisken2":max})
print(agg)

#filter

def filter_func(x):
    return x["degisken1"].std()>9
df.groupby("gruplar").filter(filter_func)

#transform her bir değişkendeki elemanları istediğimiz şekilde düzeltme
df_a = df.iloc[:,1:]
#print(df_a.transform(lambda x: (x-x.mean())/x.std()))

#apply

df = pd.DataFrame({"gruplar" : ["a","b","c","a","b","c"],
                   "degisken1" :[10,11,52,23,43,55],
                   "degisken2" :[1,13,24,28,42,70]
                   },columns=["gruplar","degisken1","degisken2"]
                )
print(df)
print(df.groupby("gruplar").apply(np.sum))

#pivot tabloloar
titanic =sns.load_dataset("titanic")

print(titanic.groupby(["sex","class"])[["survived"]].mean().unstack())

#pivot ile table

print(titanic.pivot_table("survived",index="sex",columns="class"))


#Dış Kaynaklı Veri Okuma

"""
csv pd.read_csv(adres , sep = ";" ) sep aralarındaki işaret ayraç ; yada ,

txt pd.read_csv(adres)  ayraç boşluktur 

excel
pd.read_excel()


 

"""


