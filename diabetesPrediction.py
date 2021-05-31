import tkinter as tk
from tkinter import *
from tkinter import font

import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_squared_error

# Verileri csv dosyasından programa aktarıyoruz
veriler=pd.read_csv("diabetes.csv")

# Hangi kolonda ne kadar null değer var kontrol ediyoruz

print(veriler.isnull().sum())

# Label encoding yapılacak verileri ayırıyoruz

leVeriler=veriler.iloc[:,1:17]

#Label Encoding

from sklearn import preprocessing

# Yes ve no şeklinde olan verileri daha yüksek verim elde etmek için 1 ve 0'a çeviriyoruz

df_col=list(leVeriler.columns)

for i in range(len(df_col)):
 leVeriler[df_col[i]] = preprocessing.LabelEncoder().fit_transform(leVeriler[df_col[i]])
 
# Yaş sütunuyla label encoding yaptığımız verileri birleştiriyoruz
sonVeriler=pd.concat([veriler[["Age"]],leVeriler.iloc[:,0:16]],axis=1)

# Verilerin korelasyonuna bakıyoruz

korelasyon=sonVeriler.corr()
#GİRİŞLER VE ÇIKIŞLAR

#Verileri girişler ve çıkışlar olarak ikiye ayırıyoruz
girisler=sonVeriler.iloc[:,0:-1]
cikislar=sonVeriler.iloc[:,16:17]

#TEST VE TRAİN AYIRMA

from sklearn.model_selection import train_test_split

# Girişler ve çıkışları %70 train, %30 test olacak şekilde ayırıyoruz

x_train, x_test, y_train, y_test=train_test_split(girisler,cikislar,test_size=0.30,random_state=0)


#RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Random Forest algoritmasının parametrelerini ayarlıyoruz

rfc=RandomForestClassifier(n_estimators=100,max_depth=5)

# Bu parametrelere göre algoritmayı verilere uyguluyoruz

rfc.fit(x_train,y_train)

# Algoritmaya test verilerini verip tahmin yaptırıyoruz

y_pred=rfc.predict(x_test)

# 0.5'den büyük olan değerleri 1 kabul ediyoruz

y_pred=(y_pred>0.5)

# Sonuçları kullanarak ACC, F1, KAPPA,RMSE değerlerini elde ediyoruz

rfcAcc=accuracy_score(y_test, y_pred)
rfcF1=f1_score(y_test, y_pred, average='macro')
rfcKappa=cohen_kappa_score(y_test, y_pred)
rfcMASE=mean_squared_error(y_test, y_pred)
print(rfcAcc)


master=Tk()

# Tasarım için canvas belirliyoruz

canvas=Canvas(master,height=900,width=900)
canvas.pack()

# Frame'leri belirliyoruz

frame_baslik=Frame(master)
frame_baslik.place(relx=0.1,rely=0,relwidth=0.9,relheight=0.1)

frame_yas=Frame(master,bg='#add8e6')
frame_yas.place(relx=0.1,rely=0.1,relwidth=0.3,relheight=0.1)

frame_ust=Frame(master,bg='#add8e6')
frame_ust.place(relx=0.1,rely=0.2,relwidth=0.3,relheight=0.7)

frame_sol=Frame(master,bg='#add8e6')
frame_sol.place(relx=0.5,rely=0.1,relwidth=0.4,relheight=0.8)

# Label, checkbox, ve inputları belirliyoruz.

titleLbl=Label(frame_baslik, text="Diyabet Riski Tahmin Uygulaması ", bg='#dddddd',font='Verdana 15 bold')
titleLbl.pack( side = LEFT, anchor=NW,pady=25,padx=25)
Lyas = Label(frame_yas, text="Yas: ", bg='#add8e6',font='Verdana 10 bold')
Lyas.pack( side = LEFT, anchor=NW,pady=25,padx=25)
yasVar=IntVar()
yas = Entry(frame_yas)
yas.pack(side=RIGHT, anchor=NW,pady=25,padx=25)

var1=IntVar()
C1=Checkbutton(frame_ust,text="Cinsiyet (0: erkek, 1: kadın)",variable=var1,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C1.pack(anchor=NW,pady=1,padx=25)

var2=IntVar()
C2=Checkbutton(frame_ust,text="Poliüri (fazla idrara çıkma)",variable=var2,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C2.pack(anchor=NW,pady=1,padx=25)

var3=IntVar()
C3=Checkbutton(frame_ust,text="Polidipsi (fazla su içme)",variable=var3,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C3.pack(anchor=NW,pady=1,padx=25)

var4=IntVar()
C4=Checkbutton(frame_ust,text="Ani Kilo Kaybı",variable=var4,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C4.pack(anchor=NW,pady=1,padx=25)

var5=IntVar()
C5=Checkbutton(frame_ust,text="Zayıflık",variable=var5,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C5.pack(anchor=NW,pady=1,padx=25)

var6=IntVar()
C6=Checkbutton(frame_ust,text="Polifaji (anormal açlık hissi)",variable=var6,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C6.pack(anchor=NW,pady=1,padx=25)

var7=IntVar()
C7=Checkbutton(frame_ust,text="Genital Mantar",variable=var7,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C7.pack(anchor=NW,pady=1,padx=25)

var8=IntVar()
C8=Checkbutton(frame_ust,text="Görme Bozukluğu",variable=var8,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C8.pack(anchor=NW,pady=1,padx=25)

var9=IntVar()
C9=Checkbutton(frame_ust,text="Kaşıntı",variable=var9,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C9.pack(anchor=NW,pady=1,padx=25)

var10=IntVar()
C10=Checkbutton(frame_ust,text="Sinir",variable=var10,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C10.pack(anchor=NW,pady=1,padx=25)

var11=IntVar()
C11=Checkbutton(frame_ust,text="Geç İyileşme",variable=var11,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C11.pack(anchor=NW,pady=1,padx=25)

var12=IntVar()
C12=Checkbutton(frame_ust,text="Kısmi Felç",variable=var12,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C12.pack(anchor=NW,pady=1,padx=25)

var13=IntVar()
C13=Checkbutton(frame_ust,text="Kas Sertliği",variable=var13,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C13.pack(anchor=NW,pady=1,padx=25)

var14=IntVar()
C14=Checkbutton(frame_ust,text="Alopesi (saç kıran)",variable=var14,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C14.pack(anchor=NW,pady=1,padx=25)

var15=IntVar()
C15=Checkbutton(frame_ust,text="Obezite",variable=var15,onvalue=1,offvalue=0,bg='#add8e6',font='Verdana 10 bold')
C15.pack(anchor=NW,pady=1,padx=25)
counter=0
def buttonClick():
    
    global testList
    global df
    global counter
    
    # Hangi sonucu gösterdiğini anlamamız için sayaç tanımlıyoruz

    counter=counter+1

    # Checkbox'ların değerlerini listeye aktarıyoruz.

    testList=[]
    testList.append(yas.get())
    testList.append(var1.get())
    testList.append(var2.get())
    testList.append(var3.get())
    testList.append(var4.get())
    testList.append(var5.get())
    testList.append(var6.get())
    testList.append(var7.get())
    testList.append(var8.get())
    testList.append(var9.get())
    testList.append(var10.get())
    testList.append(var11.get())
    testList.append(var12.get())
    testList.append(var13.get())
    testList.append(var14.get())
    testList.append(var15.get())
    
    # Oluşan listeyi modelimize vermek için DataFrame'e dönüştürüyoruz

    df = DataFrame (testList,columns=['Column_Name'])
    
    # Alt alta gelen verileri satır haline getirmek için transpose'unu alıyoruz
    
    df=df.transpose()

    # Kullanıcıdan alınan verileri modelimize tahmin ettirip sonucu alıyoruz

    sonuc=rfc.predict(df)
    print("Sonuc: "+ str(sonuc))
    if (sonuc==1):
        sonucText="Diyabet Riski Mevcut"
    else:
        sonucText="Diyabet Riski Mevcut Değil"

    # Sonucu yazdırıyoruz

    sonucLbl=Label(frame_sol,text=str(counter)+". Sonuc: "+ sonucText ,bg='#add8e6',font=('Verdana 10',15)).pack(padx=10,pady=10,anchor=NW)


btn = Button(frame_ust, text="Tahmin yap",font='Verdana 10 bold',command=buttonClick)
btn.pack(anchor=NW,pady=1,padx=25)



accuaryLbl=Label(frame_sol,text="Accuary: "+str(rfcAcc),bg='#add8e6',font=('Verdana 10',15)).pack(padx=10,pady=10,anchor=NW)
fmeasureLbl=Label(frame_sol,text="F-measure: "+str(rfcF1),bg='#add8e6',font=('Verdana 10',15)).pack(padx=10,pady=10,anchor=NW)
kappaLbl=Label(frame_sol,text="Kappa: "+str(rfcKappa),bg='#add8e6',font=('Verdana 10',15)).pack(padx=10,pady=10,anchor=NW)
RMSELbl=Label(frame_sol,text="RMSE: "+str(rfcMASE),bg='#add8e6',font=('Verdana 10',15)).pack(padx=10,pady=10,anchor=NW)







master.mainloop()
    
      

    
    



