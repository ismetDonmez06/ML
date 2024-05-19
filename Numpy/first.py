
import numpy as np

python_list = [1,2,3,4,5,6]

#numpy array

arraynump = np.array([1,2,3,4,5,6])
print(arraynump)
print("********************************")
print("********************************")

#dizi boyutu
numpy_array1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(numpy_array1.ndim)
numpy_array2 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
print(numpy_array2.ndim)
print("********************************")
print("********************************")
#dizinin satır ve sutun sayısı
numpy_array1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(numpy_array1.shape)
print("********************************")
print(numpy_array1.ndim)
print("********************************")
print("********************************")
#Dizinin satır ve sütun sayısını değiştirmek : ndarray.reshape()




numpy_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(numpy_array.reshape(1,10))
print("********************************")

print(numpy_array.reshape(5,2))
print("********************************")
"""
- np.arange()
np.arange() →Python’daki range() fonksiyonuna benzer. 
Belirtilen başlangıç değerinden başlayıp, her seferinde adım sayısı kadar arttırarak ,bitiş değerine kadar olan sayıları 
bulunduran bir numpy dizisi dödürür.

Not: Bitiş değerinin diziye dahil edilmediğine dikkat edelim.

Genel kullanım :

np.arange(başlangıç,bitiş,adım sayısı)

"""


numpy_Array = np.array([0,1,2,3,4,5,6,7,8,9])
numpy_Array = numpy_array.reshape(5,2)
print(numpy_Array)
print("********************************")
print("********************************")
#Dizinin herhangi bir satırını seçmek

first_Row=numpy_Array[0]
first_Rowandsecondrow = numpy_Array[0:2]
print(first_Row)
print(first_Rowandsecondrow)
print("********************************")
print("********************************")
#Dizinin herhangi bir kolonunu seçmek
first_column = numpy_Array[:,0]
first_and_second_column = numpy_Array[:,0:2]
print(first_column)
print(first_and_second_column)
print("********************************")
print("********************************")


# Diziyi ters çevirmek
print(numpy_Array[::-1])

print("********************************")
print("********************************")
#0 matrisi oluşturmak : np.zeros()

zeros = np.zeros((5,4))
print(zeros)

print("********************************")
print("********************************")

#0 matrisi oluşturmak : np.zeros()

ones = np.ones((3,3,3))
print(ones)

print("********************************")
print("********************************")
#- Birim matris oluşturmak : np.eye()
#np.eye() → Belirlenen boyutlarda birim matris oluşturmamızı sağlayan fonksiyon.

print(np.eye(4))

print("********************************")
print("********************************")

#matrisleri birleşitirmek


numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
numpy_array = numpy_array.reshape(5,2)
#Satır bazlı birleştirme
print(np.concatenate([numpy_array, numpy_array], axis =0))

#Sütun bazlı birleştirme
print(np.concatenate([numpy_array, numpy_array], axis =1))
print("********************************")
print("********************************")


#-sum(toplam), max ve min değerlerini hesaplamak
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
numpy_array = numpy_array.reshape(5,2)
print(numpy_array)
print(numpy_array.max())
print(numpy_array.min())
print(numpy_array.sum())
#Satırların toplamı
print(numpy_array.sum(axis = 1))
#Sütunların toplamı
print(numpy_array.sum(axis = 0))
print("********************************")
print("********************************")

# mean, median, varyans ve standart sapma hesaplamak
numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
print(numpy_array.mean())
print(np.median(numpy_array))
print(numpy_array.var())
print(numpy_array.std())

numpy_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
numpy_array = numpy_array.reshape(3,3)
#dizimizi 3x3 lük, 2 boyutlu bir matrise dönüştürdük.
print("********************************")
print("********************************")

print(numpy_array)

print(numpy_array + numpy_array)

print(numpy_array - numpy_array)

print(numpy_array * numpy_array)

print(numpy_array / numpy_array)

print(numpy_array + 5)

print(numpy_array * 2)

#-Matrisin transpozu
#Matrisin satırlarını sütun, sütunlarını satır yapma işlemi numpy ile “ .T” yazmak kadar kolay.


numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
numpy_array = numpy_array.reshape(5,2)

print(numpy_array.T)


#)Numpy’da Koşul İfadeleri ile Çalışmak

numpy_array = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
boolean_array = numpy_array >= 5
print(boolean_array)

print(numpy_array[numpy_array<5])

