import numpy as np
a=np.array([6,2,9])
""" print(type(a))          #tip niza
print(a.shape)          #dimenzije niza
print(a[0],a[1],a[2])   #pristup elementima niza """
a[1]=5                  #promjena elementa niza
""" print(a)                #ispis niza
print(a[1:2])           #ispisuje 1 element niza
print(a[1:-1])          #ispisuje od 1 do predzadnjeg elementa """

""" b=np.array([[1,2,3],[4,5,6]])
print(b.shape)              #dimenzije niza
print(b)                    #ispis niza
print(b[0,2],b[0,1],b[1,1]) #ispis elemenata na 0r,2s; 0r,1s; 1r,1s
print(b[0:2,0:1])           #ispisuje retke 0 i 1 u stupcu 0
print(b[:,0])               #ispisuje sve retke u stupcu 0  """


c=np.zeros((4,2))
d=np.ones((3,2))
e=np.full((1,2),5)
f=np.eye(2)
g=np.array([1,2,3],np.float32)  #niz sa float elementima
duljina=len(g)
print(duljina)
h=g.tolist()            #pretvara niz u listu
print(h)
c=g.transpose()        #transponira niz
print(g)
print(c)
h=np.concatenate((a,g))  #spaja nizove
print(h)