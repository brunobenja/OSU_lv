""" Primjer 1.1 """
"""
x=23
print(x)
x=x+7
print(x) 
"""

""" Primjer 1.2 """
""" x=23
y=x>10
print(y) """

""" Primjer 1.3 """
""" x=23
if x<10:
    print("x je manji od 10")
else:
    print("x je veci ili jednak 10") """

""" Primjer 1.4 """
""" i=5
while i>0: 
    print(i)
    i=i-1
print("Gotovo!")

for i in range(5):
    print(i) """

""" Primjer 1.5 """
""" 
#listEmpty=[]
#listFriend=['Marko','Ivan','Ana']
#listFriend.append('Jasna')
#print(listFriend[0]) #Ispisuje se Marko
#print(listFriend[0:1:2]) #Ispisuje sve elemente od 0 do 1 u koraku od 2
#print(listFriend[:2]) #Ispisuje sve elemente do 2
#print(listFriend[1:]) #Ispisuje sve elemente od 1 do kraja
#print(listFriend[1:3]) #Ispisuje sve elemente 
"""

""" 
a=[1,2,3]
b=[4,5,6]
c=a+b
print(c)
print(max(c))
c[0]=7
c.pop() #izbacuje zadnji element
print(c)
for number in c:
    print("List Number: ",number)
print("List Length: ",len(c))
print("Done") 
"""

""" Primjer 1.6 """
""" 
fruit = "banana"
index = 0
count = 0
while index < len ( fruit ) :
    letter = fruit [index]
    if letter == 'a':
        count = count + 1
    print ( letter )
    index = index + 1
print ( count )
print ( fruit [ 0 : 3 ] ) #ispisuje od 0 do 3
print ( fruit [ 0 : ] ) #ispisuje od 0 do kraja
print ( fruit [ 2 : 6 : 1 ] )   #ispisuje od 2 do 6 u koraku 1
print ( fruit [ 0 : - 1 ] ) #ispisuje od 0 do -1 (krece brojati od kraja)
 """

""" 
line='Dobrodosli u nas grad'
if(line.startswith('Dobrodosli')):
    print('Prva rijec je Dobrodosli')
elif(line.startswith('dobrodosli')):
    print('Prva rijec je dobrodosli')
line.lower()
print (line)
data='From: pero@yahoo.com'
atpos=data.find('@') #trazi poziciju znaka @
print (atpos)
"""

"""Primjer 1.7"""
""" 
letters = ('a', 'b', 'c', 'd', 'e')
numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
mixed = (1, 'Hello', 3.14)
print(letters[0])
print(letters[1:4])
for letter in letters:
    print(letter) 
"""

"""Primjer 1.8"""
""" 
hr_num = { 'jedan': 1, 'dva': 2, 'tri': 3 }
print(hr_num)
print(hr_num['dva'])
hr_num['cetiri'] = 4
print(hr_num)
"""

"""Primjer 1.9"""
""" 
import random
import math
for i in range(10):
    x = random.random()
    y = math.sin(x)
    print('Broj:', x, 'Sin(broj):', y)
 """

"""Primjer 1.10"""
"""
def print_hello () :
    print ( " Hello world " )
print_hello ()
"""

"""Primjer 1.11"""
""" 
fhand = open('example.txt')
for line in fhand:
    line = line.rstrip()
    print(line)
    words = line.split()
fhand.close() 
"""

