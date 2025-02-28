file=open('song.txt')
sadrzaj=file.read()
rjecnik={}
rijeci=sadrzaj.split()
for rijec in rijeci:
    if rijec in rjecnik:
        rjecnik[rijec]+=1
    else:
        rjecnik[rijec]=1

brojacJedinstvenih=0
for rijec in rjecnik:
    if rjecnik[rijec]==1:
        brojacJedinstvenih+=1


print("Broj jedinstvenih riječi: ",brojacJedinstvenih)
for rijec in rjecnik:
    if rjecnik[rijec]==1:
        print(rijec)
file.close()