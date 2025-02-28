lista=[]
while True:
    print("Unesite broj ili 'Done' za završetak unosa:")
    unos = input()
    if unos.lower() == 'done':
        break
    try:
        broj = float(unos)
        lista.append(broj)
    except ValueError:
        print("POGREŠAN UNOS!")
if lista:
    print("Unijeli ste ", len(lista)," brojeva")
    print("Srednja vrijednost: ", sum(lista)/len(lista))
    print("Minimalna vrijednost: ", min(lista))
    print("Maksimalna vrijednost: ", max(lista))
    lista.sort()
    print("Sortirana lista: ", lista)
else:
    print("Niste unijeli niti jedan broj")