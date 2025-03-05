file=open('SMSSpamCollection.txt')
lines=file.readlines()
humCounter=0
humLength=0
spamCounter=0
spamLength=0
SMSwithExclamationMark=0
for line in lines:
    type, poruka = line.split('\t', 1)
    words = poruka.split()
    if type == 'ham':
        humCounter += 1
        humLength += len(words)
    elif type == 'spam':
        spamCounter += 1
        spamLength += len(words)
        if poruka.strip().endswith('!'):
            SMSwithExclamationMark += 1
print("a)")
print(f"Prosječan broj riječi u ham porukama: {round(humLength/humCounter,2)}")
print("Prosječan broj riječi u spam porukama: ",round(spamLength/spamCounter,2))
print("b)")
print("Broj spam poruka koje završavaju uskličnikom: ",SMSwithExclamationMark)