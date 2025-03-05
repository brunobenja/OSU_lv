print("Zadatak 1.4.1")
print("Unesite broj radnih sati:")
hours = float(input())
print("Unesite plaću po radnom satu:")
payPerHour = float(input())
#pay = hours * payPerHour
print("Radni sati:", hours)
print("eura/h:", payPerHour)
#print("Ukupno:", pay) 

def total_euro(hours,payPerHour):
    total_pay=hours*payPerHour
    return total_pay

print("Ukupno:",total_euro(hours,payPerHour))