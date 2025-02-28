print("Zadatak 1.4.2")
print("Unesite ocjenu između 0.0 i 1.0:")
try:
    score = float(input())
    if score < 0.0 or score > 1.0:
        print("Ocjena je izvan intervala [0.0,1.0]")
    else:
        if score >= 0.9:
            print("A")
        elif score >= 0.8:
            print("B")
        elif score >= 0.7:
            print("C")
        elif score >= 0.6:
            print("D")
        else:
            print("F")
except ValueError:
    print("Unesite valjani broj između 0.0 i 1.0")