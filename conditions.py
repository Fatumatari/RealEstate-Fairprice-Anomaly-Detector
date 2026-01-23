score = int(input("score: "))
if score >= 80 and score <= 100:
    print("Distinction")
elif score >= 60 and score <= 80:
    print("Credit")
elif score >= 50 and score <= 60:
    print("Pass")
else:
    print("Fail")

#parity
x = int(input("x: "))
if x % 2 == 0:
    print("Even")
else:
    print("Odd")