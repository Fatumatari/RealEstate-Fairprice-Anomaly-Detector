#Ask the user their name
from datetime import date
name = input("What's your name? ").strip().capitalize()

print("My name is ", name)
print('Hello, ', name)
your_birth_year = int(input("What's your year of birth? "))
today_date = int(input("which year is this? "))
print("So you age is", int(today_date)-your_birth_year,  "years old")
father_name = input("what's the first name of your father? ").strip().capitalize()
print("His name is,", father_name)
#calculator
#int
x = int(input("x " )) 
y = int(input("y " ))
z = x + y 
print(z)
#float
x1 = float(input("x1 " )) 
y1 = float(input("y1 " )) 
z1 = x1 + y1
print(z1)
#rounding off
x2 = float(input("x2 " )) 
y2 = float(input("y2 " )) 
z2 = (x2 + y2)
print(f"{z:2f}")
#conditions
a = int(input("a" ))
b = int(input("b" ))
if x < y:
    print ("x is less than y")
elif x > y:
    print("x is less greater than y")
elif x == y:
    print("x is equal to y")
