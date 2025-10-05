#exceptions
while True:
    try:
        x = int(input("what's x? "))
    except ValueError:
        print("x is not a integer")
    else:
        break
print(f"x is {x}")