while True:
    n = int(input("what's n? "))
    if n  > 0:
        break
for x in range(n):
    print("Hello")

students = ["Kelvin", "Harry", "Brian"]
for student in students:
    print(student)

for i in range(len(students)):
    print(i+1, students[i])
