
ask = input("Enter a condition: ")
a = float(input("Enter a number: "))
b = float(input("Enter another number: "))

if ask == "+":
    print(a + b)
elif ask == "-":
    print(a - b)
elif ask == "*":
    print(a * b)
elif ask == "/":
    print(a / b)
else:
    print("Invalid input")
