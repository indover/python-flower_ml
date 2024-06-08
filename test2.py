from colorama import init
from colorama import Fore, Back, Style
init()

print(Fore.BLACK)
print(Back.GREEN + Style.BRIGHT)
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