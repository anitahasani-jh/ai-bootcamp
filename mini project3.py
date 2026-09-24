num1 = int(input("Enter number 1: "))
num2 = int(input("Enter number 2: "))

print("1. Addition")
print("2. Subtraction")
print("3. Multiplication")
print("4. Division")

choice = int(input("Enter your choice: "))

if choice == 1:
    # جمع
 print("Addition:", num1 + num2)

elif choice == 2:
    # تفریق
 print("Subtraction:", num1 - num2)
 
elif choice == 3:
    # ضرب
 print("Multiplication:", num1 * num2)
 

elif choice == 4:
    # تقسیم
 print("Division:", num1 / num2)


else:
    print("Invalid choice")