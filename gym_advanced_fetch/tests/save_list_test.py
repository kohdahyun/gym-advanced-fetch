import csv

list = []
a = 1
b = 3

list.append(1.1)

list.append(2)

list.append(3.3)

if ((a<2) and (b < 10)):
    with open('save_list_test_1.csv','w') as file:
        write = csv.writer(file)
        write.writerow(list)