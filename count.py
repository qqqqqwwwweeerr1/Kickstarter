import math

a = 30
sum = (a-1)*(a-2)*(a-3)/6*a/2

sum_lambda = lambda a : (a*5-1)*(a*5-2)*(a*5-3)/6*math.ceil(a/2)

loop_lambda = lambda a : loopa(a)

def loopa(a):

    for i in range(1, a):
        print(sum_lambda(i))
        i += 1



# print(sum_lambda(a))
loop_lambda(7)

print(math.ceil(1/2))

36220219930714  004X

0 1 2 3 4
400种
2000种