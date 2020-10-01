import pandas as pd

df1 = pd.DataFrame([[1,2,3,4], [6,7,8,9]], columns=['D', 'B', 'E', 'A'], index=[1,2])
df2 = pd.DataFrame([[10,20,30,40], [60,70,80,90], [600,700,800,900]], columns=['A', 'B', 'C', 'D'], index=[2,3,4])

print(df1)
print(df2)

print("------------------------------------------------------------------------------------")

a1, a2 = df1.align(df2, join='outer', axis=1)
print(a1)
print(a2)

print("------------------------------------------------------------------------------------")

a1, a2 = df1.align(df2, join='outer', axis=None)
print(a1)
print(a2)

print("------------------------------------------------------------------------------------")

a1, a2 = df1.align(df2, join='right', axis=None)
print(a1)
print(a2)

print("------------------------------------------------------------------------------------")

a1, a2 = df1.align(df2, join='inner', axis=1)
print(a1)
print(a2)