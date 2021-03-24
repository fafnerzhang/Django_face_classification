class List():
    def __init__(self):
        self.image_list = [[2,34,1,4],[1,4,532,2]]
list_1 = [100,100]
list_2 = [[1,3,2,4,1],[2,3,4,1,2]]
final = [[100,[1,4,2,1]],[100,[2,3,4,1]]]

fin = []
for i in range(2):
    fin.append([list_1[i],list_2[i]])
print(fin)

for i in final:
    print(i[0])
    for oc in i[1]:
        print(oc)
