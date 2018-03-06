import math
import networkx
from random import random


reader = open("facebook_dataset.txt", 'r')
min_timestamp = 1157454929
max_timestamp = 1232576125
time_count = 10
time_len = math.floor((max_timestamp - min_timestamp) / time_count)
time_index = [[]] * time_count

data_set = []
for record in reader:
    new_record = [int(record.split()[0]), int(record.split()[1]), int(record.split()[2])]
    data_set.append(new_record)
reader.close()

for i in range(0, len(data_set)):
    record = data_set[i]
    index = math.floor((record[2] - min_timestamp) / time_len)
    if index != time_count:
        time_index[index].append(i)
    else:
        time_index[index - 1].append(i)

now_network = networkx.Graph()
for i in range(0, time_count):
    print(i)
    network_writer = open("networks/time_" + str(i + 1) + "_network.txt", 'w')
    for j in range(0, i + 1):
        for id in time_index[j]:
            network_writer.write(str(data_set[id][0]) + " " + str(data_set[id][1]) + "\n")
    network_writer.close()

    for id in time_index[i]:
        now_network.add_edge(data_set[id][0], data_set[id][1])

    if i != 0:
        links_writer = open("links/time_" + str(i + 1) + "_link.txt", 'w')
        count = 0
        for id in time_index[i]:
            links_writer.write(str(data_set[id][0]) + " " + str(data_set[id][1]) + " 1" + "\n")
            count += 1

        temp = 0
        while temp < count:
            node1 = math.floor(random() * 63731)
            node2 = math.floor(random() * 63731)
            if node1 != node2 and not now_network.has_edge(node1, node2):
                links_writer.write(str(node1) + " " + str(node2) + " 0" + "\n")
                temp += 1
        links_writer.close()


