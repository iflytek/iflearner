import pickle
from cProfile import label

import matplotlib.pyplot as plt


def extract(path):
    with open(path, "rb") as f:
        inst = pickle.load(f)
        return (inst._x_list, inst._y_list)


def draw(all_top, client01_top, client02_top, client03_top, single_top, title, dis):
    plt.clf()
    plt.plot(all_top[0], all_top[1], label="nonfl-all")
    plt.plot(client01_top[0], client01_top[1], label=f"fl-{dis}-party1")
    plt.plot(client02_top[0], client02_top[1], label=f"fl-{dis}-party2")
    plt.plot(client03_top[0], client03_top[1], label=f"fl-{dis}-party3")
    plt.plot(single_top[0], single_top[1], label=f"nonfl-{dis}-party1")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")


all_top1 = extract("all/top1.pkl")
all_top5 = extract("all/top5.pkl")

client01_top1 = extract("client01/top1.pkl")
client01_top5 = extract("client01/top5.pkl")

client02_top1 = extract("client02/top1.pkl")
client02_top5 = extract("client02/top5.pkl")

client03_top1 = extract("client03/top1.pkl")
client03_top5 = extract("client03/top5.pkl")

single_top1 = extract("single/top1.pkl")
single_top5 = extract("single/top5.pkl")

draw(
    all_top1,
    client01_top1,
    client02_top1,
    client03_top1,
    single_top1,
    "Noniid Top1",
    "noniid",
)
draw(
    all_top5,
    client01_top5,
    client02_top5,
    client03_top5,
    single_top5,
    "Noniid Top5",
    "noniid",
)

iid_client01_top1 = extract("iid-client01/top1.pkl")
iid_client01_top5 = extract("iid-client01/top5.pkl")

iid_client02_top1 = extract("iid-client02/top1.pkl")
iid_client02_top5 = extract("iid-client02/top5.pkl")

iid_client03_top1 = extract("iid-client03/top1.pkl")
iid_client03_top5 = extract("iid-client03/top5.pkl")

iid_single_top1 = extract("iid-single/top1.pkl")
iid_single_top5 = extract("iid-single/top5.pkl")

draw(
    all_top1,
    iid_client01_top1,
    iid_client02_top1,
    iid_client03_top1,
    iid_single_top1,
    "Iid Top1",
    "iid",
)
draw(
    all_top5,
    iid_client01_top5,
    iid_client02_top5,
    iid_client03_top5,
    iid_single_top5,
    "Iid Top5",
    "iid",
)
