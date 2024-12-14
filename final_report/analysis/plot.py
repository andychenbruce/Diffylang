import matplotlib.pyplot as plt
import numpy as np
import json



data = json.loads(open("./for_loop.json", "rb").read())
fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
# plt.xticks(list(range(len(data))))

plt.savefig("test.png")
