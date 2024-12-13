import matplotlib.pyplot as plt
import numpy as np

data = [0.3795374747659379, 0.32269978092069185, 0.2837195222399832, 0.26560266400647004, 0.25876356426930824]
fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
plt.xticks(list(range(len(data))))

plt.savefig("test.png")
