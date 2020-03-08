
# lectures from lectures from https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data = {-1: np.array([[1, 2],
                      [2, 3],
                      [3, 1.5],
                      [7, -2]]),
        1: np.array([[5, 4],
                     [6, 7],
                     [8, 6],
                     [7, 4]])}

params = {}
vectors = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
vals = []
for yi in data:
    for feature_set in data[yi]:
        for feature in feature_set:
            vals.append(feature)
max_val = max(vals)
min_val = min(vals)
w_steps = [max_val * 0.1, max_val * 0.01, max_val * 0.001]
Bs = np.linspace(-max_val*10, max_val*10, 100)
new_w = [max_val*10, max_val*10]

for step in w_steps:
    w = np.array(new_w)
    step_optimized = False
    while not step_optimized:
        for b in Bs:
            Ws = [w * vector for vector in vectors]
            for w in Ws:
                constraint_satisfied = True
                for yi in data:  # yi class: 1 or -1
                    for xi in data[yi]:
                        if not yi * (np.dot(w, xi) + b) >= 1:
                            constraint_satisfied = False
                            break
                if constraint_satisfied:
                    params[np.linalg.norm(w)] = [w, b]
                    print("wv: %s and b: %s saved to params" % (str(w), str(b)))
        if w[0] < 0:  # been through the four kinds of vectors
            step_optimized = True
        else:
            w = w - step
    norms = sorted(_ for _ in params)
    best_combo = params[norms[0]]
    w = best_combo[0]
    b = best_combo[1]
    new_w = [w[0], w[0]]
    print("new w for next step: " + str(new_w))



plt.scatter(data[-1][:, 0], data[-1][:, 1], s=150, c='k')
plt.scatter(data[1][:, 0], data[1][:, 1], s=150, c='r')
x1 = 0.9*min_val
x2 = 1.1*max_val
y1 = (-w[0]*x1-b)/w[1]  # decision_boundary = w.x+b=0
y2 = (-w[0]*x2-b)/w[1]
plt.plot([x1, x2], [y1, y2], c='b')
plt.show()
