import matplotlib.pyplot as plt

def show_jug(jug1, jug2, step, final=False):
    plt.figure(figsize=(4, 4))
    plt.bar(1, 4, width=0.5, color='none', edgecolor='black')
    plt.bar(1, jug1, width=0.5, color='blue', alpha=0.5)
    plt.text(1, 4.2, '4L Jug')
    plt.text(1, jug1 + 0.3 if jug1 < 4 else jug1 - 0.3, f"{jug1}L", color='blue')
    plt.bar(2, 3, width=0.5, color='none', edgecolor='black')
    plt.bar(2, jug2, width=0.5, color='blue', alpha=0.5)
    plt.text(2, 3.2, '3L Jug')
    plt.text(2, jug2 + 0.3 if jug2 < 3 else jug2 - 0.3, f"{jug2}L", color='blue')
    plt.xlim(0, 3)
    plt.ylim(0, 5)
    plt.title(f"step {step}" + (" (Start)" if step == 1 else " (Stop)" if final else ""))
    plt.show()

steps = [(0, 3), (3, 0), (3, 3), (4, 2), (0, 2), (2, 0)]
for i, (jug1, jug2) in enumerate(steps):
    show_jug(jug1, jug2, i + 1, final=(i == len(steps) - 1))
