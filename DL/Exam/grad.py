import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x=2*np.random.rand(100, 1)
y=4 + 3 * x + np.random.randn(100, 1)
x_b=np.c_[np.ones((100, 1)), x]
theta=np.zeros((2, 1))
alpha=0.01
iterations=1000
m=len(y)
cost_history=[]
for i in range(iterations):
    predictions=x_b.dot(theta)
    errors=predictions - y
    gradients=(1/m) * x_b.T.dot(errors)
    theta=theta - alpha * gradients
    cost=(1/(2*m)) * np.sum(errors**2)
    cost_history.append(cost)
    if i>0 and abs(cost_history[-2] - cost_history[-1]) < 1e-6:
        print("Optimized parameters (theta):")
        print(theta)
        print(f"Final Cost: {cost_history[-1]:.6f}")
        print(f"Total Iterations: {i+1}")
if not (len(cost_history) > 1 and abs(cost_history[-2] - cost_history[-1]) < 1e-6):
    print("Final parameters (theta):")
    print(theta)
    print(f"Final Cost: {cost_history[-1]:.6f}")
    print(f"Total Iterations: {iterations}")

plt.scatter(x, y, color='blue', label='Training Data')
plt.plot(x, x_b.dot(theta), color='red', label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost J[0]')
plt.title('Cost Function Decreases over Iterations')
plt.show()
