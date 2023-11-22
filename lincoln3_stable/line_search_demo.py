# line_search_demo.py

import numpy as np
from scipy.optimize import line_search
import warnings
warnings.filterwarnings("ignore")

def func(x):
  # x is a np.float32 vector
  return 4*(x[0]**2) + 3*(x[1]**2) + 1  # scalar

def grad(x):
  return np.array([8*x[0], 6*x[1]], dtype=np.float32)

def main():
  print("\nBegin minimize f(x,y) = 4x^2 + 3y^2 + 1 demo")

  print("\nGradient descent ala machine learning, LR = 0.10: ")
  xk = np.array([10.0, 10.0], dtype=np.float32)
  lr = 0.10
  for i in range(15):
    g = grad(xk)
    xk = xk - (lr * g)
    fv = func(xk)
    print("i = %4d  x,y = %0.6f %0.6f fv = %0.4f" % (i, xk[0], \
xk[1], fv))

  print("\nUsing scipy line_search(): ")
  xk = np.array([9.8, 9.8], dtype=np.float32)  # start
  pk = np.array([-0.10, -0.5], dtype=np.float32)  # direction

  for i in range(200):
    results = line_search(func, grad, xk, pk, c2=0.38)  # will warn
    # print(results)
    alpha = results[0]
    if alpha is None:
      print("line_search done ")
      break

    xk = xk + alpha * pk
    fv = func(xk)
    pk = -grad(xk)
    print("i = %4d  x,y = %0.6f %0.6f fv = %0.4f" % (i, xk[0], \
xk[1], fv))

  print("\nEnd demo ")

if __name__ == "__main__":
  main()