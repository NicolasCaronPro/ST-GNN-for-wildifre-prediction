
import numpy as np
from scipy.optimize import minimize

def objective(x):
    return (x[0]-5)**2 + (x[1]-3)**2

def test_nm():
    a0, r0 = 10.0, 5.0
    x0 = np.array([a0, r0], dtype=float)
    
    delta_a = 1.0
    delta_r = 1.0
    
    initial_simplex = np.array([
        [a0,            r0           ],
        [a0 + delta_a,  r0           ],
        [a0,            r0 + delta_r ],
    ], dtype=float)

    try:
        res = minimize(
            fun=objective,
            x0=x0,
            method="Nelder-Mead",
            options={
                "initial_simplex": initial_simplex,
                "maxiter": 50,
                "xatol": 0.5,
                "fatol": 1e-4,
                "disp": True,
                "adaptive": True,
            }
        )
        print("Optimization successful")
        print(res)
    except Exception as e:
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    test_nm()
