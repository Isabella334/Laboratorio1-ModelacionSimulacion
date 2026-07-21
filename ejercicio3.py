import numpy as np
import matplotlib.pyplot as plt

tolerance = 1e-12
max_iterations = 200

def f_x(x):
    return np.exp(x) - 3*x**2

def df_x(x):
    return np.exp(x) - 6*x

def bisection(a, b):
    err = []
    xprev = a
    c = (a + b) / 2

    for _ in range(max_iterations):
        c = (a + b) / 2
        err.append(abs(c - xprev))

        if f_x(a) * f_x(c) < 0:
            b = c

        else:
            a = c

        if abs(b - a) < tolerance:
            break
        xprev = c

    return c, err

def secant(x0, x1):
    err = []
    x2 = x1

    for _ in range(max_iterations):
        fx0 = f_x(x0)
        fx1 = f_x(x1)
        x2 = x1 - fx1*(x1-x0)/(fx1-fx0)
        e = abs(x2 - x1)
        err.append(e)
        x0, x1 = x1, x2

        if e < tolerance: 
            break

    return x2, err

def newton_raphson(x0):
    err = []
    x1 = x0 - f_x(x0)/df_x(x0)

    for _ in range(max_iterations):
        e = abs(x1 - x0)
        err.append(e)
        x0 = x1

        if e < tolerance: 
            break

    return x1, err

rb, eb = bisection(3.0, 4.0)
rs, es = secant(3.0, 4.0)
rn, en = newton_raphson(3.7)
 
print(f"Biseccion: raiz={rb:.14f}  iteraciones={len(eb)}")
print(f"Secante: raiz={rs:.14f}  iteraciones={len(es)}")
print(f"Newton-Raphson: raiz={rn:.14f}  iteraciones={len(en)}")
print(f"f(raiz) Newton = {f_x(rn):.2e}")

plt.figure(figsize=(8,5))
plt.plot(range(1,len(eb)+1), eb, 'o-', label='Biseccion')
plt.plot(range(1,len(es)+1), es, 's-', label='Secante')
plt.plot(range(1,len(en)+1), en, '^-', label='Newton-Raphson')
plt.xlabel('k (iteracion)'); plt.ylabel('error e_k')
plt.title('Error absoluto por iteracion (escala lineal)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('./ej3_lineal.png', dpi=110, bbox_inches='tight')
 
plt.figure(figsize=(8,5))
plt.semilogy(range(1,len(eb)+1), eb, 'o-', label='Biseccion (p≈1)')
plt.semilogy(range(1,len(es)+1), es, 's-', label='Secante (p≈1.618)')
plt.semilogy(range(1,len(en)+1), en, '^-', label='Newton-Raphson (p=2)')
plt.xlabel('k (iteracion)'); plt.ylabel('error e_k (log)')
plt.title('Error absoluto por iteracion (escala logaritmica)')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.savefig('./ej3_log.png', dpi=110, bbox_inches='tight')
