import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x**2) + x**4 - 4*x**2 + 1

def fprime(x):
    return -2*x*np.exp(-x**2) + 4*x**3 - 8*x

TOL = 5e-7
MAXIT = 200

def bisection(a, b, tol=TOL, maxit=MAXIT):
    fa, fb = f(a), f(b)
    assert fa*fb < 0, "El intervalo no encierra un cambio de signo"
    it = 0
    history = []
    while (b - a)/2 > tol and it < maxit:
        c = (a + b)/2
        fc = f(c)
        history.append((it+1, a, b, c, fc))
        if fc == 0:
            break
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        it += 1
    c = (a + b)/2
    return c, it, history

def newton(x0, tol=TOL, maxit=MAXIT):
    x = x0
    history = []
    for it in range(1, maxit+1):
        fx = f(x)
        dfx = fprime(x)
        x_new = x - fx/dfx
        history.append((it, x, fx, dfx, x_new))
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, it, history

print("\nMetodo de Bisección\n")
intervals = [(-2.0, -1.8), (-0.75, -0.6), (0.6, 0.75), (1.8, 2.0)]
bisection_roots = []
for (a,b) in intervals:
    root, nit, hist = bisection(a,b)
    bisection_roots.append(root)
    print(f"\nIntervalo inicial [{a}, {b}]  -> raiz = {root:.6f}   iteraciones = {nit}")
    print(f"{'n':>3} {'a':>12} {'b':>12} {'c (aprox)':>12} {'f(c)':>14}")
    for row in hist[:5]:
        print(f"{row[0]:>3} {row[1]:>12.6f} {row[2]:>12.6f} {row[3]:>12.6f} {row[4]:>14.6e}")
    print("   ...")
    for row in hist[-3:]:
        print(f"{row[0]:>3} {row[1]:>12.6f} {row[2]:>12.6f} {row[3]:>12.6f} {row[4]:>14.6e}")

print("\nMetodo de Newton-Raphson\n")

x0_list = [-1.93, -0.68, 0.68, 1.93] #Puntos cercanos a las raices encontradas por bisección
newton_roots = []
for x0 in x0_list:
    root, nit, hist = newton(x0)
    newton_roots.append(root)
    print(f"\nPunto inicial x0 = {x0}  -> raiz = {root:.6f}   iteraciones = {nit}")
    print(f"{'n':>3} {'x_n':>14} {'f(x_n)':>14} {'f prime(x_n)':>14} {'x_(n+1)':>14}")
    for row in hist:
        print(f"{row[0]:>3} {row[1]:>14.8f} {row[2]:>14.6e} {row[3]:>14.6f} {row[4]:>14.8f}")

print("\nResultado (6 cifras decimales)\n")
print(f"{'Raiz':>10} {'Biseccion':>15} {'Newton-Raphson':>18}")
labels = ["x1", "x2", "x3", "x4"]
for lab, rb, rn in zip(labels, bisection_roots, newton_roots):
    print(f"{lab:>8} {rb:>16f} {rn:>18f}   f(bisec)={f(rb):.2e}  f(newton)={f(rn):.2e}")

x = np.linspace(-2.5, 2.5, 1000)
y = f(x)

plt.figure(figsize=(8,5))
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.plot(x, y, label=r'$f(x)=e^{-x^2}+x^4-4x^2+1$', color='#2E5C8A')
plt.scatter(newton_roots, [0]*len(newton_roots), color='red', zorder=5, label='Raíces')
for r in newton_roots:
    plt.annotate(f'{r:.6f}', (r, 0), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=8)

plt.title('Ceros de $f(x)=e^{-x^2}+x^4-4x^2+1$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./graphs/grafica_ejercicio2.png', dpi=150)
print("\nGrafica guardada.")