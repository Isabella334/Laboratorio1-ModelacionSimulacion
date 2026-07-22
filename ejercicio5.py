import numpy as np
import matplotlib.pyplot as plt

def TA(n):
    return 3*n*np.log(n)

def TB(n):
    return 0.5*n**1.5 + 10

def g(n):
    return TB(n) - TA(n)

TOL = 1e-6
MAXIT = 200

def bisection(a, b, tol=TOL, maxit=MAXIT):
    fa, fb = g(a), g(b)
    assert fa*fb < 0, "El intervalo no encierra un cambio de signo"
    history = []
    it = 0
    while (b - a) > tol and it < maxit:
        it += 1
        c = (a + b)/2
        fc = g(c)
        err = abs(b - a)
        history.append((it, a, b, c, fc, err))
        if fc == 0:
            a = b = c
            break
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    c = (a + b)/2
    return c, it, history


print("Paso 1: Verificacion del intervalo propuesto en el enunciado [100, 1000]\n")
a_orig, b_orig = 100, 1000
g_a_orig, g_b_orig = g(a_orig), g(b_orig)
print(f"g({a_orig})  = {g_a_orig:.6f}")
print(f"g({b_orig}) = {g_b_orig:.6f}")
print(f"g({a_orig}) * g({b_orig}) = {g_a_orig*g_b_orig:.6f}")

if g_a_orig * g_b_orig > 0:
    print("\n-> Ambos valores tienen el MISMO signo (producto positivo).")
    print("-> NO se cumple la condicion del Teorema de Bolzano: g(a)*g(b) < 0")
    print("-> Por lo tanto, NO hay garantia de raiz en [100, 1000], y la biseccion")
    print("   no puede aplicarse directamente en este intervalo.")
else:
    print("\n-> Se cumple Bolzano en el intervalo original. Se procede con este intervalo.")

print("\nPaso 2: Busqueda de un intervalo valido mediante barrido numerico\n")
print("Se evalua g(n) en un rango amplio de n para localizar todos los cambios de signo.")

n_scan = np.linspace(1, 20000, 400000)
g_scan = g(n_scan)
cruces = []
for i in range(len(n_scan) - 1):
    if g_scan[i] * g_scan[i+1] < 0:
        cruces.append((n_scan[i], n_scan[i+1]))

print(f"\nCambios de signo detectados en (0, 20000]:")
for c in cruces:
    print(f"  entre n = {c[0]:.4f} y n = {c[1]:.4f}   "
          f"(g={g(c[0]):.4f} -> g={g(c[1]):.4f})")

print("\nInterpretacion:")
print(" - El primer cruce corresponde a valores de n muy pequenos,")
print("   donde el termino constante (+10) de T_B(n) todavia domina.")
print(" - El segundo cruce es el relevante para demostrar el momento en que B supera a A:")
print("   alli T_B(n) = 0.5 n^1.5 + 10 vuelve a superar a T_A(n) = 3n*ln(n) de forma")
print("   permanente, ya que n^1.5 crece asintoticamente mas rapido que n*ln(n).")

print("\n-> Nuevo intervalo de trabajo seleccionado: [2000, 2200]")


print("\nPaso 3: Verificacion de Bolzano en el nuevo intervalo [2000, 2200]\n")
a0, b0 = 2000, 2200
g_a0, g_b0 = g(a0), g(b0)
print(f"g({a0}) = {g_a0:.6f}   (TA={TA(a0):.4f}, TB={TB(a0):.4f})")
print(f"g({b0}) = {g_b0:.6f}   (TA={TA(b0):.4f}, TB={TB(b0):.4f})")
print(f"g({a0}) * g({b0}) = {g_a0*g_b0:.6f}  -> signos opuestos, se cumple Bolzano.")

raiz, nit, hist = bisection(a0, b0)

print("\nPaso 4: METODO DE BISECCION - Punto de equilibrio (n a partir del cual B supera a A)\n")
print(f"{'k':>3} {'a_k':>14} {'b_k':>14} {'c_k=(a+b)/2':>14} {'g(c_k)':>16} {'|b_k-a_k|':>14}")
for row in hist:
    k, a, b, c, fc, err = row
    print(f"{k:>3} {a:>14.6f} {b:>14.6f} {c:>14.6f} {fc:>16.6e} {err:>14.6f}")

print(f"\nNumero de iteraciones: {nit}")
print(f"n de equilibrio (exacto, 6 decimales) = {raiz:.6f}")
print(f"g(n*) = {g(raiz):.6e}")
print(f"TA(n*) = {TA(raiz):.6f}   TB(n*) = {TB(raiz):.6f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

n_vals = np.linspace(50, 3000, 3000)
axes[0].plot(n_vals, TA(n_vals), label=r'$T_A(n)=3n\ln(n)$', color='#2E5C8A')
axes[0].plot(n_vals, TB(n_vals), label=r'$T_B(n)=0.5n^{1.5}+10$', color='#C24C2E')
axes[0].axvline(raiz, color='gray', linestyle='--', linewidth=1)
axes[0].scatter([raiz], [TA(raiz)], color='red', zorder=5)
axes[0].annotate(f'n* = {raiz:.4f}', (raiz, TA(raiz)), textcoords="offset points",
                  xytext=(10, -15), fontsize=9)
axes[0].set_xlabel('n (tamano del conjunto de datos)')
axes[0].set_ylabel('Tiempo de ejecucion (s)')
axes[0].set_title('Comparacion de tiempos T_A(n) y T_B(n)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].axhline(0, color='gray', linewidth=0.8)
axes[1].plot(n_vals, g(n_vals), color='#2E5C8A', label=r'$g(n)=T_B(n)-T_A(n)$')
axes[1].scatter([raiz], [0], color='red', zorder=5, label='Raiz (punto de equilibrio)')
axes[1].annotate(f'{raiz:.6f}', (raiz, 0), textcoords="offset points",
                  xytext=(0, 12), ha='center', fontsize=9)
axes[1].set_xlabel('n')
axes[1].set_ylabel('g(n)')
axes[1].set_title('g(n) = T_B(n) - T_A(n)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./graphs/grafica_ejercicio5.png', dpi=150)
print("\nGrafica guardada")