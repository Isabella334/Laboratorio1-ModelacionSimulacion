import numpy as np

stations = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 3.0, 1.0],
])

distances = np.array([3.0, 2.5, 2.8])

def residuals(position):
    return np.array([
        np.dot(position - station, position - station) - distance**2
        for station, distance in zip(stations, distances)
    ])

def jacobian(position):
    return np.array([
        2.0 * (position - station)
        for station in stations
    ])

def newton_raphson(initial_position, tolerance=1e-12, max_iterations=50):
    position = np.array(initial_position, dtype=float)

    for iteration in range(1, max_iterations + 1):
        F = residuals(position)
        J = jacobian(position)

        delta = np.linalg.solve(J, -F)
        position = position + delta

        print(f"Iteración {iteration:2d}: "
              f"Posición = {position}, "
              f"F = {np.linalg.norm(F):.3e}, "
              f"step = {np.linalg.norm(delta):.3e}")

        if np.linalg.norm(delta) < tolerance:
            break

    return position

print("Método Newton-Raphson multivariable:\n")
drone_position = newton_raphson(initial_position=[1.0, 1.0, 1.0])

x, y, z = drone_position
print(f"\nPosición estimada del dron:" f"(x, y, z) = ({x:.6f}, {y:.6f}, {z:.6f})\n")

print("Comparación (Real vs Calculado)")
for i, (station, distance) in enumerate(zip(stations, distances), start=1):
    computed = np.linalg.norm(drone_position - station)
    print(f"Ecuación S{i}: real = {distance}, calculado = {computed:.6f}")
