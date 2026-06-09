import pybullet as p
import pybullet_data
import time

# 1. Conectar al motor de físicas y abrir la interfaz gráfica (GUI)
physicsClient = p.connect(p.GUI)

# 2. Decirle a PyBullet dónde encontrar los modelos de prueba predeterminados
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Configurar la gravedad de la simulación (eje Z hacia abajo)
p.setGravity(0, 0, -9.81)

# 4. Cargar un suelo (plano) para que los objetos no caigan al vacío
plano = p.loadURDF("plane.urdf")

# 5. Cargar un modelo 3D (R2D2) en una posición inicial [X, Y, Z]
posicion_inicial = [0, 0, 2] # Aparecerá a 2 metros de altura
orientacion_inicial = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("r2d2.urdf", posicion_inicial, orientacion_inicial)

print("¡Simulación iniciada! Presiona Ctrl+C en la terminal para salir.")

# 6. Bucle infinito para mantener la ventana abierta y avanzar la simulación
try:
    while True:
        p.stepSimulation() # Avanza un fotograma en las físicas
        time.sleep(1./240.) # PyBullet funciona a 240Hz por defecto
except KeyboardInterrupt:
    print("\nCerrando simulación...")

# 7. Desconectar y cerrar la ventana de forma limpia
p.disconnect()
