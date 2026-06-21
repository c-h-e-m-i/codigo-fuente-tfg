import os
import sys
import time
import csv
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from gymnasium.wrappers import NormalizeReward
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from alpyne.data import SimStatus
from alpyne.env import AlpyneEnv
from alpyne.sim import AnyLogicSim
from alpyne.errors import ExitException

# Variables globales:
MAX_PARTIDAS = 128
STEPS_PARTIDA = 52
PARTIDAS_UPDATE = 16
PARTIDAS_BACKUP = 32
TAM_BATCH = 64
ENTROPIA = 0.01

EPSILON = 1e-9
NOMBRE_CSV = 'resultados.csv'

class EvacuacionEnv(AlpyneEnv):

    def __init__(self, sim: AnyLogicSim, acciones, num_enlaces, num_habs):
        # Lo inicializamos:
        super().__init__(sim)

        self._acciones = acciones
        self._num_enlaces = num_enlaces
        self._num_habs = num_habs

        # Iniciamos contadores para registrar cuántas partidas y steps llevamos:
        self.partida_actual = 0
        self.steps = 0

        # Buscamos si ya existe un historial en el CSV para retomar los contadores:
        if os.path.exists(NOMBRE_CSV):
            try:
                with open(NOMBRE_CSV, 'r') as f:
                    # Leemos todas las líneas, descartando aquellas que estén en blanco:
                    lineas = [linea for linea in f.readlines() if linea.strip()]
                    
                    # Si hay datos más allá de la cabecera (len > 1):
                    if len(lineas) > 1:
                        ultima_linea = lineas[-1].split(',')
                        
                        # Extraemos la partida (columna 0) y los steps (columna 1):
                        self.partida_actual = int(ultima_linea[0])
                        self.steps = int(ultima_linea[1])
                        
                        print(f"[SISTEMA] CSV detectado. Retomando en la partida {self.partida_actual} (Step {self.steps})...")
            except Exception as e:
                print(f"[AVISO] Error al leer el CSV: {e}")

        # Definimos la estructura del input que tomará nuestro RL en cada query:
        self.observation_space = spaces.Dict({
            "humos": spaces.Box(low=0.0, high=1.0, shape=(self._num_habs,), dtype=np.float32),
            "temperaturas": spaces.Box(low=0.0, high=1.0, shape=(self._num_habs,), dtype=np.float32),
            "densidades": spaces.Box(low=0.0, high=1.0, shape=(self._num_enlaces,), dtype=np.float32),
            "restantes": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        # Definimos la estructura del output:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(sum(self._acciones),), dtype=np.float32)

        # Como no tenemos interfaz en el entrenamiento, pondremos el modo de renderizado a 'None':
        self.render_mode = None

        # Creamos el resto de atributos necesarios (los rellenaremos en _get_obs):
        self._mediaQuemaduras = 0.0
        self._mediaAsfixia = 0.0
        self._humos = [0.0] * self._num_habs
        self._temperaturas = [0.0] * self._num_habs
        self._densidades = [0.0] * self._num_enlaces
        self._supervivencia = 0.0
        self._supervivencia_anterior = 0.0
        self._tiempo = 0.0
        self._restantes = 0.0
        self._estado = None
        self._gente = 0.0
        self._recompensa = 0.0

    def _get_obs(self, status: SimStatus) -> ObsType:
        # Leemos los datos que nos ha enviado AnyLogic:
        obs = status.observation

        # Seleccionamos los campos que nos interesan de esos datos:
        self._humos = np.array(obs.get('humos', self._humos), dtype=np.float32)
        self._temperaturas = np.array(obs.get('temperaturas', self._temperaturas), dtype=np.float32)
        self._densidades = np.array(obs.get('densidades', self._densidades), dtype=np.float32)
        self._restantes = np.array([obs.get('restantes', self._restantes)], dtype=np.float32)
 
        # Nos guardamos el resto de datos como atributos de clase, si bien no formarán parte de la X de la f(X) del modelo:
        self._mediaQuemaduras = float(obs.get('mediaQuemaduras', self._mediaQuemaduras))
        self._mediaAsfixia = float(obs.get('mediaAsfixia', self._mediaAsfixia))
        self._supervivencia = float(obs.get('supervivencia', self._supervivencia))
        self._tiempo = float(obs.get('tiempo', self._tiempo))
        self._gente = float(obs.get('gente', self._gente))
        self._estado = str(status.state).upper()

        # Devolvemos el input para nuestro RL:
        return {
            "humos": self._humos,
            "temperaturas": self._temperaturas,
            "densidades": self._densidades,
            "restantes": self._restantes,
        }

    def _calc_reward(self, status: SimStatus) -> float:
        # Aumentamos el número de steps:
        self.steps += 1

        recompensa = 0.0

        done = self._is_terminal(status)

        # RECOMPENSAS INTERMEDIAS:
        if not done:
            # Buscamos que la densidad máxima sea la mínima posible en todo momento:
            max_densidad = max(self._densidades)
            if max_densidad > 0.3:
                recompensa -= 5.0 * max_densidad

            recompensa += 5000.0 * (self._supervivencia - self._supervivencia_anterior)
            self._supervivencia_anterior = self._supervivencia

            self._recompensa += recompensa
        
        # RECOMPENSAS FINALES:
        else:
            recompensa -= 250.0 * self._mediaQuemaduras
            recompensa -= 250.0 * self._mediaAsfixia

            self._recompensa += recompensa

            if not os.path.exists(NOMBRE_CSV):
                with open(NOMBRE_CSV, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Partida", "Steps", "Tiempo", "Supervivencia", "Asfixia", "Quemaduras", "Recompensa"])

            with open(NOMBRE_CSV, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.partida_actual, self.steps, self._tiempo, 
                                 self._supervivencia, self._mediaAsfixia,
                                 self._mediaQuemaduras, self._recompensa])

            print(f"[TIEMPO] {self._tiempo:.2f} s")
            print(f"[SUPERVIVENCIA] {self._supervivencia * 100:.2f} %")
            print(f"[QUEMADURAS] {self._mediaQuemaduras * 100:.2f} %")
            print(f"[ASFIXIA] {self._mediaAsfixia * 100:.2f} %")
            print(f"[RECOMPENSA] {self._recompensa:.2f} puntos")

        return recompensa

    def _to_action(self, act: ActType) -> dict:
        # Transformamos la acción que nos devuelve el RL en una lista de números en coma flotante:
        accion = act.astype(float).tolist()

        # La devolvemos como output para AnyLogic en forma de diccionario:
        return dict(rutas=accion)
    
    def _is_terminal(self, status: SimStatus) -> bool:
        # Solo consideraremos que la simulación ha terminado naturalmente (es decir, no se ha truncado)
        # si no queda gente en el edificio:
        if self._restantes[0] <= EPSILON: return True
    
        paradas_forzosas = ["MODEL_METHOD", "STOP", "STOP_TIMEDATE", "ERROR", "FAILED", "UNKNOWN"]
        if any(estado_raro in self._estado for estado_raro in paradas_forzosas):
            print(f"\n[AVISO] Partida detenida por AnyLogic (Estado: {self._estado})")
            return True

        return False
    
    def reset(self, seed=None, options=None):
        # Actualizamos el contador de partidas:
        self.partida_actual += 1
        print(f"[PARTIDA] {self.partida_actual}")

        # Limpiamos el índice de supervivencia anterior:
        self._supervivencia_anterior = 0.0

        # También limpiamos la recompensa acumulada:
        self._recompensa = 0.0
        
        # Reiniciamos el entorno tras cada step:
        return super().reset(seed=seed, options=options)
    
def config():
    archivo = "config.txt"
    print("Esperando la topología desde AnyLogic...")
    
    while not os.path.exists(archivo):
        time.sleep(0.1)
        
    with open(archivo, "r") as f:
        lineas = f.readlines()

    # Construimos la lista de acciones:
    acciones = [int(numero) for numero in lineas[0].strip().split(",")]

    # Sacamos el número de enlaces:
    enlaces = int(lineas[1].strip())

    # Sacamos el número de habitaciones:
    habs = int(lineas[2].strip())
    
    return acciones, enlaces, habs

if __name__ == '__main__':
    # Le enviamos a AnyLogic la ruta del modelo y de Java:
    ruta_modelo = r"rl\model.jar" 
    ruta_java = r"C:\Users\usuario\Desktop\programas\OpenJDK\jdk-25.0.1\bin\java.exe"
    ruta_backup = "PPO_backup.zip"
    
    assert os.path.exists(ruta_modelo), f"No se encuentra el modelo en: {ruta_modelo}"

    # Enviamos esas rutas a la simulación de AnyLogic:
    sim = AnyLogicSim(ruta_modelo, java_exe=ruta_java, logging=True)

    # Creamos el entorno de entrenamiento:
    acciones, enlaces, habs = config()
    env = EvacuacionEnv(sim, acciones, enlaces, habs)

    # Wrappers (son como DLCs del modelo):
    env = Monitor(env) # Esto sirve para ir anotando datos para el cálculo de esas métricas internas que te da SB3 cuando acaba el entrenamiento
    env = NormalizeReward(env) # Normalizamos las recompensas para que estén entre 0 y 1
    env = DummyVecEnv([lambda: env]) # Esto también está relacionado con el cálculo de métricas. SB3 es quisquilloso y quiere que metas tu 'env' en una lista

    # Buscamos si ya había un modelo a medio entrenar:
    nuevo = False
    if os.path.exists(ruta_backup):
        print("\n[SISTEMA] ¡Modelo anterior encontrado! Cargando recuerdos de las partidas previas...")
        model = PPO.load(ruta_backup, env=env, custom_objects={"ent_coef": ENTROPIA, "n_steps": STEPS_PARTIDA * PARTIDAS_UPDATE})
    else:
        arquitectura = dict(net_arch=[256, 256])

        print("\n[SISTEMA] No hay modelo previo. Creando uno nuevo desde cero...")
        model = PPO("MultiInputPolicy", env, learning_rate=0.0003, n_steps=STEPS_PARTIDA * PARTIDAS_UPDATE, batch_size=TAM_BATCH, ent_coef=ENTROPIA, policy_kwargs=arquitectura, verbose=1)
        nuevo = True

    # Configurar el auto-guardado automático
    checkpoint_callback = CheckpointCallback(
        save_freq=STEPS_PARTIDA * PARTIDAS_BACKUP,
        save_path=".",
        name_prefix="PPO_backup",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # Entrenamos el modelo (permitiendo abortar el entrenamiento con Ctrl+C):
    print("Iniciando entrenamiento...")
    try:
        model.learn(total_timesteps=STEPS_PARTIDA * MAX_PARTIDAS, reset_num_timesteps=nuevo, callback=checkpoint_callback)
    except (KeyboardInterrupt, ExitException):
        print("Entrenamiento abortado")
        print("\n[AVISO] Entrenamiento abortado. Guardando progreso...")
        model.save(ruta_backup)
        sys.exit(0)

    model.save(ruta_backup)
    print("[MANTENIMIENTO] ¡Bloque terminado y modelo guardado!")
    print("[MANTENIMIENTO] Cerrando Python...\n")
    
    # Esto fuerza el cierre de Python de forma totalmente limpia:
    sys.exit(0)