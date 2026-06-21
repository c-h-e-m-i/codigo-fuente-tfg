import os
import numpy as np
from stable_baselines3 import PPO

class PolicyQuerier:
    def __init__(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError(f"No se encontró el archivo '{file}'")
        self.policy = PPO.load(file)

    def query(self, humos, temperaturas, densidades, restantes):
        # Creamos el diccionario de datos de AnyLogic que le enviaremos al RL:
        humos_array = np.array(humos, dtype=np.float32)
        temperaturas_array = np.array(temperaturas, dtype=np.float32)
        densidades_array = np.array(densidades, dtype=np.float32)
        restantes_array = np.array([restantes], dtype=np.float32)

        obs = {   
            "humos": humos_array,
            "temperaturas": temperaturas_array,
            "densidades": densidades_array,
            "restantes": restantes_array
        }

        # Hacemos la predicción enviando esos datos:
        accion, _ = self.policy.predict(obs, deterministic=True)
                        
        # Transformamos 'accion' de MultiDiscrete a lista de números en coma flotante:
        rutas = accion.astype(float).tolist()
        
        return rutas