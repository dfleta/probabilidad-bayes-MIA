import torch
from model import model

# Calculamos la probabilidad de un determinado evento
# dada una observacion.

rain_values = ["none", "light", "heavy"]
maintenance_values = ["yes", "no"]
train_values = ["on time", "delayed"]
appoinment_values = ["attend", "miss"]


# Cual es la probabilidad de que no llueva,
# no se realice mantenimiento,
# el tren llegue puntual
# y asistamos a la cita?

probability = model.probability(
    torch.as_tensor(
        [
            [
                rain_values.index("none"),
                maintenance_values.index("no"),
                train_values.index("on time"),
                appoinment_values.index("attend"),
            ]
        ]
    )
)

print(probability)
