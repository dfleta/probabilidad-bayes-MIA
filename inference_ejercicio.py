import torch
from model import model

# Calcular las predicciones basadas en la evidencia
# de que el tren llegue tarde (delayed)
# mediante inferencia por enumeracion

X = torch.tensor(
    [
        [
             2, # [0, 1, 2] == ["none", "light", "heavy"] -1 == no observable
            -1, # [0, 1] = ["yes", "no"] -1 == no observable
             1, # [0, 1] == ["on time", "delayed"] -1 == no observable
            -1, # [0, 1] == ["attend", "miss"] -1 == no observable
        ]
    ]
)

mask = X != -1
# mask = tensor([[False, False,  True, False]])
'''
torch.masked.MaskedTensor where the mask specifies 
which variables are observed (mask = True) 
and which ones are not observed (mask = False) for each of the values
'''

X_masked = torch.masked.MaskedTensor(X, mask)
# X_masked = MaskedTensor(
#  [
#    [      --,       --, 1,       --]
#  ]
# )

# Calculamos las predicciones de las variables ocultas
# model = The data to predict values for.
# The mask should correspond to whether the variable is observed in the example.
# predictions = A list of tensors where each tensor contains the distribution
# of values for that dimension (variable).
predictions = model.predict_proba(X_masked)

# predictions =
#  [tensor([[0.4583, 0.3069, 0.2348]]),
#  tensor([[0.3567, 0.6433]]),
#  tensor([[0., 1.]]),
#  tensor([[0.6000, 0.4000]])]

# states solo son etiquetas siguiendo la estructura de predictions
states = (
    ("rain", ["none", "light", "heavy"]),
    ("maintenance", ["yes", "no"]),
    ("train", ["on time", "delayed"]),
    ("appointment", ["attend", "miss"]),
)

# Mostramos las predicciones para cada nodo
for (node_name, values), prediction in zip(states, predictions):
    if isinstance(prediction, str):
        print(f"{node_name}: {prediction}")
    else:
        print(f"{node_name}")
        for value, probability in zip(values, prediction[0]):
            print(f"    {value}: {probability:.4f}")
