from collections import Counter

from model import model

# Rejection sampling
# Calcular la distribucion de llegar "attend" a la cita "Appointment"
# dado el hecho de que el tren es "delayed"

# Numero de muestras
N = 10000

# Cogemos las muestras con:
# [?, ?, 1, 0]  [rain, maintenance, "delayed", "attend"]
# [?, ?, 1, 1]  [rain, maintenance, "delayed", "miss"]
# ver fichero inference.py con la equivalencia de 0/1
# al dominio de las variables

samples = model.sample(N)

# samples = model.sample(10)
# tensor([[1, 1, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 0],
#        [0, 1, 0, 0],
#        [0, 0, 1, 0],   <= delated y attend
#        [0, 1, 0, 0],
#        [0, 1, 0, 0],
#        [1, 1, 1, 1],   <= delated y miss
#        [0, 0, 0, 0]], dtype=torch.int32)

# sample[2] == train, 1 == delayed
# sample[3] == appoinment, 0 == attend, 1 == miss

data = ["attend" if sample[3] == 0 else "miss"
        for sample in list(filter(lambda sample: sample[2] == 1,
        samples))]

# data = ['attend', 'miss']

print(Counter(data))

# Counter({'attend': 1, 'miss': 1})
