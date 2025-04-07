from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

# Crear distribuciones de probabilidad

# Creamos los nodos de la red
# e indicamos la distribucion de probabilidad de cada uno.

# Rain node no tiene padres
rain_distribution = Categorical([[0.7, 0.2, 0.1]])

# Track maintenance node esta condicionado por rain
maintenance_distribution = ConditionalCategorical(
    [[[0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]], [rain_distribution]
)

# Train node esta condicionado por rain y maintenance
train_distribution = ConditionalCategorical(
    [
        [
            [
                [0.8, 0.2],
                [0.9, 0.1],
            ],
            [
                [0.6, 0.4],
                [0.7, 0.3],
            ],
            [
                [0.4, 0.6],
                [0.5, 0.5],
            ],
        ]
    ],
    [rain_distribution, maintenance_distribution],
)

# Appointment node esta condicionado por train
appointment_distribution = ConditionalCategorical(
    [
        [
            [0.9, 0.1],
            [0.6, 0.4],
        ]
    ],
    [train_distribution],
)

# Creamos el modelo y añadimos las distribuciones de probabilidad
model = BayesianNetwork()
model.add_distributions(
    [
        rain_distribution,
        maintenance_distribution,
        train_distribution,
        appointment_distribution,
    ]
)

# Añadimos las aristas entre nodos para indicar dependencia condicional
model.add_edge(rain_distribution, maintenance_distribution)
model.add_edge(rain_distribution, train_distribution)
model.add_edge(maintenance_distribution, train_distribution)
model.add_edge(train_distribution, appointment_distribution)
