import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Entrenando modelo...")
np.random.seed(42)

# Datos de ejemplo
X = np.random.randn(300, 13)
y = np.random.randint(0, 2, 300)

# Entrenar
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Guardar
joblib.dump(model, "heart_disease_model.joblib")
print("Modelo guardado correctamente!")
