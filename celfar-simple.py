import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#datos de entrada
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
#datos de salida
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])



modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error' #Error cuadrático medio"poca cantidad de errores grandes es mejor que una gran cantidad de errores pequeños"
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=50, verbose=False)
print("Modelo entrenado!")

#funciónes para medir el nivel de error o acierto
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Hagamos una predicción!")
resultado = modelo.predict([120.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

#conocer los datos asigandos para los calculos
print("Variables internas del modelo")
print(capa.get_weights())
