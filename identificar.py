import speech_recognition as sr
import tensorflow as tf
import numpy as np

# Frases almacenadas
frases_almacenadas = [
    "Agente 1, avanzar una cuadra por la calle de los Deberes Hechos",
    "Nombre del agente ir a la calle del adjetivo esquina calle profe inolvidable"
]

# Convertir las frases almacenadas en vectores numéricos utilizando one-hot encoding
vocabulario = list(set(" ".join(frases_almacenadas).split()))
vocabulario.sort()
vocabulario_indices = dict((c, i) for i, c in enumerate(vocabulario))
indices_vocabulario = dict((i, c) for i, c in enumerate(vocabulario))

frases_almacenadas_encoded = []
for frase in frases_almacenadas:
    frase_encoded = np.zeros(len(vocabulario))
    for palabra in frase.split():
        frase_encoded[vocabulario_indices[palabra]] = 1
    frases_almacenadas_encoded.append(frase_encoded)

frases_almacenadas_encoded = np.array(frases_almacenadas_encoded)

# Crear y entrenar la red neuronal
input_shape = (len(vocabulario),)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=input_shape, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(frases_almacenadas), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Entrenamiento de la red neuronal (suponiendo que tienes datos de entrenamiento)
# x_train y y_train serían tus datos de entrenamiento
# model.fit(x_train, y_train, epochs=10)

# Reconocimiento de voz
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Di una frase:")
    audio = r.listen(source)

try:
    frase_entrada = r.recognize_google(audio, language="es-ES")
    print("Frase de entrada:", frase_entrada)

    # Convertir la frase de entrada en un vector numérico utilizando one-hot encoding
    frase_entrada_encoded = np.zeros(len(vocabulario))
    for palabra in frase_entrada.split():
        if palabra in vocabulario_indices:
            frase_entrada_encoded[vocabulario_indices[palabra]] = 1

    # Predecir la frase más similar utilizando la red neuronal
    predicciones = model.predict(np.array([frase_entrada_encoded]))
    indice_frase_similar = np.argmax(predicciones)
    frase_similar = frases_almacenadas[indice_frase_similar]

    print("Frase similar encontrada:", frase_similar)

except sr.UnknownValueError:
    print("No se pudo reconocer la frase de entrada.")
except sr.RequestError as e:
    print("Error al hacer la solicitud a Google Speech Recognition:", str(e))
