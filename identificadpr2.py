import speech_recognition as sr
import tensorflow as tf
import numpy as np

# Frases almacenadas
frases_almacenadas = [
        "Agente vaya a la Calle Profe Inolvidable",
        "Agente vaya a la Calle del Adjetivo",
        "Agente vaya a la Calle del SIELE",
        "Agente vaya a la Calle del Sustantivo",
        "Agente vaya a la Calle de los Errores",
        "Agente vaya a la Calle del Ser y Estar",
        "Agente vaya a la avenida Hablo Espaniol",
        "Agente vaya a la Calle de los Deberes Hechos",
        "Agente vaya a la Avenida del Indicativo",
        "Agente vaya a la Avenida del Subjuntivo",
        "Agente vaya a la Calle del Vocabulario",
        "Agente vaya a la Calle del Instituto Cervantes",
        "Agente vaya a la Avenida Profe de ELE",
        "Agente vaya a la Calle de los verbos",
        "Agente vaya a la Calle de la gramatica",
        "Agente vaya a la Calle de las dudas",
        "Agente vaya a la calle del me gusta",
        "Agente vaya a la Calle de la ñ",
        "Agente vaya a la calle de Por y Para",
        "Agente vaya a la plaza del DELE"
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
    tf.keras.layers.Dense(64, input_shape=input_shape, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(frases_almacenadas), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Entrenamiento de la red neuronal
model.fit(frases_almacenadas_encoded, np.eye(len(frases_almacenadas)), epochs=100)

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
