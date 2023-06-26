import pygame
import sys
import threading
import speech_recognition as sr
import heapq
import tensorflow as tf
import numpy as np

# Tamaño de la ventana y de los bloques del mapa
WIDTH = 914
HEIGHT = 600
BLOCK_SIZE = 20

# Colores
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

# Mapa de la ciudad
city_map = [
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Nombres de las calles con sus coordenadas de inicio y fin
#suponiendo que (y,x) ordenadas asi las coordenadas
street_names = {
    ((0, 4), (19, 4)): "Calle Profe Inolvidable", #
    ((5, 0), (5, 16)): "Calle del Adjetivo",#
    ((5, 16), (5, 28)): "Calle del SIELE",#
    ((0, 23), (19, 23)): "Calle del Sustantivo",#
    ((0, 28), (19, 28)): "Calle de los Errores",#
    ((10, 9), (19,9)): "Calle de Ser y Estar",#
    ((0,16), (17, 16)): "avenida Hablo Español",#
    ((10,28), (10, 10)): "Calle de los Deberes Hechos",#
    ((19, 18), (19, 29)): "avenida del indicativo",#
    ((19, 0), (19, 14)): "Avenida del Subjuntivo ",#
    ((19, 4), (44, 4)): "Calle del Vocabulario", #
    ((19, 9), (44, 9)): "Calle del Instituto Cervantes",#
    ((21, 16), (41, 16)): "Avenida ProfedeELE",#
    ((30, 16), (30, 28)): "Calle de los verbos",#
    ((35, 16), (35, 28)): "Calle de la gramática",
    ((38, 0), (38, 9)): "Calle de las dudas",#
    ((30, 23), (44, 23)): "calle del me gusta",#
    ((41, 16), (43, 9)): "Calle de la n",# error
    ((30, 28), (44, 28)): "calle de Por y Para",#
    #((5, 9), (13, 9)): "Monumento Nivel C2", #monumento
    #((3, 5), (3, 8)): "plaza del DELE" #plaza (6,7)
}
# Inicializar Pygame
pygame.init()

# Crear la ventana
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interfaz de Mapa")

# Cargar la imagen del mapa
map_image = pygame.image.load("image/mapa.jpg")

# Cargar la imagen del agente
agent_image = pygame.image.load("image/person.png")
agent_rect = agent_image.get_rect()

# Crear la ventana
WINDOW_SIZE = (WIDTH, HEIGHT)
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Interfaz de Mapa")

# Coordenadas iniciales del agente
agent_x = 0
agent_y = 4
# Velocidad de desplazamiento del agente
move_speed = BLOCK_SIZE

clock = pygame.time.Clock()

# Configurar el reconocimiento de voz
recognizer = sr.Recognizer()
recognizer.energy_threshold = 1000  # Ajusta el umbral de energía según tus necesidades

# Función para capturar el comando de voz en un hilo separado
def capture_voice_command():
    global agent_x, agent_y 
    while True:
        with sr.Microphone() as source:
            print("Di el comando:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            frase_entrada = recognizer.recognize_google(audio, language="es-ES").lower()
            command = obtener_frase_similar(frase_entrada)
            print("Comando reconocido:", command)
            move_to_street(command)

        except sr.UnknownValueError:
            print("No se pudo reconocer el comando de voz")
        except sr.RequestError as e:
            print("Error al solicitar el servicio de reconocimiento de voz: {0}".format(e))

# Función para calcular la ruta más corta utilizando el algoritmo A*
def calculate_shortest_path(start, end):
    open_list = []
    closed_list = set()

    # Crear un diccionario para almacenar los costos de los nodos
    g_scores = {start: 0}

    # Crear una cola de prioridad para almacenar los nodos a explorar
    heapq.heappush(open_list, (0, start))

    # Crear un diccionario para almacenar los padres de los nodos
    parents = {}

    while open_list:
        current_node = heapq.heappop(open_list)[1]

        if current_node == end:
            # Construir el camino desde el nodo final hasta el nodo inicial
            path = []
            while current_node in parents:
                path.insert(0, current_node)
                current_node = parents[current_node]
            return path

        closed_list.add(current_node)

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            g_score = g_scores[current_node] + 1

            if neighbor not in [node[1] for node in open_list]:
                heapq.heappush(open_list, (g_score, neighbor))
            elif g_score >= g_scores[neighbor]:
                continue

            parents[neighbor] = current_node
            g_scores[neighbor] = g_score

    return None

# Función para obtener los vecinos válidos de un nodo en el mapa
def get_neighbors(node):
    x, y = node
    neighbors = []
    if x > 0 and city_map[y][x-1] == 0:
        neighbors.append((x-1, y))
    if x < len(city_map[0])-1 and city_map[y][x+1] == 0:
        neighbors.append((x+1, y))
    if y > 0 and city_map[y-1][x] == 0:
        neighbors.append((x, y-1))
    if y < len(city_map)-1 and city_map[y+1][x] == 0:
        neighbors.append((x, y+1))
    return neighbors

# Función para mover el agente a lo largo de la ruta hacia la calle destino
def move_to_street(street_name):
    global agent_x, agent_y
    destination_coords = None
    for street_coords, name in street_names.items():
        if name.lower() == street_name:
            destination_coords = street_coords
            break

    if destination_coords is not None:
        start = (agent_x, agent_y)
        end = destination_coords[0]
        path = calculate_shortest_path(start, end)

        if path is not None:
            for node in path:
                agent_x, agent_y = node
                pygame.time.wait(500)  # Pausa de medio segundo entre movimientos
                pygame.event.pump()
                pygame.display.update()
                
def obtener_frase_similar(frase_entrada):
    # Comandos almacenadas
    frases_almacenadas = [
    "Calle Profe Inolvidable",
    "Calle del Adjetivo",
    "Calle del SIELE",
    "Calle del Sustantivo",
    "Calle de los Errores",
    "Calle de Ser y Estar",
    "avenida Hablo Español",
    "Calle de los Deberes Hechos",
    "avenida del indicativo",
    "Avenida del Subjuntivo ",
    "Calle del Vocabulario", 
    "Calle del Instituto Cervantes",
    "Avenida ProfedeELE",
    "Calle de los verbos",
    "Calle de la gramática",
    "Calle de las dudas",
    "calle del me gusta",
    "Calle de la n",
    "calle de Por y Para",
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

    # Convertir la frase de entrada en un vector numérico utilizando one-hot encoding
    frase_entrada_encoded = np.zeros(len(vocabulario))
    for palabra in frase_entrada.split():
        if palabra in vocabulario_indices:
            frase_entrada_encoded[vocabulario_indices[palabra]] = 1

    # Predecir la frase más similar utilizando la red neuronal
    predicciones = model.predict(np.array([frase_entrada_encoded]))
    indice_frase_similar = np.argmax(predicciones)
    frase_similar = frases_almacenadas[indice_frase_similar]

    return frase_similar

# Crear y ejecutar el hilo para la captura de voz
voice_thread = threading.Thread(target=capture_voice_command)
voice_thread.daemon = True
voice_thread.start()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Dibujar el mapa en la ventana
    window.fill(WHITE)
    for y in range(len(city_map)):
        for x in range(len(city_map[0])):
            if city_map[y][x] == 1:
                pygame.draw.rect(window, GRAY, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(window, WHITE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    # Dibujar el agente en la ventana
    pygame.draw.rect(window, BLACK, (agent_x * BLOCK_SIZE, agent_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    # Mostrar el nombre de la calle si el agente se encuentra en una calle con nombre
    for street_coords, street_name in street_names.items():
        (x_start, y_start), (x_end, y_end) = street_coords
        if x_start <= agent_x <= x_end and y_start <= agent_y <= y_end:
            font = pygame.font.Font(None, 24)
            text = font.render(street_name, True, BLACK)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 20))
            window.blit(text, text_rect)
            break

    pygame.display.flip()

    # Controlar la velocidad de fotogramas
    clock.tick(60)  # Ajusta la velocidad de fotogramas según tus necesidades

