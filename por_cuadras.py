import pygame
import sys
import threading
import speech_recognition as sr

# Tamaño de la ventana y de los bloques del mapa
WIDTH = 800
HEIGHT = 600
BLOCK_SIZE = 40

# Colores
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

# Mapa de la ciudad
city_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Nombres de las calles con sus coordenadas de inicio y fin
street_names = {
    ((1, 1), (1, 7)): "Avenida Principal",
    ((1, 9), (3, 9)): "Calle Central"
}

# Inicializar Pygame
pygame.init()

# Crear la ventana
WINDOW_SIZE = (WIDTH, HEIGHT)
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Interfaz de Mapa")

# Coordenadas iniciales del agente
agent_x = 1
agent_y = 1

# Velocidad de desplazamiento del agente
move_speed = BLOCK_SIZE

clock = pygame.time.Clock()

# Configurar el reconocimiento de voz
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000  # Ajusta el umbral de energía según tus necesidades

# Función para capturar el comando de voz en un hilo separado
def capture_voice_command():
    global agent_x, agent_y
    while True:
        with sr.Microphone() as source:
            print("Di el comando:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio, language="es-ES").lower()
            print("Comando reconocido:", command)
            # Interpretar y procesar el comando de voz aquí

            # Calcular la nueva posición del agente en función del movimiento
            move_x = 0
            move_y = 0

            if "izquierda" in command:
                move_x = -1
            elif "derecha" in command:
                move_x = 1
            elif "arriba" in command:
                move_y = -1
            elif "abajo" in command:
                move_y = 1

            new_x = agent_x + move_x
            new_y = agent_y + move_y

            # Verificar si la nueva posición está dentro de los límites del mapa y es una calle
            if (
                0 <= new_x < len(city_map[0])
                and 0 <= new_y < len(city_map)
                and city_map[new_y][new_x] == 0
            ):
                agent_x = new_x
                agent_y = new_y

        except sr.UnknownValueError:
            print("No se pudo reconocer el comando de voz")
        except sr.RequestError as e:
            print("Error al solicitar el servicio de reconocimiento de voz; {0}".format(e))

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
