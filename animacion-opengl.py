import glfw
from OpenGL.GL import *
import numpy as np
from ctypes import c_void_p
import math
from PIL import Image

# ====================================================================
# I. VARIABLES GLOBALES Y ESTADOS DE ANIMACIÓN
# ====================================================================

STATE_WALKING = 0
STATE_EXPLODING = 1
STATE_RESET = 2

current_state = STATE_WALKING
state_start_time = 0.0
walk_duration = 5.0
explosion_duration = 1.5

window = None
shader_program = None
head_vao = None
torso_vao = None
leg_vao = None

mouse_angle_h = 0.0
mouse_angle_v = 15.0
camera_distance = 5.0

lastX = 800 / 2
lastY = 600 / 2
first_mouse = True

head_texture_id = None
torso_texture_id = None
leg_texture_id = None

# ====================================================================
# II. DATOS DE GEOMETRÍA (8 VALORES POR VÉRTICE: P, UV, NORMAL)
# NOTA: Los UVs son simples (0.0 a 1.0) para usar texturas separadas.
# ====================================================================

# Normales (Nx, Ny, Nz) para cada cara:
# Delantera: 0, 0, -1 | Trasera: 0, 0, 1 | Izquierda: -1, 0, 0
# Derecha: 1, 0, 0 | Inferior: 0, -1, 0 | Superior: 0, 1, 0

# Vértices base (X, Y, Z, U, V, Nx, Ny, Nz) - 8 Componentes
SIMPLE_CUBE_UVS_NORMALS = np.array([
    # Cara frontal (Normal: 0, 0, -1)
    -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0,
    0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, -1.0,
    0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 0.0, -1.0,
    0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 0.0, -1.0,
    -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, -1.0,
    -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0,

    # Cara trasera (Normal: 0, 0, 1)
    -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
    -0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0,
    -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0,

    # Cara izquierda (Normal: -1, 0, 0)
    -0.5, 0.5, 0.5, 0.0, 1.0, -1.0, 0.0, 0.0,
    -0.5, 0.5, -0.5, 1.0, 1.0, -1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5, 1.0, 0.0, -1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5, 1.0, 0.0, -1.0, 0.0, 0.0,
    -0.5, -0.5, 0.5, 0.0, 0.0, -1.0, 0.0, 0.0,
    -0.5, 0.5, 0.5, 0.0, 1.0, -1.0, 0.0, 0.0,

    # Cara derecha (Normal: 1, 0, 0)
    0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0,
    0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0,

    # Cara inferior (Normal: 0, -1, 0)
    -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, -1.0, 0.0,
    0.5, -0.5, -0.5, 1.0, 1.0, 0.0, -1.0, 0.0,
    0.5, -0.5, 0.5, 1.0, 0.0, 0.0, -1.0, 0.0,
    0.5, -0.5, 0.5, 1.0, 0.0, 0.0, -1.0, 0.0,
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, -1.0, 0.0,
    -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, -1.0, 0.0,

    # Cara superior (Normal: 0, 1, 0)
    -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
    -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0,
    -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
], dtype=np.float32)

head_vertices = SIMPLE_CUBE_UVS_NORMALS
torso_vertices = SIMPLE_CUBE_UVS_NORMALS
leg_vertices = SIMPLE_CUBE_UVS_NORMALS

# ====================================================================
# III. SHADERS (CÓDIGO GLSL) - ILUMINACIÓN
# ====================================================================
VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec3 aPos; 
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal; // <-- NUEVO

out vec2 TexCoord;
out vec3 Normal; // Normal transformada
out vec3 FragPos; // Posición del vértice en espacio de mundo

uniform mat4 model;      
uniform mat4 view;  
uniform mat4 projection; 
uniform mat4 normalMatrix; // Matriz para transformar las normales

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    // Transformamos la normal a espacio de mundo
    Normal = mat3(normalMatrix) * aNormal; 
    TexCoord = aTexCoord;
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor; 

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D headSampler;
uniform sampler2D torsoSampler;
uniform sampler2D legSampler;

uniform vec3 objectColor;
uniform int currentPart; 

uniform vec3 lightDirection; // Dirección de la luz (vector)
uniform float ambientStrength; // Intensidad de la luz ambiental
uniform vec3 lightColor; // Color de la luz

void main()
{
    // 1. LUZ AMBIENTAL
    vec3 ambient = ambientStrength * lightColor;

    // 2. LUZ DIFFUSE (Lambert)
    vec3 norm = normalize(Normal);
    // Dirección de la luz desde la superficie (debe ser normalizada)
    vec3 lightDir = normalize(lightDirection); 

    // Cálculo del factor Diffuse (max(producto punto, 0.0))
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // 3. COLOR FINAL
    vec3 lighting = ambient + diffuse;

    vec4 textureColor = vec4(1.0);

    // Selección de la textura basada en la pieza
    if (currentPart == 0) { 
        textureColor = texture(headSampler, TexCoord);
    } else if (currentPart == 1) { 
        textureColor = texture(torsoSampler, TexCoord);
    } else if (currentPart == 2) { 
        textureColor = texture(legSampler, TexCoord); 
    } else {
        textureColor = vec4(1.0); 
    }

    // Multiplicamos la textura por el factor de iluminación y el color de control
    FragColor = vec4(textureColor.rgb * lighting, textureColor.a) * vec4(objectColor, 1.0);
}
"""


# ====================================================================
# IV. FUNCIONES DE LÓGICA Y UTILIDAD
# ====================================================================

## A. Manejo de Shaders (Sin cambios)
def compile_shader(source, type):
    shader = glCreateShader(type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        info = glGetShaderInfoLog(shader)
        print(f"Error de compilación de shader: \n{info.decode()}")
        glDeleteShader(shader)
        raise RuntimeError("Shader compilation failed")
    return shader


def create_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        info = glGetProgramInfoLog(program)
        print(f"Error de enlace de programa: \n{info.decode()}")
        glDeleteProgram(program)
        raise RuntimeError("Program linking failed")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program


## B. Configuración de Buffers (Múltiples VAOs con 8 valores)
def setup_creeper_buffers(vertices):
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)

    # 8 * 4 bytes/float = 32 bytes por vértice
    BYTES_PER_VERTEX = 8 * vertices.itemsize

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Atributo 0: Posición (X, Y, Z). Offset 0. Size 3.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, BYTES_PER_VERTEX, c_void_p(0))
    glEnableVertexAttribArray(0)

    # Atributo 1: UVs (U, V). Offset 3 floats. Size 2.
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, BYTES_PER_VERTEX, c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    # Atributo 2: Normal (Nx, Ny, Nz). Offset 5 floats (3 pos + 2 uv). Size 3.
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, BYTES_PER_VERTEX, c_void_p(5 * vertices.itemsize))
    glEnableVertexAttribArray(2)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return VAO


def initialize_all_vaos(head_v, torso_v, leg_v):
    global head_vao, torso_vao, leg_vao
    head_vao = setup_creeper_buffers(head_v)
    torso_vao = setup_creeper_buffers(torso_v)
    leg_vao = setup_creeper_buffers(leg_v)


## C. Funciones de Matemáticas (Matrices)
# --- Función para obtener la matriz normal ---
def get_normal_matrix(model_matrix):
    """Calcula la inversa de la transpuesta de la matriz Model."""
    # Matriz Normal (matriz 3x3)
    normal_matrix = model_matrix[:3, :3]
    return np.transpose(np.linalg.inv(normal_matrix)).astype(np.float32)


def identity_matrix():
    return np.identity(4, dtype=np.float32)


def translate(matrix, vec):
    t_mat = np.identity(4, dtype=np.float32)
    t_mat[3][:3] = vec
    return np.dot(matrix, t_mat)


def scale(matrix, vec):
    s_mat = np.identity(4, dtype=np.float32)
    s_mat[0][0] = vec[0]
    s_mat[1][1] = vec[1]
    s_mat[2][2] = vec[2]
    return np.dot(matrix, s_mat)


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0][0] = f / aspect
    matrix[1][1] = f
    matrix[2][2] = (far + near) / (near - far)
    matrix[2][3] = -1.0
    matrix[3][2] = (2.0 * far * near) / (near - far)
    return matrix


def rotate_x(matrix, angle_deg):
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rot_mat = np.identity(4, dtype=np.float32)
    rot_mat[1][1] = c
    rot_mat[1][2] = s
    rot_mat[2][1] = -s
    rot_mat[2][2] = c
    return np.dot(matrix, rot_mat)


def calculate_orbit_view():
    global mouse_angle_h, mouse_angle_v, camera_distance

    pitch_rad = np.radians(mouse_angle_v)
    yaw_rad = np.radians(mouse_angle_h)

    cam_x = camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
    cam_y = camera_distance * math.sin(pitch_rad)
    cam_z = camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

    cam_pos = np.array([cam_x, cam_y + 1.0, cam_z], dtype=np.float32)

    center = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    f = center - cam_pos
    f = f / np.linalg.norm(f)

    s = np.cross(f, up)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    view = np.identity(4, dtype=np.float32)

    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f

    view[3, 0] = -np.dot(s, cam_pos)
    view[3, 1] = -np.dot(u, cam_pos)
    view[3, 2] = np.dot(f, cam_pos)

    return view


##D Utilidades

# --- Función para Cargar la Textura (CON CONVERSIÓN A RGB Y ALINEACIÓN DE BYTES) ---
def load_texture(path):
    """Carga la imagen y la configura como textura de OpenGL."""
    img = Image.open(path).convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(list(img.getdata()), np.uint8)
    width, height = img.size

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # Desactivar la alineación de 4 bytes (CRÍTICO para arreglar el ruido)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    if img.mode == 'RGBA':
        format = GL_RGBA
    else:
        format = GL_RGB

    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, img_data)

    glGenerateMipmap(GL_TEXTURE_2D)

    return texture_id


## E. Control de Estados (Sin cambios)
def change_state(new_state, current_time):
    global current_state, state_start_time
    current_state = new_state
    state_start_time = current_time
    print(f"Cambio de estado a: {new_state}")


# ====================================================================
# V. FUNCIÓN DE DIBUJO DEL CREEPER (ILUMINACIÓN INTEGRADA)
# ====================================================================

def draw_creeper(animation_time, global_scale_factor=1.0):
    global head_vao, torso_vao, leg_vao, head_texture_id, torso_texture_id, leg_texture_id

    # --- 1. UBICACIONES DE UNIFORMS ---
    proj_loc = glGetUniformLocation(shader_program, "projection")
    view_loc = glGetUniformLocation(shader_program, "view")
    model_loc = glGetUniformLocation(shader_program, "model")
    color_loc = glGetUniformLocation(shader_program, "objectColor")
    currentPart_loc = glGetUniformLocation(shader_program, "currentPart")

    # --- UNIFORMS DE ILUMINACIÓN ---
    normalMatrix_loc = glGetUniformLocation(shader_program, "normalMatrix")
    lightDir_loc = glGetUniformLocation(shader_program, "lightDirection")
    ambientStrength_loc = glGetUniformLocation(shader_program, "ambientStrength")
    lightColor_loc = glGetUniformLocation(shader_program, "lightColor")

    # --- UNIFORMS DE TEXTURA ---
    headSampler_loc = glGetUniformLocation(shader_program, "headSampler")
    torsoSampler_loc = glGetUniformLocation(shader_program, "torsoSampler")
    legSampler_loc = glGetUniformLocation(shader_program, "legSampler")

    # --- 2. CONFIGURACIÓN INICIAL DE MATRICES Y LUZ ---
    aspect_ratio = 800.0 / 600.0
    proj_matrix = perspective(45.0, aspect_ratio, 0.1, 100.0)
    view_matrix = calculate_orbit_view()

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_matrix)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)

    # VALORES DE LA LUZ
    glUniform3f(lightDir_loc, 0.5, 1.0, 0.5)  # Luz Direccional (arriba/derecha)
    glUniform1f(ambientStrength_loc, 0.4)  # Luz ambiental suave (40%)
    glUniform3f(lightColor_loc, 1.0, 1.0, 1.0)  # Color de la luz (Blanco)

    # --- 3. ENLACE DE UNIDADES DE TEXTURA ---
    glUniform1i(headSampler_loc, 0)
    glUniform1i(torsoSampler_loc, 1)
    glUniform1i(legSampler_loc, 2)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, head_texture_id)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, torso_texture_id)

    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, leg_texture_id)

    glActiveTexture(GL_TEXTURE0)

    # ====================================================
    # 0. PISO / PLANO (COLOR SÓLIDO)
    # ====================================================

    glBindVertexArray(head_vao)
    glUniform1i(currentPart_loc, -1)  # Deshabilita la textura en el shader
    glUniform3f(color_loc, 0.3, 0.6, 0.2)  # Color del piso

    Floor_Model = identity_matrix()
    Floor_Model = translate(Floor_Model, np.array([0.0, 0.0, 0.0]))
    Floor_Model = scale(Floor_Model, np.array([100.0, 0.01, 100.0]))

    # Enviamos la Matriz Normal (el piso solo tiene traslación y escalado uniforme, por lo que usamos la matriz modelo)
    glUniformMatrix4fv(normalMatrix_loc, 1, GL_FALSE, get_normal_matrix(Floor_Model))

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, Floor_Model)
    glDrawArrays(GL_TRIANGLES, 0, 36)

    # ====================================================
    # RESTAURAR EL MODO TEXTURA PARA EL CREEPER
    # ====================================================
    glUniform3f(color_loc, 1.0, 1.0, 1.0)  # Color de control (blanco puro)

    # ----------------------------------------------------
    # 1. TORSO (USANDO torso_vao y currentPart = 1)
    # ----------------------------------------------------
    glBindVertexArray(torso_vao)
    glUniform1i(currentPart_loc, 1)  # PIEZA 1: TORSO

    Torso_Model = identity_matrix()
    Torso_Model = translate(Torso_Model, np.array([0.0, 0.35, 0.0]))
    Torso_Model = scale(Torso_Model, np.array([global_scale_factor, global_scale_factor, global_scale_factor]))

    Torso_Matrix = Torso_Model.copy()
    Torso_Matrix = scale(Torso_Matrix, np.array([1.0, 2.5, 0.7]))
    Torso_Matrix = translate(Torso_Matrix, np.array([0.0, 1.25, 0.0]))

    # Enviamos la Matriz Normal del Torso
    glUniformMatrix4fv(normalMatrix_loc, 1, GL_FALSE, get_normal_matrix(Torso_Matrix))

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, Torso_Matrix)
    glDrawArrays(GL_TRIANGLES, 0, 36)

    # ----------------------------------------------------
    # 2. CABEZA (USANDO head_vao y currentPart = 0)
    # ----------------------------------------------------
    glBindVertexArray(head_vao)
    glUniform1i(currentPart_loc, 0)  # PIEZA 0: CABEZA

    Head_Model = Torso_Model.copy()
    Head_Model = translate(Head_Model, np.array([0.0, 2.5, 0.0]))  # <--- CORRECCIÓN DE ALTURA
    Head_Model = scale(Head_Model, np.array([1.2, 1.2, 1.2]))

    # Enviamos la Matriz Normal de la Cabeza
    glUniformMatrix4fv(normalMatrix_loc, 1, GL_FALSE, get_normal_matrix(Head_Model))

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, Head_Model)
    glDrawArrays(GL_TRIANGLES, 0, 36)

    # ----------------------------------------------------
    # 3. PIERNAS (USANDO leg_vao y currentPart = 2)
    # ----------------------------------------------------
    glBindVertexArray(leg_vao)
    glUniform1i(currentPart_loc, 2)  # PIEZA 2: PIERNAS

    lift_amplitude = 0.4

    leg_coords = [
        [-0.55, 0.8, 1.0, 0.0],
        [0.55, -0.8, 1.0, 0.0],
        [0.55, 0.8, -1.0, 0.2],
        [-0.55, -0.8, -1.0, 0.2],
    ]

    for x_pos, z_pos, multiplier, t_offset in leg_coords:
        Leg_Model = Torso_Model.copy()

        Leg_Model = translate(Leg_Model, np.array([x_pos, 0.0, z_pos]))

        sin_phase = math.sin((animation_time + t_offset) * 2.5 * multiplier)
        lift_factor = lift_amplitude * abs(sin_phase)

        Leg_Model = translate(Leg_Model, np.array([0.0, lift_factor, 0.0]))

        Leg_Model = scale(Leg_Model, np.array([0.6, 1.0, 0.5]))

        # Enviamos la Matriz Normal de la Pierna
        glUniformMatrix4fv(normalMatrix_loc, 1, GL_FALSE, get_normal_matrix(Leg_Model))

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, Leg_Model)
        glDrawArrays(GL_TRIANGLES, 0, 36)

    glBindVertexArray(0)


# ====================================================================
# VI. EL BUCLE PRINCIPAL (EL MOTOR DEL JUEGO)
# ====================================================================

def main_loop():
    global current_state, state_start_time

    glUseProgram(shader_program)
    glClearColor(0.5, 0.8, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)

    state_start_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        elapsed_time = current_time - state_start_time

        animation_time = 0.0
        global_scale_factor = 1.0

        if current_state == STATE_WALKING:
            animation_time = elapsed_time
            if elapsed_time > walk_duration:
                change_state(STATE_EXPLODING, current_time)

        elif current_state == STATE_EXPLODING:
            progress = min(elapsed_time / explosion_duration, 1.0)
            base_inflation = 1.0 + progress * 0.5
            pulse = 1.0 + 0.1 * math.sin(elapsed_time * 20.0)
            global_scale_factor = base_inflation * pulse

            if elapsed_time > explosion_duration:
                change_state(STATE_RESET, current_time)

        elif current_state == STATE_RESET:
            global_scale_factor = 1.0
            change_state(STATE_WALKING, current_time)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_creeper(animation_time, global_scale_factor)

        process_input(window)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


# ====================================================================
# VII. SETUP Y EJECUCIÓN (El Inicio)
# ====================================================================

def process_input(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def mouse_callback(window, xpos, ypos):
    global first_mouse, lastX, lastY, mouse_angle_h, mouse_angle_v

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = ypos - lastY
    lastX = xpos
    lastY = ypos

    sensitivity = 0.2

    mouse_angle_h += xoffset * sensitivity
    mouse_angle_v += yoffset * sensitivity

    if mouse_angle_v > 89.0:
        mouse_angle_v = 89.0
    if mouse_angle_v < -89.0:
        mouse_angle_v = -89.0


def scroll_callback(window, xoffset, yoffset):
    global camera_distance
    camera_distance -= yoffset

    if camera_distance < 2.0:
        camera_distance = 2.0
    if camera_distance > 15.0:
        camera_distance = 15.0


def initialize():
    global window, shader_program, head_vao, torso_vao, leg_vao
    global head_texture_id, torso_texture_id, leg_texture_id

    if not glfw.init(): return False

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Creeper Animado", None, None)
    if not window: glfw.terminate(); return False

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    initialize_all_vaos(head_vertices, torso_vertices, leg_vertices)

    try:
        shader_program = create_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
    except RuntimeError:
        glfw.terminate()
        return False

    # Carga de Textura
    try:
        head_texture_id = load_texture("creeper_head.jpg")
        torso_texture_id = load_texture("creeper_torso.jpg")
        leg_texture_id = load_texture("creeper_leg.jpg")

        print("Texturas cargadas exitosamente (Head, Torso, Leg).")
    except FileNotFoundError:
        print(
            "ERROR: Asegúrate de tener los archivos 'creeper_head.jpg', 'creeper_torso.jpg', y 'creeper_leg.jpg' en la misma carpeta.")
        glfw.terminate()
        return False

    return True


if __name__ == "__main__":
    if initialize():
        main_loop()