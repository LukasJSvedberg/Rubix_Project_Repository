import pygame
from pygame.locals import *
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import itertools
import copy


import glfw
import OpenGL.GL.shaders
from OpenGL.raw.GLU import gluLookAt
from pyrr import matrix44, Vector3, Matrix44
import keyboard



def window_resize(window, width, height):
    glViewport(0, 0, width, height)

mouse_x, mouse_y = 0, 0
changed_color_list = np.array([0, 0, 0])
pick = False
current_color = (1, 1, 1)

def cursor_pos_callback(window, xpos, ypos):
    global mouse_x, mouse_y
    mouse_x = xpos
    mouse_y = ypos

def conv_numb_to_pos(current_pos, pos, color):
    i = (pos - 1)// 9
    if i == 0:
        i = 3
    elif i == 1:
        i = 0
    elif i == 2:
        i = 1
    elif i == 3:
        i = 5
    elif i == 4:
        i = 2
    elif i == 5:
        i = 4
    k = (pos - 1) % 3
    j =  2 - ((pos - 1) % 9 // 3)
    current_pos[i][j][k] = color
    return current_pos

def picker(current_pos):
    global changed_color_list, pick

    data = glReadPixels(mouse_x, 864 - mouse_y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
    no = data[0]
    changed_color_list[0] = data[0]
    changed_color_list[1] = data[1]
    pick = False
    if changed_color_list[0] != 0 and changed_color_list[0] != 255:
        return conv_numb_to_pos(current_pos, data[0], convert_to_int(current_color))

    return current_pos



def mouse_button_callback(window, button, action, mods):
    global pick
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        pick = True


def convert_to_int(tuple):
    if tuple == (0.09803921568627451, 0.6078431372549019, 0.2980392156862745):
        return 1
    elif tuple == (0.5372549019607843, 0.07058823529411765, 0.0784313725490196):
        return 2
    elif tuple == (1, 1, 1):
        return 5
    elif tuple == (1, 0.3333333333333333, 0.1450980392156863):
        return 4
    elif tuple == (0.996078431372549, 0.9529411764705882, 0.06666666666666667):
        return 3
    elif tuple ==(0.050980392156862744, 0.2823529411764706, 0.6745098039215687):
        return 6
current_position = np.array([[[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 2, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 3, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 4, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 5, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 6, 0],
                                  [0, 0, 0]]])
def main(current_position):

    global current_color
    # initialize glfw
    if not glfw.init():
        return

    w_width, w_height = 1536, 864
    aspect_ratio = w_width / w_height

    window = glfw.create_window(w_width, w_height, "My OpenGL window", None, None)
    current_color = (1, 1, 1)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)

    #        positions
    cube = [-0.15, -0.15,  0.15,
             0.15, -0.15,  0.15,
             0.15,  0.15,  0.15,
            -0.15,  0.15,  0.15 ]

    cube = np.array(cube, dtype=np.float32)

    indices = [ 0,  1,  2,  2,  3,  0]

    indices = np.array(indices, dtype=np.uint32)

    vertex_shader = """
    #version 330
    in layout(location = 0) vec3 position;

    uniform mat4 vp;
    uniform mat4 model;

    void main()
    {
        gl_Position =  vp * model * vec4(position, 1.0f);
    }
    """

    fragment_shader = """
    #version 330

    out vec4 outColor;
    uniform vec3 color;

    void main()
    {
        outColor = vec4(color, 1.0);
    }
    """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # vertex buffer object and element buffer object
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    #position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    ######################################################################################

    # picking texture and a frame buffer object
    pick_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, pick_texture)

    FBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1536, 864, 0, GL_RGB, GL_FLOAT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pick_texture, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)

    ######################################################################################

    glUseProgram(shader)

    glEnable(GL_DEPTH_TEST)

    view = matrix44.create_from_translation(Vector3([0.0, 0.0, -4.0]))
    projection = matrix44.create_perspective_projection_matrix(45.0, aspect_ratio, 0.1, 100.0)

    vp = matrix44.multiply(view, projection)

    vp_loc = glGetUniformLocation(shader, "vp")
    model_loc = glGetUniformLocation(shader, "model")
    color_loc = glGetUniformLocation(shader, "color")
    cube_colour_order = [(1, 85/255, 37/255), (25/255, 155/255, 76/255), (137/255, 18/255, 20/255), (13/255, 72/255, 172/255), (254/255, 243/255, 17/255), (1, 1, 1)]
    cube_positions = []
    cube_colors = []
    pick_colors = []
    check_var = True
    for k in range(4):
        for i in range(3):
            for j in range(3):
                cube_positions.append((-2.0 + i * 0.34 + k * 1.08, 0.34 - j * 0.34, 0.0))
    for k in range(2):
        for i in range(3):
            for j in range(3):
                cube_positions.append((-0.92 + i * 0.34, -0.74 - j * 0.34 + k * 2.16, 0.0))

    for k in range(6):
        for i in range(9):
            if i % 9 != 4:
                cube_colors.append((0.2824, 0.2863, 0.2941))
            else:
                cube_colors.append(cube_colour_order[k])
    for i in range(54):
        pick_colors.append((i / 255 + 1 / 255, 0, 0))
    for i in range(6):
        cube_positions.append((2.4, 1.25 - 0.5 * i, 0))
        cube_colors.append(cube_colour_order[i])
        pick_colors.append((1, i / 255, 0))

    glUniformMatrix4fv(vp_loc, 1, GL_FALSE, vp)



    # [...]

    run = True

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.2, 0.3, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_y = Matrix44.from_y_rotation(glfw.get_time() * 2)

        # draw to the default frame buffer
        for i in range(len(cube_positions)):
            model = matrix44.create_from_translation(cube_positions[i])
            glUniform3fv(color_loc, 1, cube_colors[i])
            if changed_color_list[0] and changed_color_list[0] != 255 and (changed_color_list[0] - 1) % 9 != 4:
                cube_colors[(changed_color_list[0] - 1)] = current_color
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            elif changed_color_list[0] == 255:
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                current_color = cube_colour_order[changed_color_list[1]]

            else:
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

            """if i == 0:
                if changed_color_list[0]:
                    cube_colors[changed_color_list[0]] = (1.0, 0.0, 0.0)
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                else:
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)"""

            """elif i == 1:
                if changed_color_list[1]:
                    cube_colors[1] = (1.0, 0.0, 0.0)
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                else:
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            elif i == 2:
                if changed_color_list[2]:
                    cube_colors[2] = (1.0, 0.0, 0.0)
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                else:
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            else:
                if changed_color_list[2]:
                    cube_colors[3] = (1.0, 0.0, 0.0)
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
                else:
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)"""

            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        # draw to the custom frame buffer object
        glBindFramebuffer(GL_FRAMEBUFFER, FBO)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if keyboard.is_pressed('left') and check_var == True:
            check_var = False
        elif keyboard.is_pressed('left') == False and check_var == False:
            check_var = True
        elif keyboard.is_pressed('enter'):
            check = True

            for element in current_position.flatten():
                if element == 0:
                    check = False

            if check == True:
                glfw.hide_window(window)
                return current_position

        for i in range(len(cube_positions)):
            pick_model = matrix44.create_from_translation(cube_positions[i])
            glUniform3fv(color_loc, 1, pick_colors[i])
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, pick_model)
            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        if pick:
            current_position = picker(current_position)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    current_position = main(current_position)


class miniCubes:
    def __init__(self, x_pos, y_pos, z_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.pos_array = np.array([x_pos, y_pos, z_pos])
        self.default_pos = np.array([x_pos, y_pos, z_pos])
        self.vertex_set = np.array([[1. + x_pos, -1. + y_pos, -1. + z_pos], [1. + x_pos, 1. + y_pos, -1. + z_pos],
                           [-1. + x_pos, 1. + y_pos, -1. + z_pos], [-1. + x_pos, -1. + y_pos, -1. + z_pos],
                           [1. + x_pos, -1. + y_pos, 1. + z_pos], [1. + x_pos, 1. + y_pos, 1. + z_pos],
                           [-1. + x_pos, -1. + y_pos, 1. + z_pos], [-1. + x_pos, 1. + y_pos, 1. + z_pos]])
        self.edge_set = ((0, 1),
         (0, 3),
         (0, 4),
         (1, 2),
         (1, 5),
         (2, 3),
         (2, 7),
         (3, 6),
         (4, 5),
         (4, 6),
         (5, 7),
         (6, 7)
         )
        self.surface = ((0, 1, 2, 3),
             (0, 1, 5, 4),
             (0, 4, 6, 3),
             (7, 6, 4, 5),
             (7, 5, 1, 2),
             (7, 2, 3, 6))

        self.colors = np.array([[1,1,0], [0,0,1], [1,0,0], [1,1,1], [159/255,43/255, 104/255], [0,1,0]])
        self.colors_tuple = ((1,1,0), (0,0,1), (1,0,0), (1,1,1), (159/255,43/255, 104/255), (0,1,0))
        self.colors_list = [[1,1,0], [0,0,1], [1,0,0], [1,1,1], [159/255,43/255, 104/255], [0,1,0]]
        self.colors_in_cube = []
        self.number_of_colours = []
        self.check_colour()
        self.WHITE = ((1, 1, 1))
        self.BLUE = ((0, 1, 1))
        self.ORANGE = ((1, 95/255, 31/255))
        self.GREEN = ((0, 1, 0))
        self.YELLOW = ((1, 1, 0))
        self.RED = ((1, 0, 0))

    def calculate_colour(self, default_pos):

        if abs(default_pos[2]) == 2:
            if abs(default_pos[0]) and abs(default_pos[1]) == 2:
                pass




    def find_place_in_list(self, tuple_pos):
        if tuple_pos[0] == 3:
            return 1
        elif tuple_pos[0] == -3:
            return 5
        elif tuple_pos[1] == 3:
            return 4
        elif tuple_pos[1] == -3:
            return 2
        elif tuple_pos[2] == 3:
            return 3
        elif tuple_pos[2] == -3:
            return 0

    def check_colour(self):
        self.color_pos_in_cube = {}
        i = 0
        all_surface_positions = {}
        all_surfaces_positions_list = []
        for surface in surfaces:
            surface_position = [0, 0, 0]
            isColoured = 0
            for vertex in surface:
                vertex_tested = self.vertex_set[vertex]
                for element in range(len(vertex_tested)):
                    if abs(round(vertex_tested[element], 1)) == 3:
                        isColoured += 1
                        surface_position[0] += vertex_tested[0]
                        surface_position[1] += vertex_tested[1]
                        surface_position[2] += vertex_tested[2]
                        break

            if isColoured != 4:
                self.colors[i] = [0, 0, 0]
            else:
                for element in range(len(surface_position)):
                    surface_position[element] /= 4
                tuple_surface_position = tuple(surface_position)
                all_surface_positions.update({tuple_surface_position: self.colors_tuple[i]})
                all_surfaces_positions_list.append(tuple_surface_position)


            i += 1
        for surface in surfaces:
            surface_pos = [0, 0, 0]
            counter = 0
            for vertex in surface:
                vertex_tested = self.vertex_set[vertex]
                surface_pos[0] += vertex_tested[0]
                surface_pos[1] += vertex_tested[1]
                surface_pos[2] += vertex_tested[2]
                for k in range(3):
                    if abs(round(vertex_tested[k], 1)) == 3:
                        counter += 1
                        break
            for element in range(3):
                surface_pos[element] /= 4
            surface_pos = tuple(surface_pos)
            if counter == 4:
                for element in all_surfaces_positions_list:
                    if element == surface_pos:
                        color = round(self.assign_colours(surface_pos))
                        color_in_list_place = self.find_place_in_list(surface_pos)
                        act_color = self.color_pos_in_list(color)
                        """ """
                        self.colors[color_in_list_place] = all_colors_rgb_values[act_color]





        for i in range(len(all_colors_rgb_values)):
            right_elements = 0
            for j in range(3):
                if all_colors_rgb_values[i][j] == 0:
                    right_elements += 1
            if right_elements != 3:
                self.colors_in_cube.append(all_colors_string_name[i])
                self.number_of_colours.append(all_color_numb_values[i])
                self.surface_colors_dict = all_surface_positions


    def calc_piece(self):
        i = 0
        counter = False
        all_surface_positions_list = np.array([])
        colors_np_array = np.array(
            [[1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [159 / 255, 43 / 255, 104 / 255], [0, 1, 0]])
        for surface in surfaces:
            surface_position = np.array([0, 0, 0])
            isColoured = 0
            for vertex in surface:
                vertex_tested = self.vertex_set[vertex]
                for element in (vertex_tested):
                    if abs(round(element, 1)) == 3:
                        isColoured += 1
                        surface_position[0] += round(vertex_tested[0])
                        surface_position[1] += round(vertex_tested[1])
                        surface_position[2] += round(vertex_tested[2])
                        break
            if isColoured == 4:
                for element in range(3):
                    surface_position[element] /= 4
                color = (self.assign_colours(surface_position))
                act_color = self.color_pos_in_list(color)

                surface_position = np.vstack([np.expand_dims(surface_position, 0), np.expand_dims(colors_np_array[act_color], 0)])
                """surface_position = np.vstack(
                    [np.expand_dims(surface_position, 0), np.expand_dims(self.colors[i], 0)])"""
                if counter == False:
                    all_surface_positions_list = np.expand_dims(surface_position, 0)
                    counter = True
                else:
                    all_surface_positions_list = np.vstack([all_surface_positions_list, np.expand_dims(surface_position, 0)])

            i += 1
        all_surface_positions_list = np.array(all_surface_positions_list)

        return all_surface_positions_list


    def assign_colours(self, face_coodinates):
        x = round(face_coodinates[0])
        y = round(face_coodinates[1])
        z = round(face_coodinates[2])
        colour_of_surface = 0
        if face_coodinates[2] == 3:
            colour_of_surface = current_position[0, round(1 - (y / 2)), round(1 + (x / 2))]
        elif face_coodinates[2] == -3:
            colour_of_surface = current_position[5, round(1 + (y / 2)), round(1 + (x / 2))]
        elif face_coodinates[0] == 3:
            colour_of_surface = current_position[2, round(1 - (y / 2)), round(1 - (z / 2))]
        elif face_coodinates[0] == -3:
            colour_of_surface = current_position[4, round(1 - (y / 2)), round(1 + (z / 2))]
        elif face_coodinates[1] == 3:
            colour_of_surface = current_position[1, round(1 + (z / 2)), round(1 + (x / 2))]
        elif face_coodinates[1] == -3:
            colour_of_surface = current_position[3, round(1 - (z / 2)), round(1 + (x / 2))]
        return colour_of_surface





    def render_cube(self):
        i = 0
        glBegin(GL_QUADS)
        for surface in surfaces:
            """surface_pos = [0, 0, 0]
            color = False
            for vertex in surface:
                surface_pos[0] += self.vertex_set[vertex][0]
                surface_pos[1] += self.vertex_set[vertex][1]
                surface_pos[2] += self.vertex_set[vertex][2]
            for element in range(3):
                surface_pos[element] /= 4
            surface_pos = tuple(surface_pos)
            for key in self.surface_colors_dict:
                if key == surface_pos:
                    color = True
                    glColor3fv(self.surface_colors_dict[key])"""
            glColor3fv(self.colors[i])
            """if color == False:
                glColor3fv((1, 0, 0))
"""

            i += 1
            for vertex in surface:
                glVertex3fv(self.vertex_set[vertex])

        glEnd()
        glLineWidth(5.0)
        glBegin(GL_LINES)
        for edge in edges:
            glColor3fv((0, 0, 0))
            for vertex in edge:
                glVertex3fv(self.vertex_set[vertex])

        """ def render_cube(self, colour_list):
        i = 0
        glBegin(GL_QUADS)
        for surface in surfaces:
            isColoured = 0
            for vertex in surface:
                vertex_tested = self.vertex_set[vertex]
                for element in vertex_tested:
                    if abs(round(element, 1)) == 3:
                        isColoured += 1
                        break

            if isColoured == 4:
                glColor3fv(colour_list[i])
            else:
                print(self.vertex_set)
                glColor3fv((0, 0, 0))
            i += 1
            for vertex in surface:
                glVertex3fv(self.vertex_set[vertex])

        glEnd()
        glLineWidth(5.0)
        glBegin(GL_LINES)
        for edge in edges:
            glColor3fv((0, 0, 0))
            for vertex in edge:
                glVertex3fv(self.vertex_set[vertex])"""

        glEnd()
        glLineWidth(1.0)
        """
        i = 0
        glBegin(GL_QUADS)
        for surface in surfaces:
            glColor3fv(self.RED)
            i += 1
            correct_vertices = [0, 0, 0, 0, 0, 0]
            for vertex in surface:
                vertex_judging = self.vertex_set[vertex]
                if vertex_judging[0] == -3:
                    correct_vertices[0] += 1
                if vertex_judging[0] == 3:
                    correct_vertices[1] += 1
                if vertex_judging[1] == -3:
                    correct_vertices[2] += 1
                if vertex_judging[1] == 3:
                    correct_vertices[3] += 1
                if vertex_judging[2] == -3:
                    correct_vertices[4] += 1
                if vertex_judging[2] == 3:
                    correct_vertices[5] += 1
            print(correct_vertices)
            for i in range(len(correct_vertices)):
                if correct_vertices[i] == 4:
                    for vertex in surface:
                        glVertex3fv(self.vertex_set[vertex])
                else:
                    for vertex in surface:
                        glVertex3fv(self.vertex_set[vertex])



        glEnd()
        glLineWidth(5.0)
        glBegin(GL_LINES)
        for edge in edges:
            glColor3fv((0, 0, 0))
            for vertex in edge:
                glVertex3fv(self.vertex_set[vertex])

        glEnd()
        glLineWidth(1.0)
        """
    def rotate(self, matrix):
        orientation = np.array([self.x_pos, self.y_pos, self.z_pos])

        i = 0
        for vertex in self.vertex_set:
            temp_vertex = vertex.copy()
            temp_vertex.reshape(1, 3)

            temp_vertex = np.matmul(matrix, temp_vertex)
            self.vertex_set[i] = temp_vertex.reshape(3)
            self.render_cube()
            i += 1
        orientation = np.matmul(matrix, orientation)
        orientation.reshape(3)
        self.x_pos = orientation[0]
        self.y_pos = orientation[1]
        self.z_pos = orientation[2]
        self.pos_array = orientation.copy()
    def update_coords(self):

        orientation = np.array([self.x_pos, self.y_pos, self.z_pos])
        for vertex in self.vertex_set:
            temp_vertex = vertex.copy()
            for element in temp_vertex:
                element = round(element)
        self.x_pos, self.y_pos, self.z_pos = round(orientation[0]), round(orientation[1]), round(orientation[2])

    def color_pos_in_list(self, int):
        if int == 1:
            return 3
        elif int == 2:
            return 4
        elif int == 3:
            return 1
        elif int == 4:
            return 2
        elif int == 5:
            return 5
        elif int == 6:
            return 0









vertices = [[1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, 1]]
edges = ((0, 1),
         (0, 3),
         (0, 4),
         (1, 2),
         (1, 5),
         (2, 3),
         (2, 7),
         (3, 6),
         (4, 5),
         (4, 6),
         (5, 7),
         (6, 7)
         )

surfaces = ((0, 1, 2, 3),
           (0, 1, 5, 4),
           (0, 4, 6, 3),
           (7, 6, 4, 5),
           (7, 5, 1, 2),
           (7, 2, 3, 6))

"""current_position = np.array([[[6, 5, 5],
       [6, 1, 1],
       [4, 1, 5]],
      [[1, 6, 2],
       [5, 2, 4],
       [3, 4, 6]],
        [[2, 1, 1],
       [3, 3, 3],
       [4, 5, 3]],
       [[1, 2, 6],
       [6, 4, 2],
       [1, 5, 6]],
       [[4, 6, 4],
       [3, 5, 4],
       [2, 3, 3]],
       [[5, 1, 2],
       [4, 6, 2],
       [5, 2, 3]]])"""
"""current_position = np.array([[[1, 6, 6],
  [6, 1, 6],
  [1, 1, 6]],

 [[4, 4, 4],
  [4, 2, 2],
  [2, 2, 2]],

 [[5, 3, 3],
  [5, 3, 5],
  [3, 5, 3]],

 [[4, 2, 2],
  [4, 4, 2],
  [4, 4, 2]],

 [[3, 3, 5],
  [3, 5, 3],
  [5, 5, 5]],

 [[6, 6, 1],
  [1, 6, 1],
  [6, 1, 1]]])"""


"""np.array([[[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]],
       [[2, 2, 2],
       [2, 2, 2],
       [2, 2, 2]],
        [[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]],
        [[4, 4, 4],
       [4, 4, 4],
       [4, 4, 4]],
        [[5, 5, 5],
       [5, 5, 5],
       [5, 5, 5]],
      [[6, 6, 6],
       [6, 6, 6],
       [6, 6, 6]]])"""


answer_string = ('L', 'U2', "R'", "F'", 'R', 'B', "L'", 'B', 'L')
reverse_list = ('L', 'B', 'L\'', 'B', 'R', 'F\'', 'R\'', 'U2', 'L')

"""np.array([[[6, 3, 3],
  [2, 1, 1],
  [5, 5, 1]],

 [[4, 6, 6],
  [4, 2, 2],
  [5, 2, 2]],

 [[6, 6, 4],
  [3, 3, 3],
  [2, 1, 1]],

 [[2, 2, 3],
  [4, 4, 4],
  [4, 4, 4]],

 [[5, 5, 2],
  [5, 5, 1],
  [6, 6, 1]],

 [[3, 3, 3],
  [1, 6, 6],
  [1, 5, 5]]])"""
all_possible_moves = np.array(['L', 'L\'', 'R', 'R\'', 'U', 'U\'', 'U2', 'D', 'D\'', 'D2', 'F', 'F\'', 'F2', 'B', 'B\'', 'B2'])
stage_one_moves = np.array(['L', 'L\'', 'R', 'R\'', 'U', 'U\'', 'D', 'D\'', 'F', 'F\'', 'B', 'B\''])
stage_two_moves = np.array(['L', 'L\'', 'R', 'R\'', 'U2', 'D2', 'F', 'F\'', 'B', 'B\''])
stage_three_moves = np.array(['L', 'L\'', 'R', 'R\'', 'U2', 'D2', 'F2', 'B2'])
stage_four_moves = np.array(['L2', 'R2', 'U2', 'D2', 'F2', 'B2'])
forward_set = {}
backward_set = {}

move_list = np.array([])
all_permutation_list = np.array([['D', 'D']])
different_types_of_moves = ((0, 3), (4, 9), (10, 15))
corner_pieces = np.array([])
"""current_position = np.array([[[101, 201, 301],
                     [401, 501, 601],
                     [701, 801, 901]],
                     [[102, 202, 302],
                      [402, 502, 602],
                      [702, 802, 902]],
                     [[103, 203, 303],
                      [403, 503, 603],
                      [703, 803, 903]],
                     [[104, 204, 304],
                      [404, 504, 604],
                      [704, 804, 904]],
                     [[105, 205, 305],
                      [405, 505, 605],
                      [705, 805, 905]],
                     [[106, 206, 306],
                      [406, 506, 606],
                      [706, 806, 906]]])"""

turning_faces = (((1, 2, 6, 4), 5), ((1, 2, 6, 4), 3), ((1, 3, 6, 5), 2), ((1, 3, 6, 5), 4), ((2, 3, 4, 5), 1), ((2, 3, 4, 5), 6))
depth = 6
array_for_phase_1_center = []

visible = False
object_list = []

move_up = False
move_right = False
move_down = False
move_left = False
move_z_1 = False
move_z_2 = False
stage_one_possible_grouped_moves = (('L', 'L\'', 'R', 'R\''), ('U', 'U\'', 'D', 'D\''), ('F', 'F\'', 'B', 'B\''))
stage_two_possible_grouped_moves = (('L', 'L\'', 'R', 'R\''), ('U2', 'D2'), ('F', 'F\'', 'B', 'B\''))
stage_three_possible_grouped_moves = (('L', 'L\'', 'R', 'R\''), ('U2', 'D2'), ('F2', 'B2'))
stage_four_possible_grouped_moves = (('L2', 'R2'), ('U2', 'D2'), ('F2', 'B2'))

def pos_neg(numb):
    if numb != 0:
        return abs(numb) / numb
    else:
        return 0
def rotate_matrix(direction, matrix):
    if direction == 1:
        matrix = np.roll(matrix, 3)
    elif direction == -1:
        matrix = np.roll(matrix, -3)
    elif direction == 2:
        matrix = np.roll(matrix, 6)
    return matrix
def calculate_rotation(face, direction, rotation_direction, left_right, matrix):
    all_elements = np.array([])
    opp_left_right = 0
    if left_right == 0:
        opp_left_right = 2
    for i in range(4):
        temp = turning_faces[face][0][i]
        temp_array = np.array([])
        for j in range(3):
            if face <= 1:
                temp_array = np.append(temp_array, matrix[temp - 1][j][left_right])
            elif face <= 3:
                if temp == 6:
                    temp_array = np.append(temp_array, matrix[temp - 1][opp_left_right][j])
                else:
                    temp_array = np.append(temp_array, matrix[temp - 1][left_right][j])




                """elif temp == 3:
                                    temp_array = np.append(temp_array, matrix[temp - 1][2 - j][left_right])
                                elif temp == 5:
                                    temp_array = np.append(temp_array, matrix[temp - 1][2 - j][opp_left_right])"""
            elif face <= 5:
                if temp == 2:
                    temp_array = np.append(temp_array, matrix[temp - 1][opp_left_right][j])
                elif temp == 4:
                    temp_array = np.append(temp_array, matrix[temp - 1][left_right][j])
                elif temp == 3:
                    temp_array = np.append(temp_array, matrix[temp - 1][j][left_right])

                else:
                    temp_array = np.append(temp_array, matrix[temp - 1][j][opp_left_right])



        if i == 0:
            all_elements = np.expand_dims(temp_array, 0)
        else:
            all_elements = np.append(all_elements, temp_array)

    all_elements = rotate_matrix(direction, all_elements)

    for i in range(4):
        for j in range(3):
            temp = turning_faces[face][0][i]

            if face <= 1:
                matrix[temp - 1][j][left_right] = round(all_elements[3 * i + j])
            elif face <= 3:
                element = j

                if direction == 1:
                    element = 2 - j
                if temp == 1:
                    matrix[temp - 1][left_right][j] = round(all_elements[3 * i + j])
                elif temp == 6:
                    matrix[temp - 1][opp_left_right][2 - j] = round(all_elements[3 * i + j])
                elif temp == 3:
                    matrix[temp - 1][left_right][2 - element] = round(all_elements[3 * i + j])
                elif temp == 5:
                    matrix[temp - 1][left_right][element] = round(all_elements[3 * i + j])

            elif face <= 5:
                element = j
                if direction == -1:
                    element = 2 - j
                if temp == 2:
                    matrix[temp - 1][opp_left_right][2 - element] = round(all_elements[3 * i + j])
                elif temp == 4:
                    matrix[temp - 1][left_right][2 - element] = round(all_elements[3 * i + j])
                elif temp == 3:
                    matrix[temp - 1][element][left_right] = round(all_elements[3 * i + j])
                else:
                    matrix[temp - 1][element][opp_left_right] = round(all_elements[3 * i + j])
    rotating_face = turning_faces[face][1] - 1
    face_5 = np.array(matrix[rotating_face]).T

    if rotation_direction == 1:
        matrix[rotating_face] = np.flipud(face_5)
    elif rotation_direction == -1:
        matrix[rotating_face] = np.fliplr(face_5)
    elif rotation_direction == 2:
        face_5 = np.flipud(face_5)
        matrix[rotating_face] = np.flipud(face_5)

    """for element in matrix:
        for element_1 in element:
            print(element_1)
        print()"""
    return matrix



def move_type(current, move):
    temp_rubix_pos = current
    if move == 'L':
        temp_rubix_pos = calculate_rotation(0, -1, -1, 0, current)
    elif move == 'L\'':
        temp_rubix_pos = calculate_rotation(0, 1, 1, 0, current)
    elif move == 'L2':
        temp_rubix_pos = calculate_rotation(0, 1, 1, 0, current)
        temp_rubix_pos = calculate_rotation(0, 1, 1, 0, temp_rubix_pos)
    elif move == 'R':
        temp_rubix_pos = calculate_rotation(1, 1, -1, 2, current)
    elif move == 'R\'':
        temp_rubix_pos = calculate_rotation(1, -1, 1, 2, current)
    elif move == 'R2':
        temp_rubix_pos = calculate_rotation(1, 1, -1, 2, current)
        temp_rubix_pos = calculate_rotation(1, 1, -1, 2, temp_rubix_pos)
    elif move == 'U':
        temp_rubix_pos = calculate_rotation(2, -1, -1, 0, current)
    elif move == 'U\'':
        temp_rubix_pos = calculate_rotation(2, 1, 1, 0, current)
    elif move == 'U2':
        temp_rubix_pos = calculate_rotation(2, -1, -1, 0, current)
        temp_rubix_pos = calculate_rotation(2, -1, -1, 0, temp_rubix_pos)
    elif move == 'D':
        temp_rubix_pos = calculate_rotation(3, 1, -1, 2, current)
    elif move == 'D\'':
        temp_rubix_pos = calculate_rotation(3, -1, 1, 2, current)
    elif move == 'D2':
        temp_rubix_pos = calculate_rotation(3, -1, 1, 2, current)
        temp_rubix_pos = calculate_rotation(3, -1, 1, 2, temp_rubix_pos)
    elif move == 'F':
        temp_rubix_pos = calculate_rotation(4, 1, -1, 0, current)
    elif move == 'F\'':
        temp_rubix_pos = calculate_rotation(4, -1, 1, 0, current)
    elif move == 'F2':
        temp_rubix_pos = calculate_rotation(4, -1, 1, 0, current)
        temp_rubix_pos = calculate_rotation(4, -1, 1, 0, temp_rubix_pos)
    elif move == 'B':
        temp_rubix_pos = calculate_rotation(5, -1, -1, 2, current)
    elif move == 'B\'':
        temp_rubix_pos = calculate_rotation(5, 1, 1, 2, current)
    elif move == 'B2':
        temp_rubix_pos = calculate_rotation(5, 1, 1, 2, current)
        temp_rubix_pos = calculate_rotation(5, 1, 1, 2, temp_rubix_pos)
    return temp_rubix_pos

def find_ideal_color_2(color_1, color_2, center_array):
    ideal_colors = np.array([])
    ideal_position = np.array([])
    final_ideal_position = np.array([])

    center_array_shape = center_array.shape
    counter = 0
    index_1 = 0
    val_1 = 0
    index_2 = 0
    val_2 = 0
    position_1 = np.array([])
    position_2 = np.array([])
    for i in range(center_array_shape[0]):
        if np.array_equal(center_array[i][1], color_1) or np.array_equal(center_array[i][1], color_2):
            temp_array = np.expand_dims(np.vstack([np.expand_dims(center_array[i][0], 0), np.expand_dims(center_array[i][1], 0)]), 0)
            if counter == 0:
                counter += 1
                ideal_position = temp_array
            else:
                ideal_position = np.vstack([ideal_position, temp_array])
    for i in range(3):
        if ideal_position[1][0][i] != 0:
            index_1 = i
            val_1 = ideal_position[1][0][i] * 2 / 3
    for i in range(3):
        if ideal_position[0][0][i] != 0:
            index_2 = i
            val_2 = ideal_position[0][0][i] * 2 / 3

    ideal_position[0][0][index_1] = val_1
    ideal_position[1][0][index_2] = val_2

    for i in range(2):
        if np.array_equal(color_1, ideal_position[i][1]):
            position_1 = ideal_position[i][0]
        elif np.array_equal(color_2, ideal_position[i][1]):
            position_2 = ideal_position[i][0]

    return position_1, position_2


def find_ideal_color_3(color_1, color_2, color_3, center_array):
    ideal_colors = np.array([])
    ideal_position = np.array([])
    final_ideal_position = np.array([])
    center_array_shape = center_array.shape
    counter = 0
    index_1 = 0
    index_2 = 0
    index_3 = 0
    val_1 = 0
    val_2 = 0
    val_3 = 0
    position_1 = np.array([])
    position_2 = np.array([])
    position_3 = np.array([])

    for i in range(center_array_shape[0]):
        if np.array_equal(center_array[i][1], color_1) or np.array_equal(center_array[i][1], color_2) or np.array_equal(center_array[i][1], color_3):
            temp_array = np.expand_dims(np.vstack([np.expand_dims(center_array[i][0], 0), np.expand_dims(center_array[i][1], 0)]),  0)
            if counter == 0:
                counter += 1
                ideal_position = temp_array
            else:
                ideal_position = np.vstack([ideal_position, temp_array])
    sum_of_pos = (ideal_position[0][0] + ideal_position[1][0] + ideal_position[2][0]) / 3 * 2
    for i in range(3):
        for j in range(3):
            if ideal_position[i][0][j] == 0:
                ideal_position[i][0][j] = sum_of_pos[j]

    for i in range(3):
        if np.array_equal(color_1, ideal_position[i][1]):
            position_1 = ideal_position[i][0]
        elif np.array_equal(color_2, ideal_position[i][1]):
            position_2 = ideal_position[i][0]
        elif np.array_equal(color_3, ideal_position[i][1]):
            position_3 = ideal_position[i][0]
    return position_1, position_2, position_3

def array_manipulation(array, center_or_edge):
    location_array = np.array([[], [], [], [], [], []])
    final_array = []
    final_array_2 = np.array([])
    array_shape = array.shape
    length_of_axis_0 = array_shape[0]
    length_of_axis_2 = array_shape[2]

    for i in range(0, length_of_axis_0, center_or_edge):
        if center_or_edge == 2:
            temp_array = np.vstack([np.expand_dims(array[i], 0), np.expand_dims(array[i + 1], 0)])
        else:
            temp_array = np.vstack([np.expand_dims(array[i], 0), np.expand_dims(array[i + 1], 0), np.expand_dims(array[i + 2], 0)])
        if i == 0:
            final_array = temp_array
            final_array = np.expand_dims(final_array, 0)
        else:
            temp_array = np.expand_dims(temp_array, 0)
            final_array = np.vstack([final_array, temp_array])

    final_array_shape = final_array.shape
    length_of_axis_0_final = final_array_shape[0]
    for i in range(length_of_axis_0_final):
        position_of_color_1 = np.expand_dims(final_array[i][0][0], 0)
        position_of_color_2 = np.expand_dims((final_array[i][1][0]), 0)
        position_of_color_3 = 0
        color_1_value = final_array[i][0][1]
        color_2_value = final_array[i][1][1]
        color_3_value = 0
        home_position_3 = 0
        if center_or_edge == 2:
            home_position_1, home_position_2 = find_ideal_color_2(color_1_value, color_2_value, array_for_phase_1_center)
        else:
            color_3_value = final_array[i][2][1]
            position_of_color_3 = np.expand_dims((final_array[i][2][0]), 0)
            home_position_1, home_position_2, home_position_3 = find_ideal_color_3(color_1_value, color_2_value, color_3_value, array_for_phase_1_center)
        color_1_value = np.expand_dims(color_1_value, 0)
        color_2_value = np.expand_dims(color_2_value, 0)
        home_position_1 = np.expand_dims(home_position_1, 0)
        home_position_2 = np.expand_dims(home_position_2, 0)
        if center_or_edge == 3:
            color_3_value = np.expand_dims(color_3_value, 0)
            home_position_3 = np.expand_dims(home_position_3, 0)
        """print(position_of_color_1.shape, color_1_value.shape, home_position_1.shape)
        print(position_of_color_2.shape, color_2_value.shape, home_position_2.shape)"""
        pos_1_full_array = np.expand_dims(np.vstack([position_of_color_1, color_1_value, home_position_1]), 0)
        pos_2_full_array = np.expand_dims(np.vstack([position_of_color_2, color_2_value, home_position_2]), 0)
        if center_or_edge == 2:
            all_edge_info = np.expand_dims(np.vstack([pos_1_full_array, pos_2_full_array]), 0)
            if i == 0:
                final_array_2 = all_edge_info
            else:
                final_array_2 = np.vstack([final_array_2, all_edge_info])
        else:
            pos_3_full_array = np.expand_dims(np.vstack([position_of_color_3, color_3_value, home_position_3]), 0)
            all_edge_info = np.expand_dims(np.vstack([pos_1_full_array, pos_2_full_array, pos_3_full_array]), 0)
            if i == 0:
                final_array_2 = all_edge_info
            else:
                final_array_2 = np.vstack([final_array_2, all_edge_info])
    return final_array_2

def convert_to_phase_1(edge_or_center):
    array_for_phase_1 = np.array([[]])
    i = 0
    for object in object_list:
        piece = object.calc_piece()
        if piece.shape[0] == edge_or_center:
            if i == 0:
                array_for_phase_1 = piece
            else:
                array_for_phase_1 = np.vstack([array_for_phase_1, piece])

            i += 1
    if edge_or_center == 2 or edge_or_center == 3:
        array_for_phase_1 = array_manipulation(array_for_phase_1, edge_or_center)

    return(array_for_phase_1)

def rotate_single_edge_up_y(edge_to_rotate, ideal_pos):
    original_edge = edge_to_rotate
    while abs(edge_to_rotate[1] - ideal_pos[1]) > 1:
        if edge_to_rotate[0] == 0:
            edge_to_rotate = np.array([edge_to_rotate[1], edge_to_rotate[0] + 0, edge_to_rotate[2]])
        else:
            edge_to_rotate = np.array([edge_to_rotate[1], -edge_to_rotate[0] + 0, edge_to_rotate[2]])


    edge_to_rotate = np.vstack([np.expand_dims(edge_to_rotate, 0), np.expand_dims(ideal_pos, 0)])
    return edge_to_rotate
def rotate_single_edge_up_x(edge_to_rotate):
    original_edge = edge_to_rotate
    edge_to_rotate = np.array([edge_to_rotate[0], edge_to_rotate[2], edge_to_rotate[1]])
    return edge_to_rotate

def how_many_moves(edge_to_rotate, ideal_pos, add_one_or_not):
    counter = 0
    error = False
    while abs(edge_to_rotate[0] - ideal_pos[0]) > 1 or abs(edge_to_rotate[2] - ideal_pos[2]) > 1:
        if edge_to_rotate[2] == 0:
            edge_to_rotate = np.array([edge_to_rotate[2], edge_to_rotate[1] + 0, edge_to_rotate[0]])
        else:
            edge_to_rotate = np.array([edge_to_rotate[2] * -1, edge_to_rotate[1] + 0, edge_to_rotate[0]])
        counter += 1
    if (edge_to_rotate[0] - ideal_pos[0] != 0 or edge_to_rotate[1] - ideal_pos[1] != 0 or edge_to_rotate[2] - ideal_pos[2] != 0) and add_one_or_not == 1:
        counter += 1
    return counter

def x_update(numb):
    if numb != 0:
        return 3 - numb
    return 0

def y_update(numb):
    if numb != 1:
        return 2 - numb
    return 1

def z_update(numb):
    if numb != 2:
        return 1 - numb
    return 2

def calc_numb_of_moves(edge_info, corner_or_edge):
    first_edge_pos = edge_info[0][0]
    if edge_info[0][0][2] != 0:
        first_edge_pos = edge_info[0][0]
    else:
        first_edge_pos = rotate_single_edge_up_x(edge_info[0][0])
    first_ideal_pos = edge_info[0][2]
    edges = rotate_single_edge_up_y(first_edge_pos, first_ideal_pos)
    move_number = how_many_moves(edges[0], edges[1], corner_or_edge)
    return move_number

def calc_numb_of_moves_2_corner(corner_pieces, parity_numb):
    for i in range(len(corner_pieces)):
        if parity_numb == 1:
            if np.array_equal(corner_pieces[i][1], np.array([0, 1, 0])) or np.array_equal(corner_pieces[i][1], np.array([0, 0, 1])):
                if abs(corner_pieces[i][0][0]) == 3:
                    return 0, i
                elif abs(corner_pieces[i][0][1]) == 3:
                    return 1, i
                elif abs(corner_pieces[i][0][2]) == 3:
                    return 2, i

        else:
            if np.array_equal(corner_pieces[i][1], np.array([0, 1, 0])) == False and np.array_equal(corner_pieces[i][1], np.array([0, 0, 1])) == False and abs(corner_pieces[i][0][2]) == 3:
                if abs(corner_pieces[i][0][0]) == 3:
                    return 0, i
                elif abs(corner_pieces[i][0][1]) == 3:
                    return 1, i
                elif abs(corner_pieces[i][0][2]) == 3:
                    return 2, i

def calc_numb_of_moves_2_edge(edge_piece):
    counter = 0
    for i in range(len(edge_piece)):
        if np.array_equal(edge_piece[i][1], np.array([0, 1, 0])) == False and np.array_equal(edge_piece[i][1], np.array([0, 0, 1])) == False:
            counter += 1
    if counter == 2:
        return 1
    else:
        return 0



def check_phase_2(corner_pieces, edge_pieces):
    array_of_moves_stage_2 = np.array([])
    for i in range(8):
        temp, location = calc_numb_of_moves_2_corner(corner_pieces[i], 1)
        temp_array = np.vstack([corner_pieces[i][location], np.array([[temp, temp, temp]])])
        if i == 0:
            array_of_moves_stage_2 = np.expand_dims(temp_array, 0)
        else:
            array_of_moves_stage_2 = np.vstack([array_of_moves_stage_2, np.expand_dims(temp_array, 0)])
    for i in range(12):

        is_middle = calc_numb_of_moves_2_edge(edge_pieces[i])
        temp_array = np.vstack([edge_pieces[i][0], np.array([[is_middle, is_middle, is_middle]])])
        array_of_moves_stage_2 = np.vstack([array_of_moves_stage_2, np.expand_dims(temp_array, 0)])

    return array_of_moves_stage_2

def check_phase_3(corner_pieces, edge_pieces, parity):
    array_of_moves_stage_2_corner = np.array([])
    array_of_moves_stage_2_edge = np.array([])
    edges_appended = 0
    for i in range(8):
        temp, location = calc_numb_of_moves_2_corner(corner_pieces[i], 0)
        temp_array_1 = np.array(corner_pieces[i][location])
        temp, location = calc_numb_of_moves_2_corner(corner_pieces[i], 1)
        temp_array_2 = np.array(corner_pieces[i][location])
        temp_array = np.vstack([np.expand_dims(temp_array_1, 0), np.expand_dims(temp_array_2, 0)])
        if i == 0:
            array_of_moves_stage_2_corner = np.expand_dims(temp_array, 0)
        else:
            array_of_moves_stage_2_corner = np.vstack([array_of_moves_stage_2_corner, np.expand_dims(temp_array, 0)])
    for x, element in enumerate(edge_pieces):
        temp_array = np.array([])
        for el in element:
            if abs(el[0][0]) != 3:
                temp_array = el
        if edges_appended == 0:
            array_of_moves_stage_2_edge = np.expand_dims(temp_array, 0)
            edges_appended += 1
        else:
                array_of_moves_stage_2_edge = np.vstack([array_of_moves_stage_2_edge, np.expand_dims(temp_array, 0)])
    return array_of_moves_stage_2_corner, array_of_moves_stage_2_edge

def check_phase_1(edge_pieces):
    phase_list = [0]

    all_edge_positions = np.array([])
    true_or_false_array = np.array([])
    for i in range(len(edge_pieces)):
        temp = calc_numb_of_moves(edge_pieces[i], 1)
        temp_array = np.array([0, 0, 0, 0])
        temp_array[3] = (temp % 2 == 0)
        for j in range(len(temp_array) - 1):
            temp_array[j] = ((edge_pieces[i][0][0][j] + edge_pieces[i][1][0][j]) / 5 * 2)
        if i == 0:
            all_edge_positions = np.expand_dims(temp_array, 0)
        else:
            all_edge_positions = np.vstack([all_edge_positions, np.expand_dims(temp_array, 0)])
    return all_edge_positions





def random_scramble_list(moves, numb_of_moves):

    move_order = []
    banned_move = -1
    for i in range(numb_of_moves):
        if banned_move == -1:
            banned_move = random.choice([0, 1, 2])
        elif banned_move == 0:
            banned_move = random.choice([1, 2])
        elif banned_move == 1:
            banned_move = random.choice([0, 2])
        elif banned_move == 2:
            banned_move = random.choice([0, 1])
        move_order.append(random.choice(moves[banned_move]))

    return move_order

def random_scramble(position, array):
    if type(array) is not tuple:
        array = (array,)
    for i in range(len(array)):
        position = move_type(position, array[i])
    return position


def create_permutations(all_available_moves, max_depth):
    return list(itertools.permutations(all_available_moves, r = max_depth))

def lookup(move_type):
    if move_type == 'L':
        return 'R'
    elif move_type == 'R':
        return 'L'
    elif move_type == 'U':
        return 'D'
    elif move_type == 'D':
        return 'U'
    if move_type == 'F':
        return 'B'
    elif move_type == 'B':
        return 'F'


def dice_combinations(k, move_bucket):
    memo = {}
    if k == 1:
        memo[1] = [(i,) for i in move_bucket]
        return memo[1]

    elif k in memo:
        return memo[k]

    else:
        prev_res = dice_combinations(k - 1, move_bucket)
        res = []

        for comb in prev_res:
            for j in range(len(move_bucket)):
                if move_bucket[j][0] != comb[-1][0] and lookup(move_bucket[j][0]) != comb[-1][0]:
                    res.append(comb + (move_bucket[j],))

        memo[k] = res
        return res

def dice_combinations_4(k, move_bucket):
    memo = {}
    if k == 1:
        memo[1] = [(i,) for i in move_bucket]
        return memo[1]

    elif k in memo:
        return memo[k]

    else:
        prev_res = dice_combinations_4(k - 1, move_bucket)
        res = []

        for comb in prev_res:
            for j in range(len(move_bucket)):
                if move_bucket[j][0] != comb[-1][0]:
                    res.append(comb + (move_bucket[j],))

        memo[k] = res
        return res

def cube(rubix_colours, x_position, vertices_private):
    for j in range(len(vertices_private)):
        vertices_private[j][0] += x_position
    i = 0
    glBegin(GL_QUADS)
    for surface in surfaces:
        glColor3fv(rubix_colours[i])
        i += 1
        for vertex in surface:
            glVertex3fv(vertices_private[vertex])

    glEnd()
    glLineWidth(5.0)
    glBegin(GL_LINES)
    for edge in edges:
        glColor3fv((0, 0, 0))
        for vertex in edge:
            glVertex3fv(vertices_private[vertex])
    glEnd()
    glLineWidth(1.0)
"""

def cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
"""
"""def rotate_matrix(direction, matrix):
    if direction == 1:
        matrix = np.roll(matrix, 3)
    elif direction == -1:
        matrix = np.roll(matrix, -3)
    elif direction == 2:
        matrix = np.roll(matrix, 6)
    return matrix
def calculate_rotation(face, direction, rotation_direction, left_right, matrix):
    all_elements = np.array([])
    opp_left_right = 0
    if left_right == 0:
        opp_left_right = 2
    for i in range(4):
        temp = turning_faces[face][0][i]
        temp_array = np.array([])
        for j in range(3):
            if face <= 1:
                temp_array = np.append(temp_array, matrix[temp - 1][j][left_right])
            elif face <= 3:
                if temp == 6:
                    temp_array = np.append(temp_array, matrix[temp - 1][opp_left_right][j])
                else:
                    temp_array = np.append(temp_array, matrix[temp - 1][left_right][j])




                elif temp == 3:
                                    temp_array = np.append(temp_array, matrix[temp - 1][2 - j][left_right])
                                elif temp == 5:
                                    temp_array = np.append(temp_array, matrix[temp - 1][2 - j][opp_left_right])
            elif face <= 5:
                if temp == 2:
                    temp_array = np.append(temp_array, matrix[temp - 1][opp_left_right][j])
                elif temp == 4:
                    temp_array = np.append(temp_array, matrix[temp - 1][left_right][j])
                elif temp == 3:
                    temp_array = np.append(temp_array, matrix[temp - 1][j][left_right])

                else:
                    temp_array = np.append(temp_array, matrix[temp - 1][j][opp_left_right])



        if i == 0:
            all_elements = np.expand_dims(temp_array, 0)
        else:
            all_elements = np.append(all_elements, temp_array)

    all_elements = rotate_matrix(direction, all_elements)
    for i in range(4):
        for j in range(3):
            temp = turning_faces[face][0][i]

            if face <= 1:
                matrix[temp - 1][j][left_right] = round(all_elements[3 * i + j])
            elif face <= 3:
                element = j

                if direction == 1:
                    element = 2 - j
                if temp == 1:
                    matrix[temp - 1][left_right][j] = round(all_elements[3 * i + j])
                elif temp == 6:
                    matrix[temp - 1][opp_left_right][2 - j] = round(all_elements[3 * i + j])
                elif temp == 3:
                    matrix[temp - 1][left_right][2 - element] = round(all_elements[3 * i + j])
                elif temp == 5:
                    matrix[temp - 1][left_right][element] = round(all_elements[3 * i + j])

            elif face <= 5:
                element = j
                if direction == -1:
                    element = 2 - j
                if temp == 2:
                    matrix[temp - 1][opp_left_right][2 - element] = round(all_elements[3 * i + j])
                elif temp == 4:
                    matrix[temp - 1][left_right][2 - element] = round(all_elements[3 * i + j])
                elif temp == 3:
                    matrix[temp - 1][element][left_right] = round(all_elements[3 * i + j])
                else:
                    matrix[temp - 1][element][opp_left_right] = round(all_elements[3 * i + j])
    rotating_face = turning_faces[face][1] - 1
    face_5 = np.array(matrix[rotating_face]).T

    if rotation_direction == 1:
        matrix[rotating_face] = np.flipud(face_5)
    elif rotation_direction == -1:
        matrix[rotating_face] = np.fliplr(face_5)
    elif rotation_direction == 2:
        matrix[rotating_face] = np.flipud(face_5)
        matrix[rotating_face] = np.flipud(face_5)

    for element in matrix:
        for element_1 in element:
            print(element_1)
        print()
    return matrix"""

def cursor_pos_callback(window, x_pos, y_pos):
    print(x_pos, y_pos)



def rotate_edge_x(edge_to_rotate, rotation_type):
    if edge_to_rotate[1] == 0:
        edge_to_rotate = np.array([edge_to_rotate[0], -edge_to_rotate[2] * rotation_type, edge_to_rotate[1], edge_to_rotate[3]])
    else:
        edge_to_rotate = np.array([edge_to_rotate[0], edge_to_rotate[2], edge_to_rotate[1] * rotation_type, edge_to_rotate[3]])

    return edge_to_rotate

def rotate_edge_y(edge_to_rotate, rotation_type):
    if edge_to_rotate[0] == 0:
        edge_to_rotate = np.array([edge_to_rotate[2] * - rotation_type, edge_to_rotate[1], edge_to_rotate[0], 1 - edge_to_rotate[3]])
    else:
        edge_to_rotate = np.array([edge_to_rotate[2], edge_to_rotate[1], edge_to_rotate[0] * rotation_type, 1 - edge_to_rotate[3]])

    return edge_to_rotate

def rotate_edge_z(edge_to_rotate, rotation_type):
    if edge_to_rotate[0] == 0:
        edge_to_rotate = np.array([edge_to_rotate[1] * rotation_type, edge_to_rotate[0] + 0, edge_to_rotate[2], edge_to_rotate[3]])
    else:
        edge_to_rotate = np.array([edge_to_rotate[1], -1 * rotation_type * edge_to_rotate[0] + 0, edge_to_rotate[2], edge_to_rotate[3]])
    return edge_to_rotate

def rotate_edge_x_2_corner(edge_to_rotate, rotation_type):
    edge_to_rotate = np.array([edge_to_rotate[0], -rotation_type * edge_to_rotate[2], edge_to_rotate[1] * rotation_type, abs(edge_to_rotate[0]) // 3])

    return edge_to_rotate

def rotate_edge_y_2_corner(edge_to_rotate, rotation_type):
    edge_to_rotate = np.array([-edge_to_rotate[0], edge_to_rotate[1], -edge_to_rotate[2], abs(edge_to_rotate[0]) // 3])
    return edge_to_rotate

def rotate_edge_z_2_corner(edge_to_rotate, rotation_type):
    edge_to_rotate = np.array([edge_to_rotate[1] * rotation_type, -rotation_type * edge_to_rotate[0], edge_to_rotate[2], abs(edge_to_rotate[1]) // 3])
    return edge_to_rotate


def rotate_edge_x_2_edge(edge_to_rotate, rotation_type):
    if edge_to_rotate[1] == 0:
        edge_to_rotate = np.array([edge_to_rotate[0], -edge_to_rotate[2] * rotation_type, edge_to_rotate[1], 1 - abs(edge_to_rotate[0]) // 2])
    else:
        edge_to_rotate = np.array([edge_to_rotate[0], edge_to_rotate[2], edge_to_rotate[1] * rotation_type, 1 - abs(edge_to_rotate[0]) // 2])

    return edge_to_rotate

def rotate_edge_y_2_edge(edge_to_rotate, rotation_type):
    edge_to_rotate = np.array([-edge_to_rotate[0], edge_to_rotate[1], -edge_to_rotate[2], 1 - abs(edge_to_rotate[0]) // 2])
    return edge_to_rotate

def rotate_edge_z_2_edge(edge_to_rotate, rotation_type):
    if edge_to_rotate[0] == 0:
        edge_to_rotate = np.array([edge_to_rotate[1] * rotation_type, edge_to_rotate[0] + 0, edge_to_rotate[2], 1 - abs(edge_to_rotate[1]) // 2])
    else:
        edge_to_rotate = np.array([edge_to_rotate[1], - rotation_type * edge_to_rotate[0] + 0, edge_to_rotate[2], 1 - abs(edge_to_rotate[1]) // 2])
    return edge_to_rotate

def rotation_x_3(edge_to_rotate, rotation_type):
    edge_to_rotate = np.array([edge_to_rotate[0], -rotation_type * edge_to_rotate[2], edge_to_rotate[1] * rotation_type, 1 - edge_to_rotate[3]])
    return edge_to_rotate

def rotation_y_3(edge_to_rotate):
    edge_to_rotate = np.array([-edge_to_rotate[0], edge_to_rotate[1], -edge_to_rotate[2], edge_to_rotate[3]])
    return edge_to_rotate

def rotation_z_3(edge_to_rotate):
    edge_to_rotate = np.array([-edge_to_rotate[0], -edge_to_rotate[1], edge_to_rotate[2], edge_to_rotate[3]])
    return edge_to_rotate

def rotation_x_4_corner(edge_to_rotate):
    edge_to_rotate[0] = np.array([edge_to_rotate[0][0], -edge_to_rotate[0][1], -edge_to_rotate[0][2]])
    edge_to_rotate[1][1] = 1 - edge_to_rotate[1][1]
    edge_to_rotate[1][2] = 1 - edge_to_rotate[1][2]
    edge_to_rotate[2][0] = sum(edge_to_rotate[1]) // 3
    return edge_to_rotate

def rotation_y_4_corner(edge_to_rotate):
    edge_to_rotate[0] = np.array([-edge_to_rotate[0][0], edge_to_rotate[0][1], -edge_to_rotate[0][2]])
    edge_to_rotate[1][0] = 1 - edge_to_rotate[1][0]
    edge_to_rotate[1][2] = 1 - edge_to_rotate[1][2]
    edge_to_rotate[2][0] = sum(edge_to_rotate[1]) // 3
    return edge_to_rotate

def rotation_z_4_corner(edge_to_rotate):
    edge_to_rotate[0] = np.array([-edge_to_rotate[0][0], -edge_to_rotate[0][1], edge_to_rotate[0][2]])
    edge_to_rotate[1][0] = 1 - edge_to_rotate[1][0]
    edge_to_rotate[1][1] = 1 - edge_to_rotate[1][1]
    edge_to_rotate[2][0] = sum(edge_to_rotate[1]) // 3
    return edge_to_rotate

def rotation_x_4_edge(edge_to_rotate):
    edge_to_rotate[0] = np.array([edge_to_rotate[0][0], -edge_to_rotate[0][1], -edge_to_rotate[0][2]])
    edge_to_rotate[1][1] = 1 - edge_to_rotate[1][1] * abs(edge_to_rotate[0][1]) / 2
    edge_to_rotate[1][2] = 1 - edge_to_rotate[1][2] * abs(edge_to_rotate[0][2]) / 2
    edge_to_rotate[2][0] = sum(edge_to_rotate[1]) // 3
    return edge_to_rotate

def rotation_y_4_edge(edge_to_rotate):
    edge_to_rotate[0] = np.array([-edge_to_rotate[0][0], edge_to_rotate[0][1], -edge_to_rotate[0][2]])
    edge_to_rotate[1][0] = 1 - edge_to_rotate[1][0] * abs(edge_to_rotate[0][0]) / 2
    edge_to_rotate[1][2] = 1 - edge_to_rotate[1][2] * abs(edge_to_rotate[0][2]) / 2
    edge_to_rotate[2][0] = sum(edge_to_rotate[1]) // 3
    return edge_to_rotate

def rotation_z_4_edge(edge_to_rotate):
    edge_to_rotate[0] = np.array([-edge_to_rotate[0][0], -edge_to_rotate[0][1], edge_to_rotate[0][2]])
    edge_to_rotate[1][0] = 1 - edge_to_rotate[1][0] * abs(edge_to_rotate[0][0]) / 2
    edge_to_rotate[1][1] = 1 - edge_to_rotate[1][1] * abs(edge_to_rotate[0][1]) / 2
    edge_to_rotate[2][0] = sum(edge_to_rotate[1]) // 3
    return edge_to_rotate

"""def check_position(move_sequence, array_of_moves):
    parity_numb = 0
    array_of_moves_copy = copy.copy(array_of_moves)
    for element in move_sequence:
        if element[0] == 'L':
            if element == 'L':
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[0] == -2:
                        array_of_moves_copy[x] = rotate_edge_x(edge, 1)
            else:
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[0] == -2:
                        array_of_moves_copy[x] = rotate_edge_x(edge, -1)
        elif element[0] == 'R':
            if element == 'R':
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[0] == 2:
                        array_of_moves_copy[x] = rotate_edge_x(edge, -1)

            else:
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[0] == 2:
                        array_of_moves_copy[x] = rotate_edge_x(edge, 1)

        elif element[0] == 'U':
            if element == 'U':
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[1] == 2:
                        array_of_moves_copy[x] = rotate_edge_y(edge, 1)

            else:
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[1] == 2:
                        array_of_moves_copy[x] = rotate_edge_y(edge, -1)

        elif element[0] == 'D':
            if element == 'D':
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[1] == -2:
                        array_of_moves_copy[x] = rotate_edge_y(edge, -1)

            else:
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[1] == -2:
                        array_of_moves_copy[x] = rotate_edge_y(edge, 1)

        elif element[0] == 'F':
            if element == 'F':
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[2] == 2:
                        array_of_moves_copy[x] = rotate_edge_z(edge, 1)
            else:
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[2] == 2:
                        array_of_moves_copy[x] = rotate_edge_z(edge, -1)
        elif element[0] == 'B':
            if element == 'B':
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[2] == -2:
                        array_of_moves_copy[x] = rotate_edge_z(edge, -1)
            else:
                for x, edge in enumerate(array_of_moves_copy):
                    if edge[2] == -2:
                        array_of_moves_copy[x] = rotate_edge_z(edge, 1)
    for x, element in enumerate(array_of_moves_copy):
        parity_numb += element[3] * (2 ** (10 - x))
    return move_sequence, parity_numb"""
def check_position_2(move_sequence, array_of_moves, forwards_or_back):
    parity_numb = 0
    array_of_moves_copy = copy.copy(array_of_moves)
    # L - 0 1 2 3
    # R - 8 9 10 11
    # U - 3 6 7 11
    # D - 0 4 5 8
    # F - 2 5 7 10
    # B - 1 4 6 9
    for element in move_sequence:
        if element[0] == 'L':
            if element == 'L':

                array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = array_of_moves_copy[10][3], array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3]
                array_of_moves_copy[3][3], array_of_moves_copy[1][3], array_of_moves_copy[0][3], array_of_moves_copy[2][3] = x_update(array_of_moves_copy[2][3]), x_update(array_of_moves_copy[3][3]), x_update(array_of_moves_copy[1][3]), x_update(array_of_moves_copy[0][3])

            else:
                array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3], array_of_moves_copy[8][3],
                array_of_moves_copy[3][3], array_of_moves_copy[1][3], array_of_moves_copy[0][3], array_of_moves_copy[2][3] = x_update(array_of_moves_copy[1][3]), x_update(array_of_moves_copy[0][3]), x_update(array_of_moves_copy[2][3]), x_update(array_of_moves_copy[3][3])

        elif element[0] == 'R':
            if element == 'R':
                array_of_moves_copy[16][3], array_of_moves_copy[17][3], array_of_moves_copy[19][3], array_of_moves_copy[18][3] = array_of_moves_copy[17][3], array_of_moves_copy[19][3], array_of_moves_copy[18][3], array_of_moves_copy[16][3]
                array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3], array_of_moves_copy[5][3] = x_update(array_of_moves_copy[5][3]), x_update(array_of_moves_copy[7][3]), x_update(array_of_moves_copy[6][3]), x_update(array_of_moves_copy[4][3])

            else:
                array_of_moves_copy[16][3], array_of_moves_copy[17][3], array_of_moves_copy[19][3], array_of_moves_copy[18][3] = array_of_moves_copy[18][3], array_of_moves_copy[16][3], array_of_moves_copy[17][3], array_of_moves_copy[19][3]
                array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3], array_of_moves_copy[5][3] = x_update(array_of_moves_copy[6][3]), x_update(array_of_moves_copy[4][3]), x_update(array_of_moves_copy[5][3]), x_update(array_of_moves_copy[7][3])

        elif element == 'U2':
            array_of_moves_copy[11][3], array_of_moves_copy[14][3], array_of_moves_copy[19][3], array_of_moves_copy[15][3] = array_of_moves_copy[19][3], array_of_moves_copy[15][3], array_of_moves_copy[11][3], array_of_moves_copy[14][3]
            array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[2][3], array_of_moves_copy[3][3] = array_of_moves_copy[2][3], array_of_moves_copy[3][3], array_of_moves_copy[7][3], array_of_moves_copy[6][3]

        elif element == 'D2':
            array_of_moves_copy[8][3], array_of_moves_copy[12][3], array_of_moves_copy[16][3], array_of_moves_copy[13][3] = array_of_moves_copy[16][3], array_of_moves_copy[13][3], array_of_moves_copy[8][3], array_of_moves_copy[12][3]
            array_of_moves_copy[5][3], array_of_moves_copy[4][3], array_of_moves_copy[0][3], array_of_moves_copy[1][3] = array_of_moves_copy[0][3], array_of_moves_copy[1][3], array_of_moves_copy[5][3], array_of_moves_copy[4][3]

        elif element[0] == 'F':
            if element == 'F':
                array_of_moves_copy[10][3], array_of_moves_copy[13][3], array_of_moves_copy[18][3], array_of_moves_copy[15][3] = array_of_moves_copy[13][3], array_of_moves_copy[18][3], array_of_moves_copy[15][3], array_of_moves_copy[10][3]
                array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[7][3], array_of_moves_copy[5][3] = z_update(array_of_moves_copy[5][3]), z_update(array_of_moves_copy[1][3]), z_update(array_of_moves_copy[3][3]), z_update(array_of_moves_copy[7][3])

            else:
                array_of_moves_copy[10][3], array_of_moves_copy[13][3], array_of_moves_copy[18][3], array_of_moves_copy[15][3] = array_of_moves_copy[15][3], array_of_moves_copy[10][3], array_of_moves_copy[13][3], array_of_moves_copy[18][3]
                array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[7][3], array_of_moves_copy[5][3] = z_update(array_of_moves_copy[3][3]), z_update(array_of_moves_copy[7][3]), z_update(array_of_moves_copy[5][3]), z_update(array_of_moves_copy[1][3])


        elif element[0] == 'B':
            if element == 'B':
                array_of_moves_copy[9][3], array_of_moves_copy[12][3], array_of_moves_copy[17][3], array_of_moves_copy[14][3], = array_of_moves_copy[14][3], array_of_moves_copy[9][3], array_of_moves_copy[12][3], array_of_moves_copy[17][3]
                array_of_moves_copy[0][3], array_of_moves_copy[2][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3] = z_update(array_of_moves_copy[2][3]), z_update(array_of_moves_copy[6][3]), z_update(array_of_moves_copy[4][3]), z_update(array_of_moves_copy[0][3])
            else:
                array_of_moves_copy[0][3], array_of_moves_copy[2][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3] = z_update(array_of_moves_copy[4][3]), z_update(array_of_moves_copy[0][3]), z_update(array_of_moves_copy[2][3]), z_update(array_of_moves_copy[6][3])
                array_of_moves_copy[9][3], array_of_moves_copy[12][3], array_of_moves_copy[17][3], array_of_moves_copy[14][3] = array_of_moves_copy[12][3], array_of_moves_copy[17][3], array_of_moves_copy[14][3], array_of_moves_copy[9][3]

    for x, element in enumerate(array_of_moves_copy):
        if x < 8:
            parity_numb += element[3] * (4 ** (7 - x))
        else:
            parity_numb += element[3] * (2 ** (19 - x)) * 65536
    return move_sequence, parity_numb

def check_position_3(move_sequence, array_of_moves, numb):
    parity_numb = 0
    array_of_moves_copy = copy.copy(array_of_moves)
    for element in move_sequence:
        if element[0] == 'L':
            if element == 'L':

                array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = 1 - array_of_moves_copy[10][3], 1 - array_of_moves_copy[8][3], 1 - array_of_moves_copy[9][3], 1 - array_of_moves_copy[11][3]
                array_of_moves_copy[3][3], array_of_moves_copy[1][3], array_of_moves_copy[0][3], array_of_moves_copy[2][3] = 1 - array_of_moves_copy[2][3], 1 - array_of_moves_copy[3][3], 1 - array_of_moves_copy[1][3], 1 - array_of_moves_copy[0][3]

            else:
                array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = 1 - array_of_moves_copy[9][3], 1 - array_of_moves_copy[11][3], 1 - array_of_moves_copy[10][3], 1 - array_of_moves_copy[8][3],
                array_of_moves_copy[3][3], array_of_moves_copy[1][3], array_of_moves_copy[0][3], array_of_moves_copy[2][3] = 1 - array_of_moves_copy[1][3], 1 - array_of_moves_copy[0][3], 1 - array_of_moves_copy[2][3], 1 - array_of_moves_copy[3][3]

        elif element[0] == 'R':
            if element == 'R':
                array_of_moves_copy[16][3], array_of_moves_copy[17][3], array_of_moves_copy[19][3], array_of_moves_copy[18][3] = 1 - array_of_moves_copy[17][3], 1 - array_of_moves_copy[19][3], 1 - array_of_moves_copy[18][3], 1 - array_of_moves_copy[16][3]
                array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3], array_of_moves_copy[5][3] = 1 - array_of_moves_copy[5][3], 1 - array_of_moves_copy[7][3], 1 - array_of_moves_copy[6][3], 1 - array_of_moves_copy[4][3]

            else:
                array_of_moves_copy[16][3], array_of_moves_copy[17][3], array_of_moves_copy[19][3], array_of_moves_copy[18][3] = 1 - array_of_moves_copy[18][3], 1 - array_of_moves_copy[16][3], 1 - array_of_moves_copy[17][3], 1 - array_of_moves_copy[19][3]
                array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3], array_of_moves_copy[5][3] = 1 - array_of_moves_copy[6][3], 1 - array_of_moves_copy[4][3], 1 - array_of_moves_copy[5][3], 1 - array_of_moves_copy[7][3]

        elif element == 'U2':
            array_of_moves_copy[11][3], array_of_moves_copy[14][3], array_of_moves_copy[19][3], array_of_moves_copy[15][3] = array_of_moves_copy[19][3], array_of_moves_copy[15][3], array_of_moves_copy[11][3], array_of_moves_copy[14][3]
            array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[2][3], array_of_moves_copy[3][3] = array_of_moves_copy[2][3], array_of_moves_copy[3][3], array_of_moves_copy[7][3], array_of_moves_copy[6][3]

        elif element == 'D2':
            array_of_moves_copy[8][3], array_of_moves_copy[12][3], array_of_moves_copy[16][3], array_of_moves_copy[13][3] = array_of_moves_copy[16][3], array_of_moves_copy[13][3], array_of_moves_copy[8][3], array_of_moves_copy[12][3]
            array_of_moves_copy[5][3], array_of_moves_copy[4][3], array_of_moves_copy[0][3], array_of_moves_copy[1][3] = array_of_moves_copy[0][3], array_of_moves_copy[1][3], array_of_moves_copy[5][3], array_of_moves_copy[4][3]

        elif element == 'F2':
            array_of_moves_copy[10][3], array_of_moves_copy[13][3], array_of_moves_copy[18][3], array_of_moves_copy[15][3] = array_of_moves_copy[18][3], array_of_moves_copy[15][3], array_of_moves_copy[10][3], array_of_moves_copy[13][3]
            array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[7][3], array_of_moves_copy[5][3] = array_of_moves_copy[7][3], array_of_moves_copy[5][3], array_of_moves_copy[1][3], array_of_moves_copy[3][3]

        elif element == 'B2':
            array_of_moves_copy[9][3], array_of_moves_copy[12][3], array_of_moves_copy[17][3], array_of_moves_copy[14][3] = array_of_moves_copy[17][3], array_of_moves_copy[14][3], array_of_moves_copy[9][3], array_of_moves_copy[12][3]
            array_of_moves_copy[0][3], array_of_moves_copy[2][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3] = array_of_moves_copy[6][3], array_of_moves_copy[4][3], array_of_moves_copy[0][3], array_of_moves_copy[2][3]

    for x, element in enumerate(array_of_moves_copy):
        parity_numb += element[3] * (2 ** (19 - x))
    return move_sequence, parity_numb

def check_position(move_sequence, array_of_moves, forwards_or_back):
    parity_numb = 0
    array_of_moves_copy = copy.copy(array_of_moves)
    # L - 0 1 2 3
    # R - 8 9 10 11
    # U - 3 6 7 11
    # D - 0 4 5 8
    # F - 2 5 7 10
    # B - 1 4 6 9

    for element in move_sequence:
        if element[0] == 'L':
            if element == 'L':
                array_of_moves_copy[0][3], array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[2][3] = array_of_moves_copy[2][3], array_of_moves_copy[0][3], array_of_moves_copy[1][3], array_of_moves_copy[3][3]
            else:
                array_of_moves_copy[0][3], array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[2][3] = array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[2][3], array_of_moves_copy[0][3],


        elif element[0] == 'R':
            if element != 'R':
                array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = array_of_moves_copy[10][3], array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3]
            else:
                array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3], array_of_moves_copy[8][3]

        elif element[0] == 'U':
            if element == 'U':
                array_of_moves_copy[3][3], array_of_moves_copy[6][3], array_of_moves_copy[11][3], array_of_moves_copy[7][3] = 1 - array_of_moves_copy[7][3], 1 - array_of_moves_copy[3][3], 1 - array_of_moves_copy[6][3], 1 - array_of_moves_copy[11][3]
            else:
                array_of_moves_copy[3][3], array_of_moves_copy[6][3], array_of_moves_copy[11][3], array_of_moves_copy[7][3] = 1 - array_of_moves_copy[6][3], 1 - array_of_moves_copy[11][3], 1 - array_of_moves_copy[7][3], 1 - array_of_moves_copy[3][3]
        elif element[0] == 'D':
            if element == 'D':
                array_of_moves_copy[0][3], array_of_moves_copy[4][3], array_of_moves_copy[8][3], array_of_moves_copy[5][3] = 1 - array_of_moves_copy[4][3], 1 - array_of_moves_copy[8][3], 1 - array_of_moves_copy[5][3], 1 - array_of_moves_copy[0][3]
            else:
                array_of_moves_copy[0][3], array_of_moves_copy[4][3], array_of_moves_copy[8][3], array_of_moves_copy[5][3] = 1 - array_of_moves_copy[5][3], 1 - array_of_moves_copy[0][3], 1 - array_of_moves_copy[4][3], 1 - array_of_moves_copy[8][3]
        elif element[0] == 'F':
            if element == 'F':
                array_of_moves_copy[2][3], array_of_moves_copy[5][3], array_of_moves_copy[10][3], array_of_moves_copy[7][3] = array_of_moves_copy[5][3], array_of_moves_copy[10][3], array_of_moves_copy[7][3], array_of_moves_copy[2][3]

            else:
                array_of_moves_copy[2][3], array_of_moves_copy[5][3], array_of_moves_copy[10][3], array_of_moves_copy[7][3] = array_of_moves_copy[7][3], array_of_moves_copy[2][3], array_of_moves_copy[5][3], array_of_moves_copy[10][3]

        elif element[0] == 'B':
            if element == 'B':
                array_of_moves_copy[1][3], array_of_moves_copy[4][3], array_of_moves_copy[9][3], array_of_moves_copy[6][3] = array_of_moves_copy[6][3], array_of_moves_copy[1][3], array_of_moves_copy[4][3], array_of_moves_copy[9][3]

            else:
                array_of_moves_copy[1][3], array_of_moves_copy[4][3], array_of_moves_copy[9][3], array_of_moves_copy[6][3] = array_of_moves_copy[4][3], array_of_moves_copy[9][3], array_of_moves_copy[6][3], array_of_moves_copy[1][3]

    for x, element in enumerate(array_of_moves_copy):
        parity_numb += element[3] * (2 ** (11 - x))
    return move_sequence, parity_numb

def reverse_letter(letter):
    if letter[-1] == '\'':
        letter = letter[0]
    elif letter[-1] != '2':
        letter = letter + '\''
    return letter

def check_bi_direction(dict1, dict2):
    set1 = sorted(dict1)
    set2 = sorted(dict2)

    # Check for same elements
    # using merge like process
    i = 0
    j = 0

    m = len(set1)
    n = len(set2)
    while (i < m and j < n):
        if (set1[i] < set2[j]):
            i += 1
        elif (set2[j] < set1[i]):
            j += 1
        else:
            print(bin(round(set1[i])))
            return dict1[set1[i]], dict2[set2[j]]
    return None, None

def check_all_combinations_stage_1(depth, positions):
    forward_dict = {}
    backward_dict = {}
    checker = False
    ideal_position = np.array([[-2,-2, 0, 1],
    [-2, 0,-2, 1],
    [-2, 0, 2, 1],
    [-2, 2, 0, 1],
    [0,-2,-2, 1],
    [0,-2,2, 1],
    [0, 2,-2, 1],
    [0,2,2, 1],
    [2,-2, 0, 1],
    [2, 0,-2, 1],
    [2, 0, 2, 1],
    [2, 2, 0, 1]])
    for i in range(1, depth + 1):
        if checker == True:
            break
        else:
            all_situations = dice_combinations(i, stage_one_moves)
        forward_dict = {}
        for x, element in enumerate(all_situations):
            seq, numb = check_position(element, positions, 0)
            forward_dict.update({numb: seq})
            forward_dict[numb] = seq
            if numb == 4095:
                checker = True
                return seq
        if i > 1:
            move1, move2 = check_bi_direction(forward_dict, backward_dict)
            if move1 != None:
                move1_list = list(move1)
                move2_list = list(move2)

                for x, element in enumerate(move2):
                    move2_list[x] = reverse_letter(element)

                return tuple(move1_list + move2_list[::-1])



        backward_dict = {}
        for element in all_situations:
            seq, numb = check_position(element, ideal_position, 1)
            backward_dict[numb] = seq
        move1, move2 = check_bi_direction(forward_dict, backward_dict)
        if move1 != None:
            move1_list = list(move1)
            move2_list = list(move2)
            for x, element in enumerate(move2):
                move2_list[x] = reverse_letter(element)
            return tuple(move1_list + move2_list[::-1])


def sort_position(array):
    correct_places_dict = {}
    for x, piece in enumerate(array):
        full_array = np.array([[]])
        if x < 8:
            numb = 0
            for y, element in enumerate(piece):
                if pos_neg(element) > 0 and y < 3:
                    numb += 2 ** (2 - y)
            correct_places_dict[round(numb)] = x
        if x > 7:
            numb = 0
            if abs(pos_neg(piece[0])) == 1:
                if pos_neg(piece[1]) == -1 and pos_neg(piece[2]) == 0:
                    numb = 8
                if pos_neg(piece[1]) == 0 and pos_neg(piece[2]) == -1:
                    numb = 9
                if pos_neg(piece[1]) == 0 and pos_neg(piece[2]) == 1:
                    numb = 10
                if pos_neg(piece[1]) == 1 and pos_neg(piece[2]) == 0:
                    numb = 11
                numb += (pos_neg(piece[0]) + 1) * 4
            elif pos_neg(piece[0]) == 0:
                if pos_neg(piece[1]) == -1 and pos_neg(piece[2]) == -1:
                    numb = 12
                if pos_neg(piece[1]) == -1 and pos_neg(piece[2]) == 1:
                    numb = 13
                if pos_neg(piece[1]) == 1 and pos_neg(piece[2]) == -1:
                    numb = 14
                if pos_neg(piece[1]) == 1 and pos_neg(piece[2]) == 1:
                    numb = 15
            correct_places_dict[round(numb)] = x
    for i in range(20):
        if i == 0:
            full_array = np.expand_dims(array[correct_places_dict[i]], 0)
        else:
            full_array = np.vstack([full_array, np.expand_dims(array[correct_places_dict[i]], 0)])
    return full_array





def array_manipulation_2(position_array):
    final_array = np.array([])
    i = 0
    for element in position_array:
        temp_array = np.array([element[0][0], element[0][1], element[0][2], element[3][0]])
        if i == 0:
            final_array = np.expand_dims(temp_array, 0)
        else:
            final_array = np.vstack([final_array, np.expand_dims(temp_array, 0)])
        i += 1
    return final_array



def check_all_combinations_stage_2(depth, positions):

    forward_dict = {}
    backward_dict = {}
    checker = False

    ideal_position = np.array([[-3, -2, -2, 0],
     [-3, -2,  2, 0],
     [-2, 3, -2, 0],
     [-2, 2, 3, 0],
     [2, -2, -3, 0],
     [2, -2, 3, 0],
     [2, 3, -2, 0],
     [2, 2, 3, 0],
     [-2,-3, 0, 0],
     [-2, 0, -3, 0],
     [-2, 0, 3, 0],
     [-2, 3, 0, 0],
     [0, -2, -3, 1],
     [0, -3, 2, 1],
     [0, 2, -3, 1],
     [0, 2, 3, 1],
     [3,-2, 0, 0],
     [2, 0, -3,0],
     [3, 0, 2, 0],
     [3, 2, 0, 0]])

    for i in range(1, depth + 1):
        if checker == True:
            break
        else:
            all_situations = dice_combinations(i, stage_two_moves)
        forward_dict = {}
        for x, element in enumerate(all_situations):
            seq, numb = check_position_2(element, positions, 0)
            forward_dict.update({numb: seq})
            forward_dict[numb] = seq
            if i == 1:
                if numb == 15728640:
                    return seq

        if i > 1:
            move1, move2 = check_bi_direction(forward_dict, backward_dict)
            if move1 != None:
                move1_list = list(move1)
                move2_list = list(move2)
                for x, element in enumerate(move2):
                    move2_list[x] = reverse_letter(element)
                return tuple(move1_list + move2_list[::-1])



        backward_dict = {}
        for element in all_situations:
            seq, numb = check_position_2(element, ideal_position, 1)
            backward_dict[numb] = seq
        move1, move2 = check_bi_direction(forward_dict, backward_dict)
        if move1 != None:
            move1_list = list(move1)
            move2_list = list(move2)
            for x, element in enumerate(move2):
                move2_list[x] = reverse_letter(element)
            return tuple(move1_list + move2_list[::-1])


def check_all_combinations_stage_3(depth, positions):

    forward_dict = {}
    backward_dict = {}
    checker = False

    ideal_position = np.array([[-3, -2, -2, 1],
     [-3, -2,  2, 1],
     [-2, 3, -2, 1],
     [-2, 2, 3, 1],
     [2, -2, -3, 1],
     [2, -2, 3, 1],
     [2, 3, -2, 1],
     [2, 2, 3, 1],
     [-2,-3, 0, 1],
     [-2, 0, -3, 1],
     [-2, 0, 3, 1],
     [-2, 3, 0, 1],
     [0, -2, -3, 1],
     [0, -3, 2, 1],
     [0, 2, -3, 1],
     [0, 2, 3, 1],
     [3,-2, 0, 1],
     [2, 0, -3,1],
     [3, 0, 2, 1],
     [3, 2, 0, 1]])

    for i in range(1, depth + 1):

        if checker == True:
            break
        else:
            all_situations = dice_combinations(i, stage_three_moves)
        forward_dict = {}
        for x, element in enumerate(all_situations):
            seq, numb = check_position_3(element, positions, 0)
            forward_dict.update({numb: seq})
            forward_dict[numb] = seq
            if i == 1:
                if numb == 15728640:
                    return seq

        if i > 1:
            move1, move2 = check_bi_direction(forward_dict, backward_dict)
            if move1 != None:
                move1_list = list(move1)
                move2_list = list(move2)
                for x, element in enumerate(move2):
                    move2_list[x] = reverse_letter(element)
                return tuple(move1_list + move2_list[::-1])



        backward_dict = {}
        for element in all_situations:
            seq, numb = check_position_3(element, ideal_position, 1)
            backward_dict[numb] = seq
        move1, move2 = check_bi_direction(forward_dict, backward_dict)
        if move1 != None:
            move1_list = list(move1)
            move2_list = list(move2)
            for x, element in enumerate(move2):
                move2_list[x] = reverse_letter(element)
            return tuple(move1_list + move2_list[::-1])


"""def check_position_3(sequence, position):
    counter = 0
    print(position, sequence)
    for element in sequence:
        if element[0] == 'L':
            if element == 'L':
                for x, edge in enumerate(position):
                    if edge[0] <= -2:
                        position[x] = rotation_x_3(position[x], 1)


            else:
                for x, edge in enumerate(position):
                    if edge[0] <= -2:
                        position[x] = rotation_x_3(position[x], -1)

        elif element[0] == 'R':
            if element == 'R':
                for x, edge in enumerate(position):
                    if edge[0] >= 2:
                        position[x] = rotation_x_3(position[x], -1)
            else:
                for x, edge in enumerate(position):
                    if edge[0] >= 2:
                        position[x] = rotation_x_3(position[x], 1)
        elif element[0] == 'U':
            for x, edge in enumerate(position):
                if edge[1] >= 2:
                    position[x] = rotation_y_3(position[x])
        elif element[0] == 'D':
            for x, edge in enumerate(position):
                if edge[1] <= -2:
                    position[x] = rotation_y_3(position[x])
        elif element[0] == 'F':
            for x, edge in enumerate(position):
                if edge[2] >= 2:
                    position[x] = rotation_z_3(position[x])
        elif element[0] == 'B':
            for x, edge in enumerate(position):
                if edge[2] <= -2:
                    position[x] = rotation_z_3(position[x])

    for element in position:
        counter += element[3]
    return counter"""
def x_change(numb):

    return numb + (4 * (1 - 2 * (numb // 4)))


def y_change(numb):
    return numb - 4 * ((numb % 4) // 2) + 2

def z_change(numb):
    return numb + (1 - 2 * (numb % 2))

def check_position_4(move_sequence, array_of_moves, numb):
    parity_numb = 0
    array_of_moves_copy = copy.copy(array_of_moves)

    for element in move_sequence:
        if element == 'L2':
            array_of_moves_copy[3][3], array_of_moves_copy[1][3], array_of_moves_copy[0][3], array_of_moves_copy[2][3] = y_change(z_change(array_of_moves_copy[0][3])), y_change(z_change(array_of_moves_copy[2][3])), y_change(z_change(array_of_moves_copy[3][3])), y_change(z_change(array_of_moves_copy[1][3]))

            """array_of_moves_copy[3][3], array_of_moves_copy[1][3], array_of_moves_copy[0][3], array_of_moves_copy[2][
                3] = ((array_of_moves_copy[0][3])), (
               (array_of_moves_copy[2][3])), ((array_of_moves_copy[3][3])), (
                (array_of_moves_copy[1][3]))"""

            array_of_moves_copy[8][3], array_of_moves_copy[9][3], array_of_moves_copy[11][3], array_of_moves_copy[10][3] = y_change(array_of_moves_copy[11][3]), z_change(array_of_moves_copy[10][3]), y_change(array_of_moves_copy[8][3]), z_change(array_of_moves_copy[9][3])

        elif element == 'R2':

            array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3], array_of_moves_copy[5][3] = y_change(z_change(array_of_moves_copy[4][3])), y_change(z_change(array_of_moves_copy[5][3])), y_change(z_change(array_of_moves_copy[7][3])), y_change(z_change(array_of_moves_copy[6][3]))
            array_of_moves_copy[16][3], array_of_moves_copy[17][3], array_of_moves_copy[19][3], array_of_moves_copy[18][3] = y_change(array_of_moves_copy[19][3]), z_change(array_of_moves_copy[18][3]), y_change(array_of_moves_copy[16][3]), z_change(array_of_moves_copy[17][3])

        elif element == 'U2':
            array_of_moves_copy[7][3], array_of_moves_copy[6][3], array_of_moves_copy[2][3], array_of_moves_copy[3][3] = x_change(z_change(array_of_moves_copy[2][3])), x_change(z_change(array_of_moves_copy[3][3])), x_change(z_change(array_of_moves_copy[7][3])), x_change(z_change(array_of_moves_copy[6][3]))
            array_of_moves_copy[11][3], array_of_moves_copy[14][3], array_of_moves_copy[19][3], array_of_moves_copy[15][3] = x_change(array_of_moves_copy[19][3]), z_change(array_of_moves_copy[15][3]), x_change(array_of_moves_copy[11][3]), z_change(array_of_moves_copy[14][3])

        elif element == 'D2':
            array_of_moves_copy[8][3], array_of_moves_copy[12][3], array_of_moves_copy[16][3], array_of_moves_copy[13][3] = x_change(array_of_moves_copy[16][3]), z_change(array_of_moves_copy[13][3]), x_change(array_of_moves_copy[8][3]), z_change(array_of_moves_copy[12][3])
            array_of_moves_copy[5][3], array_of_moves_copy[4][3], array_of_moves_copy[0][3], array_of_moves_copy[1][3] = x_change(z_change(array_of_moves_copy[0][3])), x_change(z_change(array_of_moves_copy[1][3])), x_change(z_change(array_of_moves_copy[5][3])), x_change(z_change(array_of_moves_copy[4][3]))

        elif element == 'F2':
            array_of_moves_copy[1][3], array_of_moves_copy[3][3], array_of_moves_copy[7][3], array_of_moves_copy[5][3] = x_change(y_change(array_of_moves_copy[7][3])), x_change(y_change(array_of_moves_copy[5][3])), x_change(y_change(array_of_moves_copy[1][3])), x_change(y_change(array_of_moves_copy[3][3]))
            array_of_moves_copy[10][3], array_of_moves_copy[13][3], array_of_moves_copy[18][3], array_of_moves_copy[15][3] = x_change(array_of_moves_copy[18][3]), y_change(array_of_moves_copy[15][3]), x_change(array_of_moves_copy[10][3]), y_change(array_of_moves_copy[13][3])

        elif element == 'B2':
            array_of_moves_copy[0][3], array_of_moves_copy[2][3], array_of_moves_copy[6][3], array_of_moves_copy[4][3] = x_change(y_change(array_of_moves_copy[6][3])), x_change(y_change(array_of_moves_copy[4][3])), x_change(y_change(array_of_moves_copy[0][3])), x_change(y_change(array_of_moves_copy[2][3]))
            array_of_moves_copy[9][3], array_of_moves_copy[12][3], array_of_moves_copy[17][3], array_of_moves_copy[14][3] = x_change(array_of_moves_copy[17][3]), y_change(array_of_moves_copy[14][3]), x_change(array_of_moves_copy[9][3]), y_change(array_of_moves_copy[12][3])
    print(array_of_moves_copy)
    temp = 0
    """if len(move_sequence) == 1 and numb == 0:
        print(array_of_moves_copy, move_sequence)"""
    for x, element in enumerate(array_of_moves_copy):
        parity_numb += element[3] * (8 ** (19 - x))
    return move_sequence, parity_numb


def array_manipulation_3(position_array_corner, position_array_edge):
    final_array = np.array([])
    i = 0
    print(position_array_corner.shape, position_array_edge.shape)
    for element in position_array_corner:
        temp_array = np.array([element[0][0][0], element[0][0][1], element[0][0][2], 1 - (abs(element[0][0][2]) - abs(element[0][2][2]))])
        if i == 0:
            final_array = np.expand_dims(temp_array, 0)
            i += 1
        else:
            final_array = np.vstack([final_array, np.expand_dims(temp_array, 0)])
    for element in position_array_edge:
        if abs(element[0][1]) == 0:
            temp_array = np.array([element[0][0], element[0][1], element[0][2], 1 - (abs(element[0][2]) - abs(element[2][2])) // 3])
        else:
            temp_array = np.array([element[0][0], element[0][1], element[0][2], 1 - (abs(element[0][1]) - abs(element[2][1])) // 3])
        final_array = np.vstack([final_array, np.expand_dims(temp_array, 0)])
    return final_array




def array_manipulation_4(position_array_corner, position_array_edge):
    final_array = np.array([])
    i = 0
    for element in position_array_corner:
        pos_array = np.array([(abs(element[0][0][0]) // 2 * 2) * pos_neg(element[0][0][0]), (abs(element[0][0][1]) // 2 * 2) * pos_neg(element[0][0][1]), (abs(element[0][0][2]) // 2 * 2) * pos_neg(element[0][0][2])])
        ideal_array = np.array([(abs(element[0][2][0]) // 2 * 2) * pos_neg(element[0][2][0]), (abs(element[0][2][1]) // 2 * 2) * pos_neg(element[0][2][1]), (abs(element[0][2][2]) // 2 * 2) * pos_neg(element[0][2][2])])
        correct_array = np.array([1 - abs(pos_array[0] - ideal_array[0]) / 4, 1 - abs(pos_array[1] - ideal_array[1]) / 4, 1 - abs(pos_array[2] - ideal_array[2]) / 4])
        correct_array_sum = np.sum(correct_array) // 3
        final_correct_array = np.array([correct_array_sum, correct_array_sum, correct_array_sum])
        temp_array = np.array([pos_array[0], pos_array[1], pos_array[2], (correct_array[0] * 4) + correct_array[1] * 2 + correct_array[2]])
        if i == 0:
            final_array = np.expand_dims(temp_array, 0)
            i += 1
        else:
            final_array = np.vstack([final_array, np.expand_dims(temp_array, 0)])

    i = 0
    final_array_2 = np.array([])
    for x, element in enumerate(position_array_edge):
        pos_array = np.array([(abs(element[0][0]) // 2 * 2) * pos_neg(element[0][0]),
                              (abs(element[0][1]) // 2 * 2) * pos_neg(element[0][1]),
                              (abs(element[0][2]) // 2 * 2) * pos_neg(element[0][2])])
        ideal_array = np.array([(abs(element[2][0]) // 2 * 2) * pos_neg(element[2][0]),
                                (abs(element[2][1]) // 2 * 2) * pos_neg(element[2][1]),
                                (abs(element[2][2]) // 2 * 2) * pos_neg(element[2][2])])
        correct_array = np.array(
            [1 - abs(pos_array[0] - ideal_array[0]) / 4, 1 - abs(pos_array[1] - ideal_array[1]) / 4,
             1 - abs(pos_array[2] - ideal_array[2]) / 4])
        correct_array_sum = np.sum(correct_array) // 3
        final_correct_array = np.array([correct_array_sum, correct_array_sum, correct_array_sum])
        temp_array = np.array([pos_array[0], pos_array[1], pos_array[2], (correct_array[0] * 4) + correct_array[1] * 2 + correct_array[2]])

        if i == 0:
            final_array_2 = np.expand_dims(temp_array, 0)
            i += 1
        else:
            final_array_2 = np.vstack([final_array_2, np.expand_dims(temp_array, 0)])
    return np.vstack([final_array, final_array_2])

def sort_4_array(array):
    array_x = np.array([])
    array_x_counter = 0
    array_y = np.array([])
    array_y_counter = 0
    array_z = np.array([])
    array_z_counter = 0
    for element in array:
        if element[0][0] == 0:
            if array_x_counter == 0:
                array_x = np.expand_dims(element, 0)
                array_x_counter += 1
            else:
                array_x = np.vstack([array_x, np.expand_dims(element, 0)])
        elif element[0][1] == 0:
            if array_y_counter == 0:
                array_y = np.expand_dims(element, 0)
                array_y_counter += 1
            else:
                array_y= np.vstack([array_y, np.expand_dims(element, 0)])
        elif element[0][2] == 0:
            if array_z_counter == 0:
                array_z = np.expand_dims(element, 0)
                array_z_counter += 1
            else:
                array_z = np.vstack([array_z, np.expand_dims(element, 0)])
    final_array = np.vstack([array_x, array_y, array_z])
    return final_array

def check_all_combinations_stage_4(depth, positions):

    forward_dict = {}
    backward_dict = {}
    checker = False
    ideal_position = np.array([[-2, -2, -2, 7],
     [-3, -2,  2, 7],
     [-2, 3, -2, 7],
     [-2, 2, 3, 7],
     [2, -2, -3, 7],
     [2, -2, 3, 7],
     [2, 3, -2, 7],
     [2, 2, 3, 7],
     [-2,-3, 0, 7],
     [-2, 0, -3, 7],
     [-2, 0, 3, 7],
     [-2, 3, 0, 7],
     [0, -2, -3, 7],
     [0, -3, 2, 7],
     [0, 2, -3, 7],
     [0, 2, 3, 7],
     [3,-2, 0, 7],
     [2, 0, -3, 7],
     [3, 0, 2, 7],
     [3, 2, 0, 7]]).astype('int64')
    number_for_testing = 0
    for i in range(1, depth):
        if checker == True:
            break
        else:

            all_situations = dice_combinations_4(i, tuple(stage_four_moves))
        forward_dict = {}
        for x, element in enumerate(all_situations):
            seq, numb = check_position_4(element, positions, 0)
            forward_dict.update({numb: seq})
            forward_dict[numb] = seq
            if i == 1:
                if numb == 1152921504606846975:
                    return seq

        if i > 1:
            move1, move2 = check_bi_direction(forward_dict, backward_dict)
            if move1 != None:
                move1_list = list(move1)
                move2_list = list(move2)
                for x, element in enumerate(move2):
                    move2_list[x] = reverse_letter(element)
                return tuple(move1_list + move2_list[::-1])

        backward_dict = {}
        for element in all_situations:
            seq, numb = check_position_4(element, ideal_position, 1)
            backward_dict[numb] = seq
        move1, move2 = check_bi_direction(forward_dict, backward_dict)
        if move1 != None:
            move1_list = list(move1)
            move2_list = list(move2)
            for x, element in enumerate(move2):
                move2_list[x] = reverse_letter(element)
            return tuple(move1_list + move2_list[::-1])



radius = 15

all_colors_for_net = np.array([[13/255, 72/255, 172/255], [254/255, 243/255, 17/255], [1, 85/255, 37/255], [25/255, 155/255, 76/255], [137/255, 18/255, 20/255], [1, 1, 1]])

all_colors_rgb_values = np.array([[13/255, 72/255, 172/255], [254/255, 243/255, 17/255], [1, 85/255, 37/255], [25/255, 155/255, 76/255], [137/255, 18/255, 20/255], [1, 1, 1]])

all_colors_string_name = np.array(['Yellow', 'Blue', 'Red', 'White', 'Orange', 'Green'])
all_color_numb_values = np.array([6, 3, 4, 1, 2, 5])


for i in range(-2, 3, 2):
    for j in range(-2, 3, 2):
        for k in range(-2, 3, 2):
            if i != 0 or j != 0 or k != 0:
                object_list.append(miniCubes(i, j, k))


# Algorithm
import time
random_moves = random_scramble_list(stage_one_possible_grouped_moves, 50)
current_position = random_scramble(current_position, tuple(random_moves))
array_for_phase_1_center = convert_to_phase_1(1)
current_position = random_scramble(current_position, tuple(random_moves))
all_moves_array = np.array([])
array_for_phase_1_edge = convert_to_phase_1(2)
positions_1 = check_phase_1(array_for_phase_1_edge)
stage_1_beginning = time.time()
stage_one_moves_to_complete = check_all_combinations_stage_1(7, positions_1)
stage_1_end = time.time()
time_1 = stage_1_end - stage_1_beginning
print(f"The total time for stage 1 is {time_1}")

current_position = random_scramble(current_position, stage_one_moves_to_complete)
array_for_phase_1_edge = convert_to_phase_1(2)
positions_1 = check_phase_1(array_for_phase_1_edge)


array_for_phase_1_center = convert_to_phase_1(1)
array_for_phase_2_edge = convert_to_phase_1(2)
array_for_phase_2_corner = convert_to_phase_1(3)
positions_2_unsimplified = check_phase_2(array_for_phase_2_corner, array_for_phase_2_edge)
positions_2_simplified = array_manipulation_2(positions_2_unsimplified)
stage_2_beginning = time.time()
moves_stage_2 = check_all_combinations_stage_2(11, positions_2_simplified)
stage_2_end = time.time()
time_2 = stage_2_end - stage_2_beginning
print(f"The time for stage 2 is {time_2}")
current_position = random_scramble(current_position, moves_stage_2)
array_for_phase_3_edge = convert_to_phase_1(2)
array_for_phase_3_corner = convert_to_phase_1(3)
unsimplified_array_for_phase_3_corner, unsimplified_array_for_phase_3_edge = check_phase_3(array_for_phase_3_corner, array_for_phase_3_edge, 0)
simplified_array_for_phase_3 = array_manipulation_3(unsimplified_array_for_phase_3_corner, unsimplified_array_for_phase_3_edge)
stage_3_beginning = time.time()
moves_stage_3 = check_all_combinations_stage_3(15, simplified_array_for_phase_3)
stage_3_end = time.time()
time_3 = stage_3_end - stage_3_beginning
print(f"The time taken for stage 3 is {time_3}")
current_position = random_scramble(current_position, moves_stage_3)
print(f"solution to complete the first 3 stages: {stage_one_moves_to_complete} {moves_stage_2} {moves_stage_3}")

"""array_for_phase_1_center = convert_to_phase_1(1)
array_for_phase_4_edge = convert_to_phase_1(2)
array_for_phase_4_corner = convert_to_phase_1(3)
unsimplified_array_for_phase_4_corner, unsimplified_array_for_phase_4_edge = check_phase_3(array_for_phase_4_corner, array_for_phase_4_edge, 1)
unsimplified_array_for_phase_4_edge = sort_4_array(unsimplified_array_for_phase_4_edge)
simplified_array_for_phase_4 = sort_position(array_manipulation_4(unsimplified_array_for_phase_4_corner, unsimplified_array_for_phase_4_edge))
simplified_array_for_phase_4 = simplified_array_for_phase_4.astype('int64')

import time
stage_4_beginning = time.time()
move_stage_4 = check_all_combinations_stage_4(15, simplified_array_for_phase_4)
stage_4_end = time.time()
time_4 = stage_4_end - stage_4_beginning
print(f"The time taken for stage 4 is {time_4}")
print(move_stage_4)
current_position = random_scramble(current_position, move_stage_4)
for cube in object_list:
    cube.check_colour()"""


def main(move_up, move_right, move_down, move_left, move_z_1, move_z_2, current_position):
    timer_rotating = 0
    timer_number = 0
    face = 0
    direct = 0
    way = 0
    vertices_rotating = 0
    pygame.init()
    display = ((800, 800))
    background_color = ((1, 105/255, 180/255, 0))


    # Changing surface color
    surface = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    surface.fill(background_color)
    pygame.display.flip()

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0, 0, -radius)
    glRotatef(0,0,0,0)

    for object in object_list:
        object.render_cube()



    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                j = 0


                if event.key == pygame.K_UP:
                    move_up = True
                elif event.key == pygame.K_DOWN:
                    move_down = True
                if event.key == pygame.K_RIGHT:
                    move_right = True
                elif event.key == pygame.K_LEFT:
                    move_left = True
                if event.key == pygame.K_9:
                    move_z_1 = True
                elif event.key == pygame.K_0:
                    move_z_2 = True

                if event.key == pygame.K_c and timer_number == 0:
                    timer_number = 1
                    direct = 1
                    way = 6
                    vertices_rotating = -2
                    current_position = calculate_rotation(0, -1, -1, 0, current_position)
                elif event.key == pygame.K_d and timer_number == 0:
                    timer_number = 3
                    direct = 1
                    way = -6
                    vertices_rotating = -2
                    current_position = calculate_rotation(0, 1, 1, 0, current_position)
                elif event.key == pygame.K_v and timer_number == 0:
                    timer_number = 4
                    direct = 1
                    way = -6
                    vertices_rotating = 2
                    current_position = calculate_rotation(1, 1, -1, 2, current_position)
                elif event.key == pygame.K_f and timer_number == 0:
                    timer_number = 2
                    direct = 1
                    way = 6
                    vertices_rotating = 2
                    current_position = calculate_rotation(1, -1, 1, 2, current_position)
                elif event.key == pygame.K_b and timer_number == 0:
                    timer_number = 8
                    direct = 2
                    way = -6
                    vertices_rotating = 2
                    current_position = calculate_rotation(2, -1, -1, 0, current_position)
                elif event.key == pygame.K_g and timer_number == 0:
                    timer_number = 6
                    direct = 2
                    way = 6
                    vertices_rotating = 2
                    current_position = calculate_rotation(2, 1, 1, 0, current_position)
                elif event.key == pygame.K_n and timer_number == 0:
                    timer_number = 5
                    direct = 2
                    way = 6
                    vertices_rotating = -2
                    current_position = calculate_rotation(3, 1, -1, 2, current_position)
                elif event.key == pygame.K_h and timer_number == 0:
                    timer_number = 7
                    direct = 2
                    way = -6
                    vertices_rotating = -2
                    current_position = calculate_rotation(3, -1, 1, 2, current_position)
                elif event.key == pygame.K_m and timer_number == 0:
                    timer_number = 12
                    direct = 3
                    way = -6
                    vertices_rotating = 2
                    current_position = calculate_rotation(4, 1, -1, 0, current_position)
                elif event.key == pygame.K_j and timer_number == 0:
                    timer_number = 10
                    direct = 3
                    way = 6
                    vertices_rotating = 2
                    current_position = calculate_rotation(4, -1, 1, 0, current_position)
                elif event.key == pygame.K_k and timer_number == 0:
                    timer_number = 11
                    direct = 3
                    way = -6
                    vertices_rotating = -2
                    current_position = calculate_rotation(5, 1, 1, 2, current_position)
                elif event.key == pygame.K_COMMA and timer_number == 0:
                    timer_number = 9
                    direct = 3
                    way = 6
                    vertices_rotating = -2
                    current_position = calculate_rotation(5, -1, -1, 2, current_position)
                elif event.key == pygame.K_a and timer_number == 0:
                    print(list(current_position))
                elif event.key == pygame.K_s and timer_number == 0:
                    array_for_phase_2_edge_temp = convert_to_phase_1(2)
                    array_for_phase_2_corner_temp = convert_to_phase_1(3)
                    positions_2_unsimplified_temp = check_phase_2(array_for_phase_2_corner_temp, array_for_phase_2_edge_temp)
                    final_pos_temp = sort_position(array_manipulation_2(positions_2_unsimplified_temp))
                    print(final_pos_temp)
                elif event.key == pygame.K_z and timer_number == 0:
                    array_for_phase_3_edge_temp = convert_to_phase_1(2)
                    array_for_phase_3_corner_temp = convert_to_phase_1(3)
                    unsimplified_array_for_phase_3_corner_temp, unsimplified_array_for_phase_3_edge_temp = check_phase_3(
                    array_for_phase_3_corner_temp, array_for_phase_3_edge_temp, 0)
                    simplified_array_for_phase_3_temp = sort_position(array_manipulation_3(unsimplified_array_for_phase_3_corner_temp,
                                                                             unsimplified_array_for_phase_3_edge_temp))
                    print(simplified_array_for_phase_3_temp)
                elif event.key == pygame.K_q and timer_number == 0:
                    array_for_phase_4_edge_temp = convert_to_phase_1(2)
                    array_for_phase_4_corner_temp = convert_to_phase_1(3)
                    unsimplified_array_for_phase_4_corner_temp, unsimplified_array_for_phase_4_edge_temp = check_phase_3(
                        array_for_phase_4_corner_temp, array_for_phase_4_edge_temp, 1)
                    unsimplified_array_for_phase_4_edge_temp = sort_4_array(unsimplified_array_for_phase_4_edge_temp)
                    simplified_array_for_phase_4_temp = sort_position(
                        array_manipulation_4(unsimplified_array_for_phase_4_corner_temp,
                                             unsimplified_array_for_phase_4_edge_temp))
                    print(simplified_array_for_phase_4_temp)




            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    move_up = False
                elif event.key == pygame.K_DOWN:
                    move_down = False
                if event.key == pygame.K_RIGHT:
                    move_right = False
                elif event.key == pygame.K_LEFT:
                    move_left = False
                if event.key == pygame.K_9:
                    move_z_1 = False
                elif event.key == pygame.K_0:
                    move_z_2 = False

        """if move_up == True and move_down == False:
            glTranslatef(0, 0, -radius)
            glRotatef(0, 0, 1, 0)
            glRotatef(1, -1, 0, 0)
            glTranslatef(0, -math.sin(1 / 180 * math.pi) * radius, radius)
        elif move_down == True:
            glTranslatef(0, 0, -radius)
            glRotatef(0, 0, 1, 0)
            glRotatef(1, 1, 0, 0)
            glTranslatef(0, math.sin(1 / 180 * math.pi) * radius, radius)
        if move_right == True and move_left == False:
            glTranslatef(0, 0, -radius)
            glRotatef(1, 0, 1, 0)
            glRotatef(0, -1, 0, 0)
            glTranslatef(-math.sin(1 / 180 * math.pi) * radius, 0, radius)
        if move_left == True:
            glTranslatef(0, 0, -radius)
            glRotatef(1, 0, -1, 0)
            glRotatef(0, 1, 0, 0)
            glTranslatef(math.sin(1 / 180 * math.pi) * radius, 0, radius)"""

        if timer_number != 0:
            timer_rotating += 1
            for object in object_list:
                if direct == 1:
                    if object.x_pos == vertices_rotating:
                        object.rotate(np.array([[1, 0, 0],
                                    [0, (math.cos(way / 180 * math.pi)),
                                     -(math.sin(way / 180 * math.pi))],
                                    [0, (math.sin(way / 180 * math.pi)),
                                     (math.cos(way / 180 * math.pi))]]))
                if direct == 2:
                    if object.y_pos == vertices_rotating:
                        object.rotate(np.array(
                [[(math.cos(way / 180 * math.pi)), 0, (math.sin(way / 180 * math.pi))],
                 [0, 1, 0],
                 [-(math.sin(way / 180 * math.pi)), 0, (math.cos(way / 180 * math.pi))]]))
                if direct == 3:
                    if object.z_pos == vertices_rotating:
                        object.rotate(np.array(
                [[(math.cos(way / 180 * math.pi)), -(math.sin(way / 180 * math.pi)), 0],
                 [(math.sin(way / 180 * math.pi)), (math.cos(way / 180 * math.pi)), 0],
                 [0, 0, 1]]))

            if timer_rotating == 15:
                for object in object_list:
                    if direct == 1:
                        if object.x_pos == vertices_rotating:
                            object.update_coords()
                    if direct == 2:
                        if object.y_pos == vertices_rotating:
                            object.update_coords()
                    if direct == 3:
                        if object.z_pos == vertices_rotating:
                            object.update_coords()
                timer_number = timer_rotating = direct = way = vertices_rotating = 0





        if (move_up):
            glRotatef(1.5, 1, 0, 0)
        if (move_down):
            glRotatef(1.5, -1, 0, 0)
        if move_right:
            glRotatef(1.5, 0, -1, 0)
        if move_left:
            glRotatef(1.5, 0, 1, 0)
        if move_z_1:
            glRotatef(1.5, 0, 0, 1)
        if move_z_2:
            glRotatef(1.5, 0, 0, -1)



        glClearColor(1, 105/255, 180/255, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        for object in object_list:
            object.render_cube()
        rect = pygame.draw.rect(surface, (255, 255, 255), (200, 200, 800, 800))

        pygame.display.flip()
        pygame.time.wait(10)



main(move_up, move_right, move_down, move_left, move_z_1, move_z_2, current_position)

# x, y = pygame.mouse.get_rel()

"""
if event.type == pygame.KEYDOWN:
    if event.key == pygame.K_UP:
        glRotatef(1, 2, 0, 0)
    elif event.key == pygame.K_DOWN:
        glRotatef(1, -2, 0, 0)
    if event.key == pygame.K_RIGHT:
        glRotatef(1, 0, 2, 0)
    elif event.key == pygame.K_LEFT:
        glRotatef(1, 0, -2, 0)
"""

"""                if event.key == pygame.K_c:
                    for object in object_list:
                        if object.x_pos == -2:
                            object.rotate(1, 90)
                elif event.key == pygame.K_d:
                    for object in object_list:
                        if object.x_pos == -2:
                            object.rotate(1, -90)
                elif event.key == pygame.K_v:
                    for object in object_list:
                        if object.x_pos == 2:
                            object.rotate(1, 90)
                elif event.key == pygame.K_f:
                    for object in object_list:
                        if object.x_pos == 2:
                            object.rotate(1, -90)
                elif event.key == pygame.K_b:
                    for object in object_list:
                        if object.y_pos == -2:
                            object.rotate(2, 90)
                elif event.key == pygame.K_g:
                    for object in object_list:
                        if object.y_pos == -2:
                            object.rotate(2, -90)
                elif event.key == pygame.K_n:
                    for object in object_list:
                        if object.y_pos == 2:
                            object.rotate(2, 90)
                elif event.key == pygame.K_h:
                    for object in object_list:
                        if object.y_pos == 2:
                            object.rotate(2, -90)
                elif event.key == pygame.K_m:
                    for object in object_list:
                        if object.z_pos == -2:
                            object.rotate(3, 90)
                elif event.key == pygame.K_j:
                    for object in object_list:

                        if object.z_pos == -2:
                            object.rotate(3, -90)
                elif event.key == pygame.K_COMMA:
                    for object in object_list:
                        if object.z_pos == 2:
                            object.rotate(3, 90)
                elif event.key == pygame.K_k:
                    for object in object_list:
                        if object.z_pos == 2:
                            object.rotate(3, -90)"""
"""                if event.key == pygame.K_c:
                    for object in object_list:
                        if object.x_pos == -2:
                            object.rotate(1, 90)
                elif event.key == pygame.K_d:
                    for object in object_list:
                        if object.x_pos == -2:
                            object.rotate(1, -90)
                elif event.key == pygame.K_v:
                    for object in object_list:
                        if object.x_pos == 2:
                            object.rotate(1, 90)
                elif event.key == pygame.K_f:
                    for object in object_list:
                        if object.x_pos == 2:
                            object.rotate(1, -90)
                elif event.key == pygame.K_b:
                    for object in object_list:
                        if object.y_pos == -2:
                            object.rotate(2, 90)
                elif event.key == pygame.K_g:
                    for object in object_list:
                        if object.y_pos == -2:
                            object.rotate(2, -90)
                elif event.key == pygame.K_n:
                    for object in object_list:
                        if object.y_pos == 2:
                            object.rotate(2, 90)
                elif event.key == pygame.K_h:
                    for object in object_list:
                        if object.y_pos == 2:
                            object.rotate(2, -90)
                elif event.key == pygame.K_m:
                    for object in object_list:
                        if object.z_pos == -2:
                            object.rotate(3, 90)
                elif event.key == pygame.K_j:
                    for object in object_list:

                        if object.z_pos == -2:
                            object.rotate(3, -90)
                elif event.key == pygame.K_COMMA:
                    for object in object_list:
                        if object.z_pos == 2:
                            object.rotate(3, 90)
                elif event.key == pygame.K_k:
                    for object in object_list:
                        if object.z_pos == 2:
                            object.rotate(3, -90)
"""
