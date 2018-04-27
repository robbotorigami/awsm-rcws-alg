# PyQt4 imports
from PyQt5 import QtGui, QtCore, QtOpenGL, QtWidgets, Qt
from PyQt5.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import matplotlib.pyplot as plt, matplotlib.cm as cm

from analyzetests import loadTrial1DataSets, loadSetImages
from wftools.datatypes import zernike_wfe, optical_setup
from imagetools import preprocess, featureextraction
from algorithm import finitedifferences

# Window creation function.
def create_window(window_class):
    """Create a Qt window in Python, or interactively in IPython with Qt GUI
    event loop integration:
        # in ~/.ipython/ipython_config.py
        c.TerminalIPythonApp.gui = 'qt'
        c.TerminalIPythonApp.pylab = 'qt'
    See also:
        http://ipython.org/ipython-doc/dev/interactive/qtconsole.html#qt-and-the-qtconsole
    """
    app_created = False
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app_created = True
    app.references = set()
    window = window_class()
    app.references.add(window)
    window.show()
    if app_created:
        app.exec_()
    return window

def compile_vertex_shader(source):
    """Compile a vertex shader from source."""
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, source)
    gl.glCompileShader(vertex_shader)
    # check compilation error
    result = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(vertex_shader))
    return vertex_shader

def compile_fragment_shader(source):
    """Compile a fragment shader from source."""
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, source)
    gl.glCompileShader(fragment_shader)
    # check compilation error
    result = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(fragment_shader))
    return fragment_shader

def link_shader_program(vertex_shader, fragment_shader):
    """Create a shader program with from compiled shaders."""
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program

# Vertex shader
VS = """

uniform float z1;
uniform float z2;
uniform float z3;
uniform float z4;
uniform float z5;
uniform float z6;

float z1_mag(float rho, float theta) {return z1;}
float z2_mag(float rho, float theta) {return z2*2.0*rho*cos(theta);}
float z3_mag(float rho, float theta) {return z3*2.0*rho*sin(theta);}
float z4_mag(float rho, float theta) {return z4*sqrt(3.0)*(2.0*rho*rho-1.0);}
float z5_mag(float rho, float theta) {return z5*sqrt(6.0)*rho*rho*sin(2.0*theta);}
float z6_mag(float rho, float theta) {return z6*sqrt(6.0)*rho*rho*cos(2.0*theta);}

varying vec4 color;

void main()
{
    float rho = sqrt(gl_Vertex.x*gl_Vertex.x+gl_Vertex.z*gl_Vertex.z);
    float theta = atan(gl_Vertex.z, gl_Vertex.x);
    float mag = z1_mag(rho, theta) + z2_mag(rho, theta) + z3_mag(rho, theta)
     + z4_mag(rho, theta) + z5_mag(rho, theta) + z6_mag(rho, theta);
    mag = mag * 5.0;
    color = vec4(mag + 3.0, 0.0, -mag, 1.0);
    //  Set vertex position
    gl_Position = gl_ModelViewProjectionMatrix * (gl_Vertex + 0.1*vec4(0.0, mag, 0.0, 1.0));
}
"""

# Fragment shader
FS = """
varying vec4 color;
void main()
{
   gl_FragColor = color;
}
"""

class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 600, 600
    ph, th = 20, 0

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0, 0, 0, 0)
        # create a Vertex Buffer Object with the specified data
        # compile the vertex shader
        vs = compile_vertex_shader(VS)
        # compile the fragment shader
        fs = compile_fragment_shader(FS)
        # compile the vertex shader
        self.shaders_program = link_shader_program(vs, fs)

    def paintGL(self):
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_PROJECTION);
        gl.glRotated(self.ph,1,0,0)
        gl.glRotated(self.th,0,1,0)

        """Paint the scene."""
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.shaders_program)
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z1'), np.float32(self.zernikies[0]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z2'), np.float32(self.zernikies[1]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z3'), np.float32(self.zernikies[2]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z4'), np.float32(self.zernikies[3]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z5'), np.float32(self.zernikies[4]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z6'), np.float32(self.zernikies[5]))
        # draw "count" points from the VBO
        step = 0.05
        for i in np.arange(-0.5, 0.5, step):
            for j in np.arange(-0.5, 0.5, step):
                if (i**2 + j**2) < 0.5**2:
                    gl.glBegin(gl.GL_QUADS)
                    gl.glVertex3f(i, 0, j)
                    gl.glVertex3f(i+step, 0, j)
                    gl.glVertex3f(i+step, 0, j+step)
                    gl.glVertex3f(i, 0, j+step)
                    gl.glEnd()


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

    def spin(self):
        self.th += 1
        self.updateGL()

if __name__ == '__main__':
    # import numpy for generating random data points
    import sys
    import numpy as np

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure


    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()

            window = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout()
            # initialize the GL widget
            self.widget = GLPlotWidget()
            self.widget.updateGL()
            layout.addWidget(self.widget,0,0,10,1)
            layout.setColumnStretch(0,400)
            layout.setColumnMinimumWidth(0,400)
            window.setLayout(layout)


            # a figure instance to plot on
            self.figure = Figure()
            # this is the Canvas Widget that displays the `figure`
            # it takes the `figure` instance as a parameter to __init__
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas, 1,1)


            #Run the processing code
            sets = loadTrial1DataSets()
            pre_im, pos_im = loadSetImages(sets[2])
            wfe = zernike_wfe([0, 0, 0.07])
            opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, 1e-3, 0.05)
            pre_im, pos_im = preprocess.crop(pre_im, pos_im)
            pre_raw, pos_raw = pre_im, pos_im
            pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 21)
            pre_im, pos_im = preprocess.normalize(pre_im, pos_im)
            coms, pre_im, pos_im = featureextraction.parse_tip_tilt(pre_im, pos_im, opt)
            masks = featureextraction.create_masks(pre_im, pos_im)
            laplacian = featureextraction.extract_laplacians(pre_im, pos_im, masks, opt)
            normals = featureextraction.extract_normals(pre_im, pos_im, masks, opt)
            wavefront = finitedifferences.solve_wavefront(laplacian, normals)
            recovered = wfe.from_image(wavefront)
            img = recovered.as_image(512, 512, 1, True)
            coef = recovered.coef
            coef[0] = 0
            self.widget.zernikies = 100*np.array(coef)


            # Create non 3d Plots
            # create an axis
            ax = self.figure.add_subplot(221)
            ax.clear()
            # plot data
            ax.imshow(pre_raw, cmap=cm.gray, interpolation='none')
            self.canvas.draw()
            ax = self.figure.add_subplot(222)
            ax.clear()
            # plot data
            ax.imshow(pos_raw, cmap=cm.gray, interpolation='none')
            self.canvas.draw()
            ax = self.figure.add_subplot(223)
            ax.clear()
            # plot data
            ax.imshow(img, cmap=cm.hot, interpolation='none')
            self.canvas.draw()

            recovered.coef[1] -= coms[0]
            recovered.coef[2] -= coms[1]

            coef = recovered.coef
            coef = np.array(coef)

            coef = coef / (np.max(np.abs(coef)))

            colors = []

            thresh = 0.1
            for i, val in enumerate(coef):
                if np.abs(val) > thresh:
                    colors.append('r')
                else:
                    colors.append('g')

            ax = self.figure.add_subplot(224)
            ax.bar(range(len(coef)), coef, color=colors)
            ax.set_xticklabels([str(i) for i in range(7)])


            timer = QtCore.QTimer(self.widget)
            timer.timeout.connect(self.widget.spin)
            timer.start(10)

            self.setCentralWidget(window)
            self.show()
            return

    # show the window
    win = create_window(TestWindow)