# PyQt4 imports
from PyQt5 import QtGui, QtCore, QtOpenGL, QtWidgets, Qt
from PyQt5.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import matplotlib.pyplot as plt, matplotlib.cm as cm
import threading
import time

from analyzetests import loadTrial1DataSets, loadSetImages
from wftools.datatypes import zernike_wfe, optical_setup
from imagetools import preprocess, featureextraction
from algorithm import finitedifferences
from displaytools import wavefrontvisshader

# Window creation function.
def create_window(window_class):
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



class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 600, 600
    ph, th = 20, 330
    lightth = 0

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0, 0, 0, 0)
        # create a Vertex Buffer Object with the specified data
        # compile the vertex shader
        vs = compile_vertex_shader(wavefrontvisshader.VS)
        # compile the fragment shader
        fs = compile_fragment_shader(wavefrontvisshader.FS)
        # compile the vertex shader
        self.shaders_program = link_shader_program(vs, fs)

        self.zernikies = 6*[0]
        self.maxval = 1
        self.minval = -1
        self.wireframe = False
        self.dim = 1.0
        gl.glClearDepth(1.0)
        gl.glClearColor(0.6, 0.7, 0.7, 1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def paintGL(self):
        w = self.width*self.devicePixelRatio()
        h = self.height*self.devicePixelRatio()
        asp = w/h
        gl.glViewport(0,0,w,h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        zmin = self.dim/16
        zmax = 16*self.dim
        ydim = zmin*np.tan(55*np.pi/360)
        xdim = ydim*asp
        gl.glFrustum(-xdim, +xdim, -ydim, +ydim, zmin, zmax)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslated(0, 0, -2*self.dim)
        gl.glRotated(self.ph,1,0,0)
        gl.glRotated(self.th,0,1,0)

        """Paint the scene."""
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_LIGHTING)

        ## Draw Axes
        center = np.array([-0.5, -0.5, -0.5])
        xend = center + np.array([1, 0, 0])
        yend = center + np.array([0, 1, 0])
        zend = center + np.array([0, 0, 1])
        gl.glUseProgram(0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor(0.0,0.0,0.0,0.0)
        gl.glVertex3f(center[0], center[1], center[2])
        gl.glVertex3f(xend[0], xend[1], xend[2])
        gl.glVertex3f(center[0], center[1], center[2])
        gl.glVertex3f(yend[0], yend[1], yend[2])
        gl.glVertex3f(center[0], center[1], center[2])
        gl.glVertex3f(zend[0], zend[1], zend[2])
        gl.glEnd()

        ## Do lighting
        gl.glColor3f(1,1,1)
        gl.glPushMatrix()
        gl.glTranslated(1.0*np.cos(self.lightth *np.pi/180),
                                                    1.5, 1.0 *np.sin(self.lightth*np.pi/180))
        gl.glScaled(0.1, 0.1, 0.1)
        gl.glPointSize(10)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(0,0,0)
        gl.glEnd()
        gl.glPopMatrix()

        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1.0*np.cos(self.lightth *np.pi/180),
                                                    1.5, 1.0 *np.sin(self.lightth*np.pi/180)])


        gl.glUseProgram(self.shaders_program)
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z1'), np.float32(self.zernikies[0]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z2'), np.float32(self.zernikies[1]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z3'), np.float32(self.zernikies[2]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z4'), np.float32(self.zernikies[3]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z5'), np.float32(self.zernikies[4]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'z6'), np.float32(self.zernikies[5]))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'max'), np.float32(self.maxval))
        gl.glUniform1f(gl.glGetUniformLocation( self.shaders_program, 'min'), np.float32(self.minval))


        # Draw wavefront
        rt_to_xy = lambda r, t: (np.cos(t)*r, np.sin(t)*r)
        radiusSteps = 0.1
        thetaSteps = 2*np.pi/32
        # draw center of circle
        radius = radiusSteps
        if self.wireframe:
            gl.glBegin(gl.GL_LINES)
        else:
            gl.glBegin(gl.GL_TRIANGLES)
        for theta in reversed(np.arange(0, 2*np.pi, thetaSteps)):
            theta2 = theta + thetaSteps
            p1 = rt_to_xy(0, theta)
            p2 = rt_to_xy(radius, theta)
            p3 = rt_to_xy(radius, theta2)
            gl.glVertex3f(p1[0], 0, p1[1])
            gl.glVertex3f(p2[0], 0, p2[1])
            gl.glVertex3f(p3[0], 0, p3[1])
        gl.glEnd()

        #draw rest of circle
        if self.wireframe:
            gl.glBegin(gl.GL_LINES)
        else:
            gl.glBegin(gl.GL_QUADS)
        for radius1 in np.arange(radiusSteps, 0.7, radiusSteps):
            for theta1 in np.arange(0, 2*np.pi + thetaSteps, thetaSteps):
                radius2 = radius1 + radiusSteps
                theta2 = theta1 + thetaSteps
                p1 = rt_to_xy(radius1, theta1)
                p2 = rt_to_xy(radius1, theta2)
                p3 = rt_to_xy(radius2, theta2)
                p4 = rt_to_xy(radius2, theta1)
                gl.glVertex3f(p1[0], 0, p1[1])
                gl.glVertex3f(p2[0], 0, p2[1])
                gl.glVertex3f(p3[0], 0, p3[1])
                gl.glVertex3f(p4[0], 0, p4[1])
                if self.wireframe:
                    gl.glVertex3f(p2[0], 0, p2[1])
                    gl.glVertex3f(p3[0], 0, p3[1])
                    gl.glVertex3f(p4[0], 0, p4[1])
                    gl.glVertex3f(p1[0], 0, p1[1])
        gl.glEnd()


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

    def spin(self):
        self.lightth += 2
        self.lightth %= 360
        self.updateGL()

    def setWireframe(self, wireframe = True):
        self.wireframe = wireframe
        self.updateGL()

    def mousePressEvent(self, event):
        self.oldPos = event.pos()

    def mouseMoveEvent(self, event):
        dpos = event.pos() - self.oldPos
        self.th = (self.th - dpos.x()) % 360
        self.ph = (self.ph - dpos.y()) % 360
        self.oldPos = event.pos()
        self.updateGL()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.dim += 0.1
        else:
            self.dim -= 0.1

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
            layout.setColumnStretch(0,800)
            layout.setColumnMinimumWidth(0,800)
            window.setLayout(layout)


            #Matplotlib figures
            self.prefocalfig = Figure()
            self.postfocalfig = Figure()
            self.wave2dfig = Figure()
            self.barfig = Figure()
            self.prefocalcanvas = FigureCanvas(self.prefocalfig)
            self.postfocalcanvas = FigureCanvas(self.postfocalfig)
            self.wave2dcanvas = FigureCanvas(self.wave2dfig)
            self.barcanvas = FigureCanvas(self.barfig)

            layout.addWidget(self.prefocalcanvas, 1,1, 5, 1)
            layout.addWidget(self.postfocalcanvas, 1,2, 5, 1)
            layout.addWidget(self.wave2dcanvas, 5,1, 5, 1)
            layout.addWidget(self.barcanvas, 5,2, 5, 1)

            self.trialLabel = QtWidgets.QLabel("None")
            layout.addWidget(self.trialLabel, 10,0)

            run_tests_button = QtWidgets.QPushButton("Run Tests")
            run_tests_button.clicked.connect(self.run_tests)
            layout.addWidget(run_tests_button, 10,1)

            self.mode_butt = QtWidgets.QPushButton("Wireframe")
            self.mode_butt.clicked.connect(self.handleDisplayMode)
            layout.addWidget(self.mode_butt, 10, 2)

            # Core animation timer
            timer = QtCore.QTimer(self.widget)
            timer.timeout.connect(self.widget.spin)
            timer.start(10)

            self.setCentralWidget(window)
            self.show()
            return

        def handleDisplayMode(self):
            if self.mode_butt.text() == "Wireframe":
                self.mode_butt.setText("Solid")
                self.widget.setWireframe(True)
            else:
                self.mode_butt.setText("Wireframe")
                self.widget.setWireframe(False)

        def run_tests(self):
            self.stop_event=threading.Event()
            self.c_thread=threading.Thread(target=self.thread_run_tests, args=(self.stop_event,))
            self.c_thread.start()

        def thread_run_tests(self, stopevent):
            for i in range(1, 5):
                self.dispTrial(i)
                time.sleep(10)

        def processTrial(self, trialNum):
            # Load trial images
            sets = loadTrial1DataSets()
            pre_im, pos_im = loadSetImages(sets[trialNum])

            # Set up optics parameters
            wfe = zernike_wfe([0, 0, 0.07])
            opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, 1e-3, 0.05)

            # Preprocess images
            pre_im, pos_im = preprocess.crop(pre_im, pos_im)
            pre_raw, pos_raw = pre_im, pos_im
            pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 21)
            pre_im, pos_im = preprocess.normalize(pre_im, pos_im)

            # Extract Features
            coms, pre_im, pos_im = featureextraction.parse_tip_tilt(pre_im, pos_im, opt)
            masks = featureextraction.create_masks(pre_im, pos_im)
            laplacian = featureextraction.extract_laplacians(pre_im, pos_im, masks, opt)
            normals = featureextraction.extract_normals(pre_im, pos_im, masks, opt)

            # Recreate wavefront and parse zernike coefficients
            wavefront = finitedifferences.solve_wavefront(laplacian, normals)
            recovered = wfe.from_image(wavefront)
            coef = recovered.coef
            waveimg = recovered.as_image(256, 256, 1, True)

            return coef, pre_raw, pos_raw, waveimg

        def dispTrial(self, trialNum):
            self.trialLabel.setText("Computing...")
            coef, pre_raw, pos_raw, img = self.processTrial(trialNum)
            self.trialLabel.setText("Trial {}".format(trialNum))

            # Create non 3d Plots
            # create an axis
            ax = self.prefocalfig.gca()
            ax.clear()
            # plot data
            ax.imshow(pre_raw, cmap=cm.gray, interpolation='none')
            ax.set_title("Prefocal Image")
            ax = self.postfocalfig.gca()
            ax.clear()
            # plot data
            ax.imshow(pos_raw, cmap=cm.gray, interpolation='none')
            ax.set_title("Postfocal Image")
            ax = self.wave2dfig.gca()
            ax.clear()
            # plot data
            ax.set_title("2D Reconstructed Wavefront")
            ax.imshow(img, cmap=cm.hot, interpolation='none')

            #Scale coefficients by maximum
            coef = np.array(coef)
            coef = coef / (np.max(np.abs(coef)))

            colors = []
            thresh = 0.1
            for i, val in enumerate(coef):
                if np.abs(val) > thresh:
                    colors.append('r')
                else:
                    colors.append('g')

            ax = self.barfig.gca()
            ax.bar(range(len(coef)), coef, color=colors)
            ax.set_xticklabels([str(i) for i in range(7)])
            ax.set_title("Zernike Coefficient Amplitudes")

            self.prefocalcanvas.draw()
            self.postfocalcanvas.draw()
            self.wave2dcanvas.draw()
            self.barcanvas.draw()
            self.widget.zernikies = coef
            self.widget.max = np.max(img)
            self.widget.min = np.min(img)

            return

    # show the window
    win = create_window(TestWindow)
