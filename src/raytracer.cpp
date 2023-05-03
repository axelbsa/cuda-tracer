#include <iostream>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_helpers/helper_cuda.h"
#include "cuda_helpers/helper_gl.h"
#include "cuda_helpers/helper_math.h"

#include "raytracer.h"

// clang++ x11_opengl_window.cpp -o x11_opengl_window -lGL -lX11

// XXX TODO MOVE GLOBALS SOMEWHERE ELSE
// X11 Windowing vars
Display* display;
Window window;
Screen* screen;
XEvent ev;
XVisualInfo* visual;
XSetWindowAttributes windowAttribs;
GLXContext context;
int screenId;

GLuint pbo;  // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource = NULL;
uint *d_output; // device output buffer (d_ for device)

// Image
int imageW = 960;
int imageH = 540;
dim3 windowSize(imageW, imageH);
dim3 threads(8, 8); // 2 * https://xkcd.com/221/
const auto aspect_ratio = 16.0f / 9.0f;
const int image_width = windowSize.x;
const int image_height = static_cast<int>(image_width / aspect_ratio);

// Camera
auto viewport_height = 2.0f;
auto viewport_width = aspect_ratio * viewport_height;
auto focal_length = 1.0f;
auto origin = make_float3(0.0f, 0.0f, 0.0f);
auto horizontal = make_float3(viewport_width, 0.0f, 0.0f);
auto vertical = make_float3(0.0f, viewport_height, 0.0f);
auto lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - make_float3(0.0f, 0.0f, focal_length);

extern "C" void trace(
        dim3 blocks,
        dim3 threads,
        uint *d_output,
        uint imageW,
        uint imageH,
        float3 lower_left_corner,
        float3 horizontal,
        float3 vertical,
        float3 origin
);

void init_gl_buffers(dim3 windowSize) {
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

    glBufferData(
        GL_PIXEL_UNPACK_BUFFER_ARB,
        windowSize.x * windowSize.y * sizeof(GLubyte) * 4,
        0,
        GL_STREAM_DRAW_ARB
    );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(
        cudaGraphicsGLRegisterBuffer(
            &cuda_pbo_resource,
            pbo,
            cudaGraphicsMapFlagsWriteDiscard
        )
    );
}

inline void x11_handle_keyboard_events(Display *display, XEvent ev, bool *running)
{
    char str[25] = {0};
    int len = 0;
    KeySym keysym = 0;
    XNextEvent(display, &ev);
    switch (ev.type)
    {
        case Expose:
            XWindowAttributes attribs;
            XGetWindowAttributes(display, window, &attribs);
            glViewport(0, 0, attribs.width, attribs.height);
            break;
        case KeymapNotify:
            XRefreshKeyboardMapping(&ev.xmapping);
            break;
        case KeyPress:
            len = XLookupString(&ev.xkey, str, 25, &keysym, NULL);
#ifdef DEBUG
            if (len > 0)
                std::cerr << "Key pressed: " << str << " - " << len << " - " << keysym << " . " << XK_Escape <<'\n';
#endif
            if (keysym == XK_Escape) {
                *running = false;
            }
            break;
        case KeyRelease:
            len = XLookupString(&ev.xkey, str, 25, &keysym, NULL);
            break;
        default:
            break;
    }
}

void x11_destroy_opengl_window()
{
	// Cleanup GLX
	glXDestroyContext(display, context);

	// Cleanup X11
	XFree(visual);
	XFreeColormap(display, windowAttribs.colormap);
	XDestroyWindow(display, window);
	XCloseDisplay(display);
}

int x11_init_opengl_window(int imageW, int imageH)
{
	// Open the display
	display = XOpenDisplay(NULL);
	if (display == NULL) {
		std::cout << "Could not open display\n";
		return 1;
	}
	screen = DefaultScreenOfDisplay(display);
	screenId = DefaultScreen(display);

	// Check GLX version
	GLint majorGLX, minorGLX = 0;
	glXQueryVersion(display, &majorGLX, &minorGLX);
	if (majorGLX <= 2 && minorGLX < 1) {
		std::cout << "GLX 2.1 or greater is required.\n";
		XCloseDisplay(display);
		return 1;
}
	else {
		std::cout << "GLX version: " << majorGLX << "." << minorGLX << '\n';
	}

	// GLX, create XVisualInfo, this is the minimum visuals we want
	GLint glxAttribs[] = {
		GLX_RGBA,
		GLX_DOUBLEBUFFER,
		GLX_DEPTH_SIZE,     24,
		GLX_STENCIL_SIZE,   8,
		GLX_RED_SIZE,       8,
		GLX_GREEN_SIZE,     8,
		GLX_BLUE_SIZE,      8,
		GLX_SAMPLE_BUFFERS, 0,
		GLX_SAMPLES,        0,
		None
	};
	visual = glXChooseVisual(display, screenId, glxAttribs);

	if (visual == 0) {
		std::cout << "Could not create correct visual window.\n";
		XCloseDisplay(display);
		return 1;
	}

	// Open the window
	windowAttribs.border_pixel = BlackPixel(display, screenId);
	windowAttribs.background_pixel = WhitePixel(display, screenId);
	windowAttribs.override_redirect = True;
	windowAttribs.colormap = XCreateColormap(display, RootWindow(display, screenId), visual->visual, AllocNone);
	windowAttribs.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask;
	window = XCreateWindow(
            display,
            RootWindow(display, screenId),
            0, 0, imageW, imageH, 0, 
            visual->depth, InputOutput, 
            visual->visual, 
            CWBackPixel | CWColormap | CWBorderPixel | CWEventMask, &windowAttribs
    );

	// Create GLX OpenGL context
	context = glXCreateContext(display, visual, NULL, GL_TRUE);
	glXMakeCurrent(display, window, context);

	std::cout << "GL Vendor: " << glGetString(GL_VENDOR) << "\n";
	std::cout << "GL Renderer: " << glGetString(GL_RENDERER) << "\n";
	std::cout << "GL Version: " << glGetString(GL_VERSION) << "\n";
	std::cout << "GL Shading Language: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

	// Show the window
	XClearWindow(display, window);
	XMapRaised(display, window);

    return 0;
}

void render() {
    size_t num_bytes;

    // map PBO to get CUDA device pointer
    checkCudaErrors(
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0)
    );

    checkCudaErrors(
        cudaGraphicsResourceGetMappedPointer(
            (void **)&d_output,
            &num_bytes,
            cuda_pbo_resource
        )
    );

    //Hittable_list world;
    //world.add(make_shared<Sphere>(make_float3(0.0f, 0.0f, -1.0f), 0.5f));
    //world.add(make_shared<Sphere>(make_float3(0.0f, -100.5f,-1.0f), 100.0f));

    // call CUDA kernel, writing results to PBO
    trace(
        windowSize, threads, d_output, windowSize.x, windowSize.y,
        lower_left_corner, horizontal, vertical, origin
    );

    //getLastCudaError("render_kernel failed");
    checkCudaErrors(
        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0)
    );
}


int main(int argc, char** argv) {


    if (x11_init_opengl_window(imageW, imageH))
    {
        std::cerr << "[Error] Fatal error initializing X11 display \n";
    }

    findCudaDevice(argc, (const char **)argv);

    init_gl_buffers(windowSize);

	// Set GL Sample stuff
	glClearColor(0.5f, 0.6f, 0.7f, 0.1f);

    bool running = true;

	while (running) {

        x11_handle_keyboard_events(display, ev, &running);

		// Present frame
        glViewport(0, 0, windowSize.x, windowSize.y);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        // display results
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.4, 0.1, 0.6, 1.0);  // Set some annoying colour that is neither white nor black
        
        // Setup buffer and call cuda
        render();

        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(0, 0);

        // We bind to the buffer, draw, then release
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glDrawPixels(windowSize.x, windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		glXSwapBuffers(display, window);

	}

    x11_destroy_opengl_window();

	return 0;
}
