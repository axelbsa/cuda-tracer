#include <iostream>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "cuda_helpers/helper_cuda.h"
#include "cuda_helpers/helper_gl.h"
#include "cuda_helpers/helper_math.h"

#include "raytracer.h"

// clang++ x11_opengl_window.cpp -o x11_opengl_window -lGL -lX11

// TODO MOVE GLOBALS SOMEWHERE
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
//int imageW = 960;
//int imageH = 540;

int imageW = 1920;
int imageH = 1080;

dim3 windowSize(imageW, imageH);
dim3 threads(8, 8); // 2 * https://xkcd.com/221/

curandState *d_rand_state;
Hittable **d_list;
Hittable **d_world;
Camera **d_camera;


extern "C" void init_cuda_scene(
        dim3 windowSize,
        Hittable **d_list,
        Hittable **d_world,
        Camera **d_camera
);

extern "C" void trace(
        dim3 blocks,
        dim3 threads,
        uint *d_output,
        uint imageW,
        uint imageH,
        curandState *d_rand_state,
        Hittable **d_world,
        Camera **d_camera
);

extern "C" void init_cuda_rng_state(
        dim3 windowSize,
        dim3 threads,
        curandState *d_rand_state
);

void cuda_malloc_scene()
{
    ssize_t num_pixels = windowSize.x * windowSize.y * sizeof(float3);
    checkCudaErrors(
            cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState))
    );
    checkCudaErrors(
            cudaMalloc(( void **)&d_list, 2 * sizeof(Hittable *) )
    );

    checkCudaErrors(
            cudaMalloc( (void **)&d_world, sizeof(Hittable *) )
    );

    checkCudaErrors(
            cudaMalloc((void **)&d_camera, sizeof(Camera *))
    );
}

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
    std::cout << "DefaultScreen set screenId to: " << screenId << "\n";

    // Check GLX version
    GLint majorGLX, minorGLX = 0;
    glXQueryVersion(display, &majorGLX, &minorGLX);
    if (majorGLX <= 1 && minorGLX < 4) {
        std::cout << "GLX 2.1 or greater is required.\n";
        XCloseDisplay(display);
        return 1;
    }
    else 
    {
        std::cout << "GLX version: " << majorGLX << "." << minorGLX << '\n';
    }

#define glxChooseVisual 0
#if glxChooseVisual
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
        std::cout << "Could not create correct visual window. " << visual << "\n";
        XCloseDisplay(display);
        return 1;
    }
#else
    // Instead of glXChooseVisual we can glXChooseFBConfig
    // To try to get a better combinations of GLXFBConfig (see glxinfo)
    static int visual_attribs[] =
    {
        GLX_RENDER_TYPE     , GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
        GLX_DOUBLEBUFFER    , True,
        GLX_RED_SIZE        , 8,
        GLX_GREEN_SIZE      , 8,
        GLX_BLUE_SIZE       , 8,
        GLX_ALPHA_SIZE      , 8,
        GLX_SAMPLE_BUFFERS  , 1,
        GLX_SAMPLES         , 4,
        None
    };

    int fbcount = 0;
    GLXFBConfig *fbc = glXChooseFBConfig( display, screenId, visual_attribs, &fbcount );
    if (fbc == 0) {
        std::cout << "Failed to retrieve framebuffer.\n";
        XCloseDisplay(display);
        return 1;
    }

    int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
    for (int i = 0; i < fbcount; ++i) {
        XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
        if ( vi != 0) {
            int samp_buf, samples, depth;
            glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS      , &samp_buf );
            glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES             , &samples  );
            glXGetFBConfigAttrib( display, fbc[i], GLX_DEPTH_SIZE          , &depth  );

            std::cout << "VisualInfo: " << samp_buf << " : " << samples << " : " << depth << "\n";

            if ( best_fbc < 0 || (samp_buf && samples > best_num_samp) ) {
                best_fbc = i;
                best_num_samp = samples;
            }
            if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
                worst_fbc = i;
            worst_num_samp = samples;
        }
        XFree( vi );
    }

    GLXFBConfig bestFbc = fbc[ best_fbc ];

    visual = glXGetVisualFromFBConfig( display, bestFbc );
    if (visual == 0) {
        std::cout << "Could not create correct visual window. " << visual << "\n";
        XCloseDisplay(display);
        return 1;
    }

    XFree( fbc );
#endif

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

    isGLVersionSupported(2,1);

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

    // call CUDA kernel, writing results to PBO
    trace(
        windowSize, threads, d_output, windowSize.x,
        windowSize.y, d_rand_state, d_world, d_camera
    );

    // UnMap cuda resource to be painted by openGl
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

    cuda_malloc_scene();

    init_cuda_scene(windowSize, d_list, d_world, d_camera);

    init_cuda_rng_state(windowSize, threads, d_rand_state);

    // Set GL Sample stuff
    glClearColor(0.4f, 0.6f, 0.5f, 1.0f);

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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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
