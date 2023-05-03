#include <float.h>
#include "cuda_helpers/helper_cuda.h"
#include "cuda_helpers/helper_math.h"

#include "raytracer.h"

typedef unsigned char uchar;

//const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

__device__ __inline__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ __inline__ uchar4 to_uchar4(float4 vec) {
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}


__device__ float3 color(const Ray& r, Hittable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec))
    {
        return make_float3(
            rec.normal.x + 1.0f, 
            rec.normal.y + 1.0f,
            rec.normal.z + 1.0f
        ) * 0.5f;
    }
    else
    {
        float3 unit_direction = unit_vector(r.direction());
        float t = (unit_direction.y + 1.0f) * 0.5f;
        return (1.0f-t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0);
    }
}

__global__ void d_render(
        uchar4 *output, uint imageW, uint imageH, float3 lower_left_corner,
        float3 horizontal, float3 vertical, float3 origin, Hittable **world)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)imageW;
    float v = y / (float)imageH;

    if ((x < imageW) && (y < imageH)) {
        uint i = y * imageW + x;  // Array transformation to 1D?

        Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
        output[i] = to_uchar4( make_float4( color(r, world), 1.0 ) * 255.99 );
    }
}

__global__ void create_world(Hittable **d_list, Hittable **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new Sphere(make_float3(0.0f, 0.0f, -1.0f), 0.5f);
        *(d_list + 1) = new Sphere(make_float3(0.0f, -100.5f,-1.0f), 100.0f);
        *d_world = new Hittable_list(d_list, 2);
    }
}

extern "C" void trace(
        dim3 blocks,
        dim3 threads,
        uchar4 *d_output,
        uint imageW,
        uint imageH,
        float3 lower_left_corner, 
        float3 horizontal, 
        float3 vertical, 
        float3 origin)
{

    Hittable **d_list;
    Hittable **d_world;

    checkCudaErrors(
        cudaMalloc(( void **)&d_list, 2 * sizeof(Hittable *) )
    );

    checkCudaErrors(
        cudaMalloc( (void **)&d_world, sizeof(Hittable *) )
    );

    create_world <<<1, 1>>> (d_list,d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //fprintf(stderr, "Calling draw\n");
    d_render<<<blocks, threads>>>(
            d_output, imageW, imageH,
            lower_left_corner, horizontal, vertical, origin, d_world);
}

