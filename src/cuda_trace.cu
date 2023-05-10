#include <float.h>
#include <stdio.h>

#include <curand_kernel.h>

#include "cuda_helpers/helper_cuda.h"
#include "cuda_helpers/helper_math.h"

#include "raytracer.h"

typedef unsigned char uchar;

//const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

#define RANDVEC3 make_float3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
__device__ float3 random_in_unit_sphere(curandState *local_rand_state) {
    float3 p;
    do {
        p = 2.0f*RANDVEC3 - make_float3(1,1,1);
    } while (length_squared(p) >= 1.0f);
    return p;
}

__device__ __inline__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ __inline__ uchar4 to_uchar4(float4 vec) {
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}


__device__ float3 color(const Ray& r, Hittable **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < 50; ++i) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            float3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = Ray(rec.p, target - rec.p);
        } else {
            float3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y + 1.0f);
            float3 c = (1.0f - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void d_render(
        uchar4 *output, uint imageW, uint imageH,
        Hittable **world, Camera **d_camera, curandState *d_rand_state)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float3 col = make_float3(0.0f, 0.0f, 0.0f);

    if((x >= imageW) || (y >= imageH))
        return;

    uint i = y * imageW + x;  // Array transformation to 1D?
    curandState local_rand_state = d_rand_state[i];
#define ns 50
    for(int s=0; s < ns; s++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(imageW);
        float v = float(y + curand_uniform(&local_rand_state)) / float(imageH);
        Ray r = (*d_camera)->get_ray(u,v);
        col += color(r, world, &local_rand_state);
    }
    d_rand_state[i] = local_rand_state;
    col /= float(ns);
    //printf("Color contrib: %f %f %f\n",col.x, col.y, col.z);
    output[i] = to_uchar4( make_float4( col, 1.0 ) * 255.99 );
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {

   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;

   if((i >= max_x) || (j >= max_y))
       return;

   int pixel_index = j * max_x + i;

   //Each thread gets same seed, a different sequence number, no offset
   curand_init(j * 1984 + i, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void create_world(dim3 windowSize, Hittable **d_list, Hittable **d_world, Camera **d_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new Sphere(make_float3(0.0f, 0.0f, -1.0f), 0.5f);
        *(d_list + 1) = new Sphere(make_float3(0.0f, -100.5f,-1.0f), 100.0f);
        *d_world = new Hittable_list(d_list, 2);
        *d_camera = new Camera(windowSize);
    }
}

extern "C" void init_cuda_scene(
        dim3 windowSize,
        Hittable **d_list,
        Hittable **d_world,
        Camera **d_camera)
{

    create_world <<<1, 1>>> (windowSize, d_list, d_world, d_camera);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Remember to free somewhere
    //checkCudaErrors(cudaFree(d_camera));
    //checkCudaErrors(cudaFree(d_world));
    //checkCudaErrors(cudaFree(d_list));
}

extern "C" void init_cuda_rng_state(
        dim3 windowSize,
        dim3 threads,
        curandState *d_rand_state)
{

    render_init<<<windowSize, threads>>>(windowSize.x, windowSize.y, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void trace(
        dim3 blocks,
        dim3 threads,
        uchar4 *d_output,
        uint imageW,
        uint imageH,
        curandState *d_rand_state,
        Hittable **d_world,
        Camera **d_camera)
{
    //fprintf(stderr, "Calling draw\n");
    d_render<<<blocks, threads>>>(
            d_output, imageW, imageH, d_world, d_camera, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
