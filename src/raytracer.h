#ifndef RAY_H
#define RAY_H

#include <curand_kernel.h>
#include "cuda_helpers/helper_cuda.h"
#include "cuda_helpers/helper_math.h"

class Material;  // Define some class material
typedef unsigned char uchar;
const double pi = 3.1415926535897932385;


//__device__ __inline__ float3 reflect()

__device__ __inline__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ __inline__ uchar4 to_uchar4(float4 vec) {
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

// Most math functions will be in the above include.
__host__ __device__ inline float3 unit_vector(float3 v)
{
    return v / length(v);
}

__host__ __device__ inline float length_squared(float3 v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

#define RANDVEC3 make_float3(     \
    curand_uniform(local_rand_state), \
    curand_uniform(local_rand_state), \
    curand_uniform(local_rand_state)  \
)
__device__ static inline float3 random_in_unit_sphere(curandState *local_rand_state) {
    float3 p;
    do {
        p = 2.0f*RANDVEC3 - make_float3(1,1,1);
    } while (length_squared(p) >= 1.0f);
    return p;
}

class Ray {
public:
    __device__ Ray() {}
    __device__ Ray(const float3& a, const float3& b) { A = a; B = b; }
    __device__ float3 origin() const       { return A; }
    __device__ float3 direction() const    { return B; }
    __device__ float3 point_at_parameter(float t) const { return A + t*B; }

    float3 A;
    float3 B;
};


struct hit_record {
    float3 p;
    float3 normal;
    float t;
    Material *mat_prt;

    bool front_face;

    inline __device__ void set_face_normal(const Ray& r, float3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class Camera {
public:
    __host__ __device__ Camera(dim3 windowSize)
    {
        calculate_camera(windowSize.x, windowSize.y);
    }

    __host__ __device__ void calculate_camera(uint imageW, uint imageH)
    {
        image_width = imageW;
        image_height = static_cast<int>(image_width / aspect_ratio);

        // Default camera
        origin = make_float3(0.0f, 0.0f, 0.0f);
        horizontal = make_float3(viewport_width, 0.0f, 0.0f);
        vertical = make_float3(0.0f, viewport_height, 0.0f);
        //lower_left_corner = make_float3(-2.0f, -1.0f, -1.0f);
        lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - make_float3(0.0f, 0.0f, focal_length);
    }

    __device__ Ray get_ray(float u, float v)
    {
        //math!
        return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }

    const float aspect_ratio = 16.0f / 9.0f;
    int image_width;
    int image_height;

    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;

    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
};


class Hittable {
public:
    __device__ virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};


class Sphere : public Hittable {
public:
    __device__ Sphere() {}
    __device__ Sphere(float3 cen, double r, Material *m) : center(cen), radius(r), mat_ptr(m) {};

    __device__ bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const
    {
        float3 oc = r.origin() - center;
        auto a = length_squared(r.direction());
        auto half_b = dot(oc, r.direction());
        auto c = float(length_squared(oc) - radius*radius);

        auto discriminant = float(half_b*half_b - a*c);

        if (discriminant < 0)
        {
            return false;
        }

        auto sqrtd = sqrtf(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.point_at_parameter(rec.t);
        rec.normal = (rec.p - center) / radius;
        float3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_prt = mat_ptr;

        return true;
    }

    float3 center;
    float radius;
    Material *mat_ptr;
};


class Hittable_list : public Hittable {
public:
    __device__ Hittable_list() {}
    __device__ Hittable_list(Hittable **object, int count)
    {
        list_size = count;
        list = object;
    }

    __device__ bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const 
    {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = t_max;

        for (int i = 0; i < list_size; ++i) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

private:
    int list_size;
    Hittable **list;
};

class Material {
public:
    __device__ virtual bool scatter(
            const Ray& r_in,
            const hit_record& rec,
            float3& attenuation,
            Ray& scattered,
            curandState *local_rand_state
    ) const = 0;
};

class Lambertian : public Material {
public:
    __device__ Lambertian ( const float3& a ) : albedo(a) {}

    __device__ bool scatter(const Ray& r_in, const hit_record& rec,
                            float3& attenuation, Ray& scattered, curandState *local_rand_state) const
    {
        float3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state); // A new vector in a random direction from where it hit
        scattered = Ray(rec.p, target - rec.p); // A new ray with vector pointing at target - rec.p (eye)
        attenuation = albedo; // Attenuation == how much light do we have left 0.0 - 1.0
                              // In this case it's just the color of the object
        return true;
    }

    float3 albedo;
};


class Metal : public Material {
public:
    __device__ Metal (const float3& a, float f) {
        albedo = a;
        (f < 1.0f) ? fuzzy = f : fuzzy = 1;

    }
    __device__ bool scatter(const Ray& r_in, const hit_record& rec,
                            float3& attenuation, Ray& scattered, curandState *local_rand_state) const
    {
        // Metals are like perfect mirrors if they are smooth enough
        // So we take the ray going into the scene, get the normal of the object we are hitting
        // And call reflect to give us the reflected new vector.
        float3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzzy * random_in_unit_sphere(local_rand_state)); // Create a new ray with the new vector
        attenuation = albedo; // Same as above, light radiation is just the color of the object

        // Check if the angle between the incomming ray's normal (rec.normal) and the new reflected ray (scattered)
        // Is greater than 0.0f. Try to imagine where the ray would be 0
        // https://www.haroldserrano.com/blog/vectors-in-computer-graphics
        return (dot(scattered.direction(), rec.normal)) > 0;
    }

    float3 albedo;
    float fuzzy;
};

#endif

