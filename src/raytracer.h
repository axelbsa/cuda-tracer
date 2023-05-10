#ifndef RAY_H
#define RAY_H

#include "cuda_helpers/helper_math.h"

// Most math functions will be in the above include.
inline __host__ __device__ float3 unit_vector(float3 v)
{
    return v / length(v);
}

inline __host__ __device__ float length_squared(float3 v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

class Ray
{
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

    bool front_face;

    inline __device__ void set_face_normal(const Ray& r, float3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class Camera
{
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
    __device__ Sphere(float3 cen, double r) : center(cen), radius(r) {};

    //__host__ __device__ bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;


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

        return true;
    }

    float3 center;
    float radius;
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
            //for (const auto& object : list) {
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
    //std::vector<make_shared<>(Hittable)> objects;
    //std::vector<shared_ptr<Hittable>> objects;
};

#endif

