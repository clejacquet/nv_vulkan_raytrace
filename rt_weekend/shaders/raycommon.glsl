struct hitPayload
{
  vec3 hitValue;
  bool hasHit;
  int depth;
};

struct Sphere
{
  highp vec3  center;
  highp float radius;
};

struct Aabb
{
  vec3 minimum;
  vec3 maximum;
};

struct HitRecord {
  highp vec3 p;
  highp vec3 normal;
  highp float t;
};

#define KIND_SPHERE 0
#define KIND_CUBE 1