#ifndef MATERIAL_OBJ_H
#define MATERIAL_OBJ_H

#include <nvmath/nvmath.h>

// Structure holding the material
struct MaterialObj
{
    nvmath::vec3f ambient       = nvmath::vec3f(0.1f, 0.1f, 0.1f);
    nvmath::vec3f diffuse       = nvmath::vec3f(0.7f, 0.7f, 0.7f);
    nvmath::vec3f specular      = nvmath::vec3f(1.0f, 1.0f, 1.0f);
    nvmath::vec3f transmittance = nvmath::vec3f(0.0f, 0.0f, 0.0f);
    nvmath::vec3f emission      = nvmath::vec3f(0.0f, 0.0f, 0.10);
    float         shininess     = 0.f;
    float         ior           = 1.0f;  // index of refraction
    float         dissolve      = 1.f;   // 1 == opaque; 0 == fully transparent
        // illumination model (see http://www.fileformat.info/format/material/)
    int illum     = 0;
    int textureID = -1;
    int textureIDspec = -1;
};


#endif