/******************************************************************************
 * Copyright 1998-2018 NVIDIA Corp. All Rights Reserved.
 *****************************************************************************/

#pragma once
#include "fileformats/tiny_obj_loader.h"
#include "nvmath/nvmath.h"
#include "../material/material_obj.hpp"
#include <array>
#include <iostream>
#include <unordered_map>
#include <vector>


// OBJ representation of a vertex
// NOTE: BLAS builder depends on pos being the first member
struct VertexObj
{
  nvmath::vec3f pos;
  nvmath::vec3f nrm;
  nvmath::vec3f color;
  nvmath::vec2f texCoord;
};


struct shapeObj
{
  uint32_t offset;
  uint32_t nbIndex;
  uint32_t matIndex;
};

class ObjLoader
{
public:
  void loadModel(const std::string& filename);

  std::vector<VertexObj>   m_vertices;
  std::vector<uint32_t>    m_indices;
  std::vector<MaterialObj> m_materials;
  std::vector<std::string> m_textures;
  std::vector<int32_t>     m_matIndx;
};
