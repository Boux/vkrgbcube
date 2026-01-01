project('vkrgbcube', 'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++17', 'warning_level=2']
)

# Dependencies
vulkan_dep = dependency('vulkan')
glfw_dep = dependency('glfw3')
glm_dep = dependency('glm')

# Shader compilation
glslang = find_program('glslangValidator')

shader_files = files(
  'src/shaders/cube.vert.glsl',
  'src/shaders/cube.frag.glsl',
  'src/shaders/ui.vert.glsl',
  'src/shaders/ui.frag.glsl',
)

shader_targets = []
foreach shader : shader_files
  shader_name = shader.full_path().split('/')[-1]
  out_name = shader_name.replace('.glsl', '.spv.h')
  shader_targets += custom_target(
    out_name,
    input : shader,
    output : out_name,
    command : [glslang, '-V', '--vn', shader_name.underscorify(), '-o', '@OUTPUT@', '@INPUT@']
  )
endforeach

# Source files
src_files = files(
  'src/main.cpp',
)

executable('vkrgbcube',
  src_files,
  shader_targets,
  dependencies : [vulkan_dep, glfw_dep, glm_dep],
  include_directories : [include_directories('src'), include_directories('.')],
)
