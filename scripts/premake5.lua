-- Copyright (c) 2023 Thakee Nathees

-- ---------------------------------------------------------------------------
-- Variables
-- ---------------------------------------------------------------------------

local root_dir_rel       = ".."
local root_dir_abs       = path.getdirectory(os.getcwd())
local project_name       = path.getbasename(root_dir_abs)
local dir_build          = root_dir_rel .. "/build"

local dir_raylib         = root_dir_rel .. "/thirdparty/raylib-4.5.0"
local dir_thirdparty     = root_dir_rel .. "/thirdparty"

local dir_bin_raylib     = dir_build .. "/bin/%{cfg.buildcfg}/raylib"
local dir_bin_thirdparty = dir_build .. "/bin/%{cfg.buildcfg}/thirdparty"
local dir_bin_project    = dir_build .. "/bin/%{cfg.buildcfg}/" .. project_name

-- ---------------------------------------------------------------------------
-- Workspace
-- ---------------------------------------------------------------------------

workspace (project_name)
  location (dir_build)
  configurations { "Debug", "Release"}
  architecture "x64"
  cppdialect "C++17"
  startproject (project_name)


-- ---------------------------------------------------------------------------
-- Raylib
-- ---------------------------------------------------------------------------

-- Copied from and modified:
--   https://github.com/raylib-extras/game-premake/blob/main/raylib_premake5.lua

function platform_defines()
  defines{"PLATFORM_DESKTOP"}

  filter {"options:graphics=opengl43"}
    defines{"GRAPHICS_API_OPENGL_43"}

  filter {"options:graphics=opengl33"}
    defines{"GRAPHICS_API_OPENGL_33"}

  filter {"options:graphics=opengl21"}
    defines{"GRAPHICS_API_OPENGL_21"}

  filter {"options:graphics=opengl11"}
    defines{"GRAPHICS_API_OPENGL_11"}

  filter {"system:macosx"}
    disablewarnings {"deprecated-declarations"}

  filter {"system:linux"}
    defines {"_GNU_SOURCE"}
-- This is necessary, otherwise compilation will fail since
-- there is no CLOCK_MONOTOMIC. raylib claims to have a workaround
-- to compile under c99 without -D_GNU_SOURCE, but it didn't seem
-- to work. raylib's Makefile also adds this flag, probably why it went
-- unnoticed for so long.
-- It compiles under c11 without -D_GNU_SOURCE, because c11 requires
-- to have CLOCK_MONOTOMIC
-- See: https://github.com/raysan5/raylib/issues/2729

  filter{}
end


function link_raylib()

  links {"raylib"}

  includedirs { dir_raylib .. "/src" }
  includedirs { dir_raylib .. "/src/external" }
  includedirs { dir_raylib .. "/src/external/glfw/include" }
  platform_defines()

  filter "action:vs*"
    defines{"_WINSOCK_DEPRECATED_NO_WARNINGS", "_CRT_SECURE_NO_WARNINGS"}
    dependson {"raylib"}
    links {"raylib.lib"}
    libdirs { dir_bin_raylib }
    characterset ("MBCS")

  filter "system:windows"
    defines{"_WIN32"}
    links {"winmm", "kernel32", "opengl32", "gdi32"}
    libdirs { dir_bin_raylib }

  filter "system:linux"
    links {"pthread", "GL", "m", "dl", "rt", "X11"}

  filter "system:macosx"
    links {
      "OpenGL.framework",
      "Cocoa.framework",
      "IOKit.framework",
      "CoreFoundation.framework",
      "CoreAudio.framework",
      "CoreVideo.framework"
    }

  filter{}
end


project "raylib"
  kind "StaticLib"
  platform_defines()
  location (dir_build)
  language "C"
  targetdir (dir_bin_raylib)

  filter "action:vs*"
    defines{"_WINSOCK_DEPRECATED_NO_WARNINGS", "_CRT_SECURE_NO_WARNINGS"}
    characterset ("MBCS")

  filter{}

  print ("Using raylib dir " .. dir_raylib);

  includedirs {
    dir_raylib .. "/src",
    dir_raylib .. "/src/external/glfw/include"
  }

  files {
    dir_raylib .. "/src/*.h",
    dir_raylib .. "/src/*.c"
  }

  filter {
    "system:macosx",
    "files:" .. dir_raylib .. "/src/rglfw.c"
  }

  compileas "Objective-C"

  filter{}


-- ---------------------------------------------------------------------------
-- Main Project
-- ---------------------------------------------------------------------------

project (project_name)
  kind "ConsoleApp"
  language "C++"
  location (dir_build)
  targetdir (dir_bin_project)

  filter "configurations:Debug"
    defines { "DEBUG" }
    symbols "On"

  filter "configurations:Release"
    defines { "NDEBUG" }
    optimize "On"

  filter "action:vs*"
    defines{"_WINSOCK_DEPRECATED_NO_WARNINGS", "_CRT_SECURE_NO_WARNINGS"}
    characterset ("MBCS")

  filter {}

  files {
    root_dir_rel .. "/src/**.c",
    root_dir_rel .. "/src/**.h",
    root_dir_rel .. "/src/**.cpp",
    root_dir_rel .. "/src/**.hpp",
  }

  includedirs {
    root_dir_rel .. "/src/",
  }

  -- Enable if needed.
  -- buildoptions { "-Wall" }

  links { "winmm.lib" } -- TODO: This should be a conditional link for windows.

  link_raylib()


-- Copy files files after build.
postbuildcommands {
  -- "cp " .. source_dir .. " " .. target_dir
}
