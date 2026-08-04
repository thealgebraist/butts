#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Ensure temp build dir
TEMP_DIR=$(mktemp -d)
echo "Building in $TEMP_DIR"

IMGUI_SRC="imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp"

clang++ -std=c++23 -O2 \
    -I/opt/homebrew/include \
    -I./imgui -I./imgui/backends \
    src/main.cpp $IMGUI_SRC \
    -L/opt/homebrew/lib \
    -lglfw \
    -framework OpenGL \
    -framework Cocoa \
    -framework CoreGraphics \
    -framework ImageIO \
    -framework CoreFoundation \
    -o ./thrash_annotator

chmod +x ./thrash_annotator
codesign --force --sign - ./thrash_annotator
echo "Build complete: ./thrash_annotator"
