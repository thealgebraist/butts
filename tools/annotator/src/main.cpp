#include <GLFW/glfw3.h>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <set>
#include <cmath>
#include "../json.hpp"
#include "ImageLoader.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Vertex {
    float x, y;
};

enum class ShapeType {
    POLYGON,
    DONUT
};

struct Annotation {
    ShapeType type = ShapeType::POLYGON;
    std::string label;
    std::vector<Vertex> vertices;
    Vertex center = {0,0};
    float outer_radius = 0;
    float inner_radius = 0;
};

enum class AppState {
    IDLE,
    PUT_VERTEX,
    PUT_DONUT_CENTER,
    PUT_DONUT_OUTER,
    PUT_DONUT_INNER,
    TEXT_INPUT
};

enum class AnnotationMode {
    POLYGON,
    DONUT
};

std::vector<std::string> get_all_images(const std::string& root_dir) {
    std::vector<std::string> images;
    std::set<std::string> exts = {".jpg", ".jpeg", ".png", ".heic"};
    for (const auto& entry : fs::recursive_directory_iterator(root_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (exts.contains(ext)) {
                images.push_back(entry.path().string());
            }
        }
    }
    std::sort(images.begin(), images.end(), [](const std::string& a, const std::string& b) {
        return fs::last_write_time(a) > fs::last_write_time(b);
    });
    return images;
}

std::set<std::string> load_annotation_defs(const std::string& path) {
    std::set<std::string> defs;
    if (fs::exists(path)) {
        try {
            std::ifstream f(path);
            json j;
            f >> j;
            for (const auto& item : j) {
                defs.insert(item.get<std::string>());
            }
        } catch (const std::exception& e) {
            std::cerr << "JSON parse error in " << path << ": " << e.what() << std::endl;
        }
    }
    return defs;
}

void save_annotation_defs(const std::string& path, const std::set<std::string>& defs) {
    json j = json::array();
    for (const auto& d : defs) {
        j.push_back(d);
    }
    std::ofstream f(path);
    f << j.dump(4);
}

void load_image_annotations(const std::string& img_path, std::vector<Annotation>& completed_annotations, float img_width, float img_height) {
    completed_annotations.clear();
    std::string annot_path = img_path + "_annot.json";
    if (fs::exists(annot_path)) {
        try {
            std::ifstream f(annot_path);
            json j; f >> j;
            for (const auto& item : j) {
                Annotation a;
                a.label = item["label"].get<std::string>();
                if (item.contains("type") && item["type"] == "donut") {
                    a.type = ShapeType::DONUT;
                    a.center = {item["center"]["x"].get<float>() / img_width, item["center"]["y"].get<float>() / img_height};
                    a.outer_radius = item["outer_radius"].get<float>() / img_width;
                    a.inner_radius = item["inner_radius"].get<float>() / img_width;
                } else {
                    a.type = ShapeType::POLYGON;
                    if (item.contains("polygon")) {
                        for (const auto& v : item["polygon"]) {
                            a.vertices.push_back({v["x"].get<float>() / img_width, v["y"].get<float>() / img_height});
                        }
                    }
                }
                completed_annotations.push_back(a);
            }
        } catch (...) {}
    }
}

int main() {
    std::cout << "Starting annotator..." << std::endl;
    if (!glfwInit()) {
        std::cerr << "glfwInit failed" << std::endl;
        return -1;
    }
    std::cout << "GLFW initialized." << std::endl;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Thrash Annotator", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    std::cout << "Window created." << std::endl;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.FontGlobalScale = 1.1f;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    std::cout << "Scanning for images..." << std::endl;
    std::string root_dir = "/Users/anders/projects/thrash";
    std::vector<std::string> images = get_all_images(root_dir);
    std::cout << "Found " << images.size() << " images." << std::endl;
    if (images.empty()) {
        std::cerr << "No unannotated images found." << std::endl;
        return 0;
    }

    int current_img_idx = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        if (!fs::exists(images[i] + "_annot.json")) {
            current_img_idx = i;
            break;
        }
    }
    LoadedImage current_texture = ImageLoader::loadTexture(images[current_img_idx]);

    AppState state = AppState::IDLE;
    AnnotationMode current_mode = AnnotationMode::POLYGON;
    Annotation current_annotation;
    std::vector<Annotation> completed_annotations;
    load_image_annotations(images[current_img_idx], completed_annotations, current_texture.width, current_texture.height);
    
    std::string defs_path = root_dir + "/annotation_def.json";
    std::set<std::string> annotation_defs = load_annotation_defs(defs_path);

    char input_buffer[256] = "";
    float zoom_factor = 1.0f;
    ImVec2 pan_offset(0.0f, 0.0f);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (state == AppState::IDLE) {
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
                current_img_idx = std::min((int)images.size() - 1, current_img_idx + 1);
                if (current_texture.textureID != 0) glDeleteTextures(1, &current_texture.textureID);
                current_texture = ImageLoader::loadTexture(images[current_img_idx]);
                load_image_annotations(images[current_img_idx], completed_annotations, current_texture.width, current_texture.height);
                zoom_factor = 1.0f; pan_offset = ImVec2(0,0);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
                current_img_idx = std::max(0, current_img_idx - 1);
                if (current_texture.textureID != 0) glDeleteTextures(1, &current_texture.textureID);
                current_texture = ImageLoader::loadTexture(images[current_img_idx]);
                load_image_annotations(images[current_img_idx], completed_annotations, current_texture.width, current_texture.height);
                zoom_factor = 1.0f; pan_offset = ImVec2(0,0);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_P)) {
                current_mode = AnnotationMode::POLYGON;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_D)) {
                current_mode = AnnotationMode::DONUT;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
                int max_num = 0;
                for (const auto& a : completed_annotations) {
                    try {
                        int num = std::stoi(a.label);
                        if (num > max_num) max_num = num;
                    } catch (...) {}
                }
                current_annotation = Annotation();
                if (current_mode == AnnotationMode::POLYGON) {
                    current_annotation.type = ShapeType::POLYGON;
                    state = AppState::PUT_VERTEX;
                } else {
                    current_annotation.type = ShapeType::DONUT;
                    state = AppState::PUT_DONUT_CENTER;
                }
                strncpy(input_buffer, std::to_string(max_num + 1).c_str(), sizeof(input_buffer));
            }
            for (int i = ImGuiKey_0; i <= ImGuiKey_9; ++i) {
                if (ImGui::IsKeyPressed((ImGuiKey)i)) {
                    current_annotation = Annotation();
                    if (current_mode == AnnotationMode::POLYGON) {
                        current_annotation.type = ShapeType::POLYGON;
                        state = AppState::PUT_VERTEX;
                    } else {
                        current_annotation.type = ShapeType::DONUT;
                        state = AppState::PUT_DONUT_CENTER;
                    }
                    std::string num_str = std::to_string(i - ImGuiKey_0);
                    strncpy(input_buffer, num_str.c_str(), sizeof(input_buffer));
                    break;
                }
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin("Mode", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
        ImGui::Text("Mode: %s (Press 'p' for Polygon, 'd' for Donut)", current_mode == AnnotationMode::POLYGON ? "POLYGON" : "DONUT");
        ImGui::End();

        if (!io.WantCaptureMouse) {
            if (io.MouseWheel != 0.0f) {
                float prev_zoom = zoom_factor;
                zoom_factor += io.MouseWheel * 0.1f * zoom_factor;
                if (zoom_factor < 0.1f) zoom_factor = 0.1f;
                if (zoom_factor > 20.0f) zoom_factor = 20.0f;
                pan_offset.x = io.MousePos.x - (io.MousePos.x - pan_offset.x) * (zoom_factor / prev_zoom);
                pan_offset.y = io.MousePos.y - (io.MousePos.y - pan_offset.y) * (zoom_factor / prev_zoom);
            }
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                pan_offset.x += io.MouseDelta.x;
                pan_offset.y += io.MouseDelta.y;
            }
        }

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Image", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
        ImVec2 p_max(pan_offset.x + io.DisplaySize.x * zoom_factor, pan_offset.y + io.DisplaySize.y * zoom_factor);
        ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)current_texture.textureID, pan_offset, p_max);

        auto screen_to_norm = [&](ImVec2 pos) {
            return Vertex{ (pos.x - pan_offset.x) / (io.DisplaySize.x * zoom_factor),
                           (pos.y - pan_offset.y) / (io.DisplaySize.y * zoom_factor) };
        };
        auto norm_to_screen = [&](const Vertex& v) {
            return ImVec2(pan_offset.x + v.x * io.DisplaySize.x * zoom_factor,
                          pan_offset.y + v.y * io.DisplaySize.y * zoom_factor);
        };

        for (auto it = completed_annotations.begin(); it != completed_annotations.end(); ) {
            auto& a = *it;
            bool clicked = false;
            if (a.type == ShapeType::POLYGON) {
                for (size_t i = 0; i < a.vertices.size(); ++i) {
                    ImVec2 p1 = norm_to_screen(a.vertices[i]);
                    ImVec2 p2 = norm_to_screen(a.vertices[(i + 1) % a.vertices.size()]);
                    ImGui::GetWindowDrawList()->AddLine(p1, p2, IM_COL32(0, 255, 0, 255), 2.0f);
                }
                if (!a.vertices.empty()) {
                    ImVec2 pt = norm_to_screen(a.vertices[0]);
                    std::string ulabel = a.label;
                    std::transform(ulabel.begin(), ulabel.end(), ulabel.begin(), ::toupper);
                    ImVec2 text_size = ImGui::CalcTextSize(ulabel.c_str());
                    ImVec2 rect_min = ImVec2(pt.x - 2, pt.y - 2);
                    ImVec2 rect_max = ImVec2(pt.x + text_size.x + 2, pt.y + text_size.y + 2);
                    ImGui::GetWindowDrawList()->AddRectFilled(rect_min, rect_max, IM_COL32(0, 0, 0, 255));
                    ImGui::GetWindowDrawList()->AddText(pt, IM_COL32(0, 255, 255, 255), ulabel.c_str());
                    if (ImGui::IsMouseHoveringRect(rect_min, rect_max) && ImGui::IsMouseDoubleClicked(0) && state == AppState::IDLE) clicked = true;
                }
            } else if (a.type == ShapeType::DONUT) {
                ImVec2 pt = norm_to_screen(a.center);
                float r_out = a.outer_radius * io.DisplaySize.x * zoom_factor;
                float r_in = a.inner_radius * io.DisplaySize.x * zoom_factor;
                ImGui::GetWindowDrawList()->AddCircle(pt, r_out, IM_COL32(0, 255, 0, 255), 64, 2.0f);
                ImGui::GetWindowDrawList()->AddCircle(pt, r_in, IM_COL32(0, 255, 0, 255), 64, 2.0f);
                std::string ulabel = a.label;
                std::transform(ulabel.begin(), ulabel.end(), ulabel.begin(), ::toupper);
                ImVec2 text_size = ImGui::CalcTextSize(ulabel.c_str());
                ImVec2 rect_min = ImVec2(pt.x - 2, pt.y - 2);
                ImVec2 rect_max = ImVec2(pt.x + text_size.x + 2, pt.y + text_size.y + 2);
                ImGui::GetWindowDrawList()->AddRectFilled(rect_min, rect_max, IM_COL32(0, 0, 0, 255));
                ImGui::GetWindowDrawList()->AddText(pt, IM_COL32(0, 255, 255, 255), ulabel.c_str());
                if (ImGui::IsMouseHoveringRect(rect_min, rect_max) && ImGui::IsMouseDoubleClicked(0) && state == AppState::IDLE) clicked = true;
            }

            if (clicked) {
                current_annotation = a;
                state = AppState::TEXT_INPUT;
                strncpy(input_buffer, a.label.c_str(), sizeof(input_buffer));
                it = completed_annotations.erase(it);
            } else {
                ++it;
            }
        }

        if (state == AppState::PUT_VERTEX) {
            for (size_t i = 1; i < current_annotation.vertices.size(); ++i) {
                ImVec2 p1 = norm_to_screen(current_annotation.vertices[i - 1]);
                ImVec2 p2 = norm_to_screen(current_annotation.vertices[i]);
                ImGui::GetWindowDrawList()->AddLine(p1, p2, IM_COL32(255, 0, 0, 255), 2.0f);
            }
            if (!current_annotation.vertices.empty()) {
                ImVec2 p1 = norm_to_screen(current_annotation.vertices.back());
                ImGui::GetWindowDrawList()->AddLine(p1, io.MousePos, IM_COL32(255, 255, 0, 255), 2.0f);
            }

            if (ImGui::IsMouseReleased(0) && !ImGui::GetIO().WantCaptureMouse) {
                ImVec2 drag_delta = ImGui::GetMouseDragDelta(0);
                if (std::abs(drag_delta.x) < 4.0f && std::abs(drag_delta.y) < 4.0f) {
                    if (!current_annotation.vertices.empty()) {
                        ImVec2 first_pt = norm_to_screen(current_annotation.vertices[0]);
                        float dx = io.MousePos.x - first_pt.x;
                        float dy = io.MousePos.y - first_pt.y;
                        if (std::sqrt(dx * dx + dy * dy) < 20.0f && current_annotation.vertices.size() > 2) {
                            state = AppState::TEXT_INPUT;
                        } else {
                            current_annotation.vertices.push_back(screen_to_norm(io.MousePos));
                        }
                    } else {
                        current_annotation.vertices.push_back(screen_to_norm(io.MousePos));
                    }
                }
            }
        } else if (state == AppState::PUT_DONUT_CENTER) {
            if (ImGui::IsMouseReleased(0) && !ImGui::GetIO().WantCaptureMouse) {
                ImVec2 drag_delta = ImGui::GetMouseDragDelta(0);
                if (std::abs(drag_delta.x) < 4.0f && std::abs(drag_delta.y) < 4.0f) {
                    current_annotation.center = screen_to_norm(io.MousePos);
                    state = AppState::PUT_DONUT_OUTER;
                }
            }
        } else if (state == AppState::PUT_DONUT_OUTER) {
            ImVec2 c_screen = norm_to_screen(current_annotation.center);
            float current_r = std::sqrt(std::pow(io.MousePos.x - c_screen.x, 2) + std::pow(io.MousePos.y - c_screen.y, 2));
            ImGui::GetWindowDrawList()->AddCircle(c_screen, current_r, IM_COL32(255, 0, 0, 255), 64, 2.0f);
            if (ImGui::IsMouseReleased(0) && !ImGui::GetIO().WantCaptureMouse) {
                ImVec2 drag_delta = ImGui::GetMouseDragDelta(0);
                if (std::abs(drag_delta.x) < 4.0f && std::abs(drag_delta.y) < 4.0f) {
                    current_annotation.outer_radius = current_r / (io.DisplaySize.x * zoom_factor);
                    state = AppState::PUT_DONUT_INNER;
                }
            }
        } else if (state == AppState::PUT_DONUT_INNER) {
            ImVec2 c_screen = norm_to_screen(current_annotation.center);
            float r_out_screen = current_annotation.outer_radius * io.DisplaySize.x * zoom_factor;
            ImGui::GetWindowDrawList()->AddCircle(c_screen, r_out_screen, IM_COL32(255, 0, 0, 255), 64, 2.0f);
            
            float current_r = std::sqrt(std::pow(io.MousePos.x - c_screen.x, 2) + std::pow(io.MousePos.y - c_screen.y, 2));
            ImGui::GetWindowDrawList()->AddCircle(c_screen, current_r, IM_COL32(255, 255, 0, 255), 64, 2.0f);
            
            if (ImGui::IsMouseReleased(0) && !ImGui::GetIO().WantCaptureMouse) {
                ImVec2 drag_delta = ImGui::GetMouseDragDelta(0);
                if (std::abs(drag_delta.x) < 4.0f && std::abs(drag_delta.y) < 4.0f) {
                    current_annotation.inner_radius = current_r / (io.DisplaySize.x * zoom_factor);
                    state = AppState::TEXT_INPUT;
                }
            }
        }
        ImGui::End();

        if (state == AppState::TEXT_INPUT) {
            ImGui::Begin("select annotation:");
            
            std::string current_match = "";
            for (const auto& def : annotation_defs) {
                if (std::string(input_buffer).empty() || def.find(input_buffer) != std::string::npos) {
                    current_match = def;
                    break;
                }
            }

            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            
            bool enter_pressed = ImGui::InputText("##label", input_buffer, IM_ARRAYSIZE(input_buffer), ImGuiInputTextFlags_EnterReturnsTrue);
            
            if (ImGui::IsKeyPressed(ImGuiKey_Tab)) {
                if (!current_match.empty()) {
                    strncpy(input_buffer, current_match.c_str(), sizeof(input_buffer));
                    enter_pressed = true;
                }
            }

            if (enter_pressed) {
                std::string label = input_buffer;
                if (!label.empty()) {
                    current_annotation.label = label;
                    completed_annotations.push_back(current_annotation);
                    annotation_defs.insert(label);
                    save_annotation_defs(defs_path, annotation_defs);

                    std::string annot_path = images[current_img_idx] + "_annot.json";
                    json j;
                    if (fs::exists(annot_path)) {
                        try {
                            std::ifstream f(annot_path); f >> j;
                        } catch (...) { j = json::array(); }
                    } else {
                        j = json::array();
                    }

                    if (current_annotation.type == ShapeType::POLYGON) {
                        float minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
                        json verts = json::array();
                        for (const auto& v : current_annotation.vertices) {
                            float px = v.x * current_texture.width;
                            float py = v.y * current_texture.height;
                            verts.push_back({{"x", px}, {"y", py}});
                            minX = std::min(minX, px); minY = std::min(minY, py);
                            maxX = std::max(maxX, px); maxY = std::max(maxY, py);
                        }
                        j.push_back({
                            {"type", "polygon"},
                            {"label", label},
                            {"polygon", verts},
                            {"bbox", {{"x", minX}, {"y", minY}, {"width", maxX - minX}, {"height", maxY - minY}}}
                        });
                    } else {
                        float cx = current_annotation.center.x * current_texture.width;
                        float cy = current_annotation.center.y * current_texture.height;
                        float ro = current_annotation.outer_radius * current_texture.width;
                        float ri = current_annotation.inner_radius * current_texture.width;
                        
                        j.push_back({
                            {"type", "donut"},
                            {"label", label},
                            {"center", {{"x", cx}, {"y", cy}}},
                            {"outer_radius", ro},
                            {"inner_radius", ri},
                            {"bbox", {{"x", cx - ro}, {"y", cy - ro}, {"width", ro * 2}, {"height", ro * 2}}}
                        });
                    }

                    std::ofstream f(annot_path);
                    f << j.dump(4);

                    state = AppState::IDLE;
                }
            }
            
            for (const auto& def : annotation_defs) {
                if (std::string(input_buffer).empty() || def.find(input_buffer) != std::string::npos) {
                    if (ImGui::Selectable(def.c_str())) {
                        strncpy(input_buffer, def.c_str(), sizeof(input_buffer));
                    }
                }
            }
            ImGui::End();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
