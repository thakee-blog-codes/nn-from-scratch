
#include <raylib.h>
#include <vector>
#include <string>
#include <algorithm>

#include "raygui.h"
#include "matrix.hpp"
#include "nn.hpp"
#include "utils.hpp"


class UI {
public:

  enum State {
    IDLE,
    TRAINING,
    DRAWING,
    TESTING,
  };

  UI(NN* nn, DsMinist* dset_train, DsMinist* dset_test);
  
  void handle_inputs();
  void update();
  void cleanup();

  void message(const std::string& msg);
  void push_error(float value);
  void set_texture(Texture* texture);


  State get_state() const;
  void set_state(State state);

  void render();

  void draw_nn_graph();
  void draw_error_graph();
  void draw_progress();
  void draw_inputs();
  void draw_neuron_info();
  void draw_message();

  // Draw a texture under the error graph, Note that the texture
  // width will be the same as graph width.
  void draw_texture();
  void draw_drawing_canvas();

private:
  Color _interpolated_color(Color from, Color to, float weight);
  void _update_area();
  bool _nn_graph_has_mouse();

  void _forward_canvas();
  void _clear_canvas();

  void _check_box(Rectangle area, const char* label, bool* active);

  State state = State::IDLE;
  bool training = true; // Either we're training or testing.

  NN* nn = nullptr;
  DsMinist* dset_train = nullptr;
  DsMinist* dset_test = nullptr;

  Texture* texture = nullptr;

  // To draw an image to test.
  RenderTexture2D canvas;
  Texture drawn;
  float brush_size = 20.f;
  Vector2 canvas_size = {
    28*20,
    28*20,
  };

  // Notification message.
  const float msg_max_time = 2;
  float msg_time = 0;
  std::string msg;

  float err_min = 99999.f;
  float err_max = 0.f;
  int err_reset_count = 600;
  std::vector<float> errors;

  // x = layer index, y = neuron index.
  Vector2 selected_neuron = { -1, -1 };

  Camera2D cam_nn;

  Rectangle area_error_graph = { 0 };
  Rectangle area_texture = { 0 };
  Rectangle area_input = { 0 };
  Rectangle area_progress = { 0 };
  Rectangle area_neuron_info = { 0 };
  Rectangle area_canvas = { 0 };
  Rectangle area_nn = { 0 };

  const int font_size        = 25;
  const float btn_width      = 150.f;
  const float btn_height     = 30.f;
  const float padding        = 15.f;
  const float neuron_radius  = 40.f;
  const float neuron_gap     = 100.f;
  const float layer_gap      = 400.f;
  const float zoom_increment = 0.5f;

  Color color_nn_area     = GetColor(0x252525ff);
  Color color_selected_neuron = PURPLE;
  Color color_neuron_min  = WHITE;
  Color color_neuron_max  = DARKBLUE;
  Color color_conn_min    = color_nn_area;
  Color color_conn_max    = GRAY;
  Color color_pannel      = { 100, 100, 100, 0xff };
  Color color_progress    = { 80, 207, 112, 0xff };
};


#ifdef SINGLE_SOURCE_IMPL

UI::UI(NN* nn, DsMinist* dset_train, DsMinist* dset_test)
  : nn(nn), dset_train(dset_train), dset_test(dset_test) {

  cam_nn = { 0 };
  update();
  cam_nn.offset = { area_nn.x, area_nn.y };
  cam_nn.target = cam_nn.offset;
  cam_nn.zoom = 1;

  // Raylib initialize.
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(800, 450, "nn");
  SetTargetFPS(120);
  MaximizeWindow();
  SetExitKey(0);

  GuiSetStyle(DEFAULT, TEXT_SIZE, 20);

  canvas = LoadRenderTexture(canvas_size.x, canvas_size.y);
  BeginTextureMode(canvas);
  ClearBackground(BLACK);
  EndTextureMode();
}


bool UI::_nn_graph_has_mouse() {
  Vector2 mouse_pos = GetMousePosition();
  return
    !CheckCollisionPointRec(mouse_pos, area_error_graph) &&
    !CheckCollisionPointRec(mouse_pos, area_texture) &&
    !CheckCollisionPointRec(mouse_pos, area_input) &&
    !CheckCollisionPointRec(mouse_pos, area_progress) &&
    (
      (selected_neuron.x < 0 || selected_neuron.y < 0) ||
      !CheckCollisionPointRec(mouse_pos, area_neuron_info)
    ) &&
    CheckCollisionPointRec(mouse_pos, area_nn);
}


void UI::_forward_canvas() {
  Image img = LoadImageFromTexture(canvas.texture);
  ImageFlipVertical(&img); // Render texture canvas are up side down.
  ImageColorGrayscale(&img);
  Matrix input = DsMinist::image_to_input(&img); // This will resize the image.

  if (IsTextureReady(drawn)) UnloadTexture(drawn);
  drawn = LoadTextureFromImage(img);

  texture = &drawn;
  UnloadImage(img);

  nn->forward(input);
}


void UI::_clear_canvas() {
  BeginTextureMode(canvas);
  ClearBackground(BLACK);
  EndTextureMode();
}


void UI::handle_inputs() {
  Vector2 mouse_pos = GetMousePosition();

  // Since this is an immediate mode ui, we'll mostly handle inputs when
  // we draw the components. Here we handle global events or key press.
  if ((state == IDLE || state == TRAINING || state == TESTING) && _nn_graph_has_mouse()) {

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
      Vector2 delta = GetMouseDelta();
      delta = Vector2Scale(delta, -1.0f / cam_nn.zoom);
      cam_nn.target = Vector2Add(cam_nn.target, delta);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
      Vector2 mouse_world_pos = GetScreenToWorld2D(GetMousePosition(), cam_nn);
      cam_nn.offset = GetMousePosition();
      cam_nn.target = mouse_world_pos;
      cam_nn.zoom += (wheel * zoom_increment);
      if (cam_nn.zoom < zoom_increment)
        cam_nn.zoom = zoom_increment;
    }
  }
}


Color UI::_interpolated_color(Color from, Color to, float weight) {
  weight = 1.f / (1.f + expf(-weight));
  Color r;
  r.r = (unsigned char) Lerp(from.r, to.r, weight);
  r.g = (unsigned char) Lerp(from.g, to.g, weight);
  r.b = (unsigned char) Lerp(from.b, to.b, weight);
  r.a = 0xff;
  return r;
}


void UI::_update_area() {

  float screen_width = GetScreenWidth();
  float screen_height = GetScreenHeight();

  area_nn = {
    0.f,
    0.f,
    (float)screen_width,
    (float)screen_height
  };

  float progress_bar_height = 30.f;
  area_progress = {
    padding,
    screen_height - padding - progress_bar_height,
    screen_width - 2 * padding,
    progress_bar_height,
  };

  area_error_graph = {
    padding,
    padding,
    300,
    180
  };

  area_texture = {
    area_error_graph.x,
    area_error_graph.y + area_error_graph.height + padding,
    area_error_graph.width,
    area_texture.height,
  };

  const float area_input_y = area_texture.y + area_texture.height + padding;
  area_input = {
    area_error_graph.x,
    area_input_y,
    area_error_graph.width,
    screen_height - area_input_y - padding - area_progress.height - padding,
  };

  const Vector2 info_size = {
    400,
    screen_height - 2 * padding - area_progress.height - padding
  };
  area_neuron_info = {
    screen_width - padding - info_size.x,
    padding,
    info_size.x,
    info_size.y
  };

  const Vector2 canvas_area_size = {
    canvas_size.x + 2 * padding,
    canvas_size.y + 2 * padding + btn_height + padding,
  };
  area_canvas = {
    (screen_width - canvas_area_size.x) / 2.f,
    (screen_height - canvas_area_size.y) / 2.f,
    canvas_area_size.x,
    canvas_area_size.y,
  };
}


void UI::update() {
  _update_area();
  if (!msg.empty() && GetTime() - msg_time > msg_max_time) {
    msg = "";
  }
}


void UI::cleanup() {
  if (IsTextureReady(drawn)) UnloadTexture(drawn);
}


void UI::push_error(float value) {
  errors.push_back(value);
  if (errors.size() > err_reset_count) errors.erase(errors.begin());
  if (errors.size() > 0) {
    err_min = *std::min_element(errors.begin(), errors.end());
    err_max = *std::max_element(errors.begin(), errors.end());
  }
}


void UI::set_texture(Texture* texture) {
  this->texture = texture;
}


void UI::draw_error_graph() {
  Rectangle area = area_error_graph;

  DrawRectangleRec(area, color_pannel);
  float error = 0.f;
  if (errors.size() > 0) {
    for (size_t i = 0; i < errors.size() - 1; i++) {
      float x1 = area.x + area.width / errors.size() * i;
      float y1 = area.y + (1 - (errors[i] - err_min) / (err_max - err_min)) * area.height;
      float x2 = area.x + area.width / errors.size() * (i + 1);
      float y2 = area.y + (1 - (errors[i + 1] - err_min) / (err_max - err_min)) * area.height;
      DrawLineEx(
        { x1, y1 },
        {x2, y2},
        1, //area.height * .01,
        BLACK);
    }
    error = errors.at(errors.size() - 1);
  }

  std::string txt = std::string("error = ") + std::to_string(error);
  DrawText(
    txt.c_str(),
    area.x + padding,
    area.y + padding,
    font_size,
    BLUE);
}


void UI::_check_box(Rectangle area, const char* label, bool* active) {

  Vector2 mouse_pos = GetMousePosition();
  Color color_border = GRAY;
  if (CheckCollisionPointRec(mouse_pos, area)) {
    color_border = BLUE;
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
      *active = !(*active);
    }
  }

  DrawRectangleLinesEx(area, 2, color_border);
  DrawText(label, area.x + area.width + padding, area.y, font_size, { 37, 37, 37, 0xff });

  const int pad = 2;
  if (*active) {
    area.x += pad; area.y += pad;
    area.width -= 2 * pad; area.height -= 2 * pad;
    DrawRectangleRec(area, BLUE);
  }
}


void UI::draw_inputs() {
  DrawRectangleRec(area_input, color_pannel);

  Rectangle comp_area = {
      area_input.x + padding,
      area_input.y + padding,
      area_input.width - 2 * padding,
      btn_height,
  };

  {
    Rectangle area = comp_area;
    area.width = area.height;
    const char* label = (training) ? "training" : "testing";
    _check_box(area, label, &training);
  }

  { // Train, Test / Pause btn.
    comp_area.y += comp_area.height + padding;
    const char* btn_label = (
      (state == TRAINING || state == TESTING) ? "pause" : (
        (training) ? "train" : "test"
      )
    );

    if (
      GuiButton(comp_area, btn_label) ||
      (training && IsKeyReleased(KEY_SPACE))
    ) {
      switch (state) {
        case TESTING:
        case TRAINING:
        state = IDLE;
        break;

        case IDLE:
        state = (training) ? TRAINING : TESTING;
        break;
      }
    }
  }

  // FIXME:
  static bool iter = false;
  if (iter) {
    iter = false;
    state = IDLE;
  }

  { // Iter btn.
    comp_area.y += comp_area.height + padding;
    if ((GuiButton(comp_area, "iter") || IsKeyReleased(KEY_N))) {
      switch (state) {
        case IDLE:
        iter = true;
        state = (training) ? TRAINING : TESTING;
        break;

        case TRAINING:
        case TESTING:
        state = IDLE;
        break;

        default:
        break;

      }
    }
  }

  { // Save btn.
    comp_area.y += comp_area.height + padding;
    if (GuiButton(comp_area, "save model") && state != DRAWING) {
      nn->save("nn");
      message("Model saved to \"./nn\"!");
    }
  }

  { // Load btn.
    comp_area.y += comp_area.height + padding;
    if (GuiButton(comp_area, "load model") && state != DRAWING) {
      nn->load("nn");
      message("Model loaded from \"./nn\"!");
    }
  }

  { // Draw.
    comp_area.y += comp_area.height + padding;
    if ((GuiButton(comp_area, "draw number") || IsKeyReleased(KEY_D)) && state != DRAWING) {
      state = DRAWING;
    }
  }
}


void UI::draw_neuron_info() {
  if (selected_neuron.x < 0 || selected_neuron.y < 0) return;

  const Layer& layer = nn->layers[(int)selected_neuron.x];
  matrix_t activation = layer.outputs.at(0, (int)selected_neuron.y);
  matrix_t biased = layer.biased.at(0, (int)selected_neuron.y);

  Rectangle area = area_neuron_info;
  DrawRectangleRec(area, color_pannel);

  Vector2 pos = { area.x, area.y };

  pos.x += padding; pos.y += padding;
  DrawText((std::string("Layer: ") + std::to_string((int)selected_neuron.x)).c_str(), pos.x, pos.y, font_size, BLACK);

  pos.y += font_size + padding;
  DrawText((std::string("Neuron: ") + std::to_string((int)selected_neuron.y)).c_str(), pos.x, pos.y, font_size, BLACK);

  pos.y += font_size + padding;
  DrawText((std::string("Activation: ") + std::to_string(activation)).c_str(), pos.x, pos.y, font_size, BLACK);

  pos.y += font_size + padding;
  DrawText((std::string("Biased: ") + std::to_string(biased)).c_str(), pos.x, pos.y, font_size, BLACK);

  if (selected_neuron.x > 0) {
    const Layer& prev = nn->layers[(int)(selected_neuron.x - 1)];
    for (int i = 0; i < prev.weights.rows(); i++) {
      matrix_t a = prev.outputs.at(0, i);
      matrix_t w = prev.weights.at(i, (int)selected_neuron.y);

      pos.y += font_size + padding;
      char buff[2048];
      snprintf(buff, 2048, "i: %.6f   w: %.6f", a, w);
      DrawText(buff, pos.x, pos.y, font_size, BLACK);
    }
    
  }
}


void UI::draw_message() {
  if (msg.empty()) return;

  // There will be a padding inside the msg box.
  Vector2 area = { 400, 60 };

  DrawRectangleRec(
    {
      (float)(GetScreenWidth() - padding - area.x),
      padding,
      area.x,
      area.y
    },
    { 221, 244, 255, 0xff });

  GuiDrawText(
    msg.c_str(),
    {
      GetScreenWidth() - padding - area.x + padding,
      2 * padding,
      area.x - 2 * padding,
      area.y - 2 * padding,
    },
    0, BLACK);
}


void UI::draw_progress() {
  Rectangle area = area_progress;

  float progress = 0.f;

  if (dset_train && dset_train->count() > 0) {
    progress = (nn->data_index) / (float) dset_train->count();
  }

  DrawRectangle(area.x, area.y, area.width, area.height, color_pannel);
  DrawRectangle(area.x, area.y, area.width * progress, area.height, color_progress);

  // Draw the progress presentage.
  char buff[64]; snprintf(buff, 64, "%.2f%%", progress * 100);
  DrawText(
    buff,
    (int) area.x + 10,
    (int) area.y + 2,
    30,
    BLACK);
}


void UI::draw_texture() {
  if (texture == nullptr) return;
  float scale = area_error_graph.width / (float) texture->width;
  area_texture.height = texture->height * scale;
  DrawTextureEx(
    *texture,
    { area_texture.x, area_texture.y },
    0.f,
    scale,
    WHITE);
}


void UI::draw_drawing_canvas() {
  brush_size += GetMouseWheelMove() * 5;
  if (brush_size < 2) brush_size = 2;
  if (brush_size > 50) brush_size = 50;

  // Clear the canvas.
  if (IsKeyPressed(KEY_C)) {
    _clear_canvas();
  }

  Vector2 mouse_pos = GetMousePosition();
  Rectangle area = {
    area_canvas.x + padding,
    area_canvas.y + padding,
    canvas_size.x,
    canvas_size.y,
  };

  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse_pos, area)) {
    // Paint circle into render texture
    // NOTE: To avoid discontinuous circles, we could store
    // previous-next mouse points and just draw a line using brush size
    BeginTextureMode(canvas);
    DrawCircle(
      (int)(mouse_pos.x - area.x),
      (int)(mouse_pos.y - area.y),
      brush_size,
      WHITE);
    EndTextureMode();
  }

  // Draw the canvas container.
  DrawRectangleRec(area_canvas, color_pannel);

  // NOTE: Render texture must be y-flipped due to default OpenGL coordinates (left-bottom)
  DrawTextureRec(
    canvas.texture,
    {
      0,
      0,
      (float)canvas.texture.width,
      (float)-canvas.texture.height
    },
    {
      area_canvas.x + padding,
      area_canvas.y + padding
    },
    WHITE);

  // Draw buttons.
  Rectangle btn{
    area.x,
    area.y + area.height + padding,
    btn_width,
    btn_height,
  };

  if (GuiButton(btn, "forward") || IsKeyReleased(KEY_F)) {
    _forward_canvas();
    _clear_canvas();
    state = IDLE;
  }

  btn.x += btn_width + padding;
  if (GuiButton(btn, "close")) {
    _clear_canvas();
    state = IDLE;
  }

  // Draw bursh as a reference.
  if (CheckCollisionPointRec(mouse_pos, area)) {
    DrawCircle(mouse_pos.x, mouse_pos.y, brush_size, WHITE);
  }
}


void UI::message(const std::string& msg) {
  this->msg = msg;
  msg_time = GetTime();
}


UI::State UI::get_state() const {
  return state;
}


void UI::set_state(UI::State state) {
  this->state = state;
}


void UI::render() {
  draw_nn_graph();
  draw_error_graph();
  draw_texture();
  draw_inputs();
  draw_progress();
  draw_neuron_info();
  draw_message();

  if (state == State::DRAWING) {
    draw_drawing_canvas();
  }
}


void UI::draw_nn_graph() {
#define return ?

  // If a neuron is selected we'll set selected_neuron to new_selection,
  // if it clicked inside a neuron
  bool is_neuron_selected = (
    (state == TRAINING || state == IDLE) &&
    _nn_graph_has_mouse() &&
    IsMouseButtonReleased(MOUSE_BUTTON_LEFT));
  Vector2 new_neuron_selection = { -1, -1 };

  // Mouse position in the graph coordinate.
  Vector2 mouse_pos_graph = GetScreenToWorld2D(GetMousePosition(), cam_nn);

  int max_activation_count = 0;
  for (const Layer& layer : nn->layers) {
    max_activation_count = std::max(max_activation_count, layer.outputs.cols());
  }

  // This will be the length between first neuron and last neuron of the longest layer.
  float max_layer_height = (max_activation_count - 1) * (neuron_gap + 2 * neuron_radius);

  // We'll draw from here to make sure the NN is in the middle of the view.
  float offset_x = (area_nn.width - ((nn->layers.size() - 1) * layer_gap)) / 2.f;
  float offset_y = (area_nn.height - max_layer_height) / 2.f;

  // Returns the position of a neuron.
  auto get_pos = [=](int layer_index, int neuron_index) {
    int cols = nn->layers[layer_index].outputs.cols();
    float curr_layer_height = (cols - 1) * (neuron_gap + 2 * neuron_radius);
    float x = offset_x + layer_gap * layer_index;
    float y = offset_y + (max_layer_height - curr_layer_height) / 2.f;
    y += (neuron_index) * (2 * neuron_radius + neuron_gap);
    #undef return
    return Vector2{ x, y };
    #define return ?
  };


  DrawRectangleRec(area_nn, color_nn_area);
  BeginMode2D(cam_nn);
  {

    // Draw connections.
    for (int layer_index = (int)(nn->layers.size()) - 1; layer_index >= 0; layer_index--) {
      const Layer& layer = nn->layers[layer_index];

      for (int neuron_index = 0; neuron_index < layer.outputs.cols(); neuron_index++) {
        Vector2 pos = get_pos(layer_index, neuron_index);
        Vector2 screen_pos = GetWorldToScreen2D({ pos.x, pos.y }, cam_nn);
        if (CheckCollisionPointRec(screen_pos, area_nn)) {
          if (layer_index > 0) {
            int prev_cols = nn->layers[layer_index - 1].outputs.cols();
            for (int j = 0; j < prev_cols; j++) {
              Vector2 pos_prev = get_pos(layer_index - 1, j);

              matrix_t w = nn->layers[layer_index - 1].weights.at(j, neuron_index);
              Color color = _interpolated_color(color_conn_min, color_conn_max, w);
              DrawLineEx(pos_prev, pos, 1, color);
            }
          }
        }
      }

    }

    // Draw the neuron.
    for (int layer_index = (int)(nn->layers.size()) - 1; layer_index >= 0; layer_index--) {
      const Layer& layer = nn->layers[layer_index];

      // Get the maximum confident neuron.
      int confident_neuron_index = -1;
      matrix_t max_conf = 0.f;
      for (int i = 0; i < layer.outputs.cols(); i++) {
        matrix_t curr = layer.outputs.at(0, i);
        if (curr >= max_conf) {
          confident_neuron_index = i;
          max_conf = curr;
        }
      }

      for (int neuron_index = 0; neuron_index < layer.outputs.cols(); neuron_index++) {
        Vector2 pos = get_pos(layer_index, neuron_index);
        Vector2 screen_pos = GetWorldToScreen2D({ pos.x, pos.y }, cam_nn);

        if (CheckCollisionPointRec(screen_pos, area_nn)) {

          // Handle mouse click in neuron.
          if (is_neuron_selected &&
              CheckCollisionPointCircle(mouse_pos_graph, pos, neuron_radius)) {

            // If not already selected, unselect.
            if (!(layer_index == selected_neuron.x && neuron_index == selected_neuron.y)) {
              new_neuron_selection = {
                (float) layer_index,
                (float) neuron_index
              };
            }

          }

          matrix_t activation = layer.outputs.at(0, neuron_index);
          Color color = _interpolated_color(color_neuron_min, color_neuron_max, activation);
          if (selected_neuron.x == layer_index && selected_neuron.y == neuron_index) {
            color = color_selected_neuron;
          }

          DrawCircle((int)pos.x, (int)pos.y, neuron_radius, color);

          // FIXME: Hardcoded values.
          char buff[64]; snprintf(buff, 64, "%.4f", activation);
          DrawText(
            buff,
            (int) pos.x - neuron_radius * .8,
            (int) pos.y - 5,
            20,
            BLACK);

          if (layer_index == nn->layers.size() - 1) {
            DrawText(
              nn->output_labels[neuron_index].c_str(),
              pos.x + neuron_radius + padding,
              pos.y - 15,
              40,
              ((neuron_index == confident_neuron_index) && max_conf >= .5)
                ? BLUE
                : RAYWHITE
            );
          }
        }
      }

    }

  }
  EndMode2D();

  if (is_neuron_selected) {
    selected_neuron = new_neuron_selection;
  }
#undef return
}

#endif // SINGLE_SOURCE_IMPL
