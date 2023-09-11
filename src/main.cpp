
#include <vector>
#include <stdlib.h>
#include <time.h>

#define Matrix RaylibMatrix
  #include <raylib.h>
  #include <raymath.h>
  #define RAYGUI_IMPLEMENTATION
  #include "raygui.h"
  #undef RAYGUI_IMPLEMENTATION
#undef Matrix


#define assert(cond)             \
  do {                           \
    if (!(cond)) __debugbreak(); \
  } while (false)

#define SINGLE_SOURCE_IMPL
  #include "matrix.hpp"
  #include "nn.hpp"
  #include "utils.hpp"
  #include "ui.hpp"
#undef SINGLE_SOURCE_IMPL


// Returns the error.
float train(NN & nn, Dataset& dataset, int index) {
  
  float cost = 0.f;
  if (index < dataset.count()) {
    Matrix expected = dataset.get_output(index);

    nn.forward(dataset.get_input(index));
    cost = error(nn.get_outputs(), expected);
    nn.backprop(expected);
  }
  return cost;
}


int main(void) {

  DsMinist dataset(
    "../dataset/train-labels.idx1-ubyte",
    "../dataset/train-images.idx3-ubyte");

  NN nn({ 784, 20, 10, 10 }, { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" });

  UI ui(&nn, &dataset);

  Texture tex = LoadTextureFromImage(dataset.images[0]);
  ui.set_texture(&tex);

  while (!WindowShouldClose()) {
    ui.handle_inputs();

    switch (ui.get_state()) {
      case UI::TRAINING:
      {
        if (nn.data_index == dataset.count()) {
          nn.trained++;
          if (nn.trained >= 3) { // TODO: parameterize number 3.
            ui.set_state(UI::IDLE);
            ui.message("Model trained!");
            break;
          }
          nn.data_index = 0;
        }

        Image img = dataset.images[nn.data_index];
        if (IsTextureReady(tex)) UnloadTexture(tex);

        tex = LoadTextureFromImage(img);
        ui.set_texture(&tex);

        float cost = train(nn, dataset, nn.data_index);
        ui.push_error(cost);

        nn.data_index++;
        break;
      }
    }

    ui.update();

    BeginDrawing();
    {
      ClearBackground(RAYWHITE);

      ui.render();
    }
    EndDrawing();
  }

  if (IsTextureReady(tex)) UnloadTexture(tex);

  ui.cleanup();
  CloseWindow();

  return 0;
}
