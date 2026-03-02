#include "wav2letter.h" 
#include <stdio.h>
#include "menu.h"
#include "tflite.h"

#include <cstring>
#include "model/wav2letter_pruned_int8.h"
#include "test_data/test_input.h"
#include "test_data/test_output.h"  

#include "playground_util/console.h"

static void wav2letter_pruned_init(void) {
  tflite_load_model(wav2letter_pruned_int8_tflite, wav2letter_pruned_int8_tflite_len);
}



static void do_golden_tests() {
  printf("Running golden test...\n");

  printf("Setting model input...\n");
  tflite_set_input(g_test_input_data);

  printf("Running inference...\n");
  tflite_classify();

  int8_t* output = tflite_get_output();
  printf("Inference complete, comparing output...\n");

  bool passed = true; 
  for (size_t i = 0; i < g_test_output_data_len; ++i) {
    int8_t expected_val = (int8_t)g_test_output_data[i];
    if (output[i] != expected_val) {
      printf("\n*** FAIL: Golden test failed.\n");
      printf("Mismatch at byte index %u:\n", (unsigned int)i);
      printf("  Actual:   %d\n", output[i]);
      printf("  Expected: %d\n", expected_val);
      
      passed = false;
      break; 
    }
  }
  
  if (passed) {
    printf("\nOK   Golden tests passed!\n");
  }
}

static struct Menu MENU = {
    "Tests for wav2letter_pruned",
    "wav2letter",
    {
        MENU_ITEM('g', "Run golden tests", do_golden_tests),
        MENU_END,
    },
};

void wav2letter_pruned_menu() {
  wav2letter_pruned_init();
  menu_run(&MENU);
}