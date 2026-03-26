#include "additionally.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GEMMCONV

#define TO 12
#define TI 4
int const run_single_image_test = 1;

// hex 저장용 이미지 카운터 (첫 번째 이미지에서만 저장)


#define MAX_VAL_8       127
#define MAX_VAL_16      32767
#define MAX_VAL_32      2147483647
#define MAX_VAL_UINT_8  255

// hex 파일 저장 경로
#define HEX_DIR "/Volumes/Yun_ssd/AIX2026/AIX/hex"

static int32_t layer_max_accumulator = 0;

int max_abs(int src, int max_val) {
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val - 1;
    return src;
}

short int max_abs_short(short int src, short int max_val) {
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val - 1;
    return src;
}

// ----------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------
int8_t im2col_get_pixel_int8(int8_t *im, int height, int width, int channels,
    int row, int col, int channel, int pad) {
    row -= pad; col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

void im2col_cpu_int8(int8_t* data_im, int channels, int height, int width,
    int ksize, int stride, int pad, int8_t* data_col) {
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col  = (width  + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im     = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row   = h_offset + h * stride;
                int im_col   = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_int8(
                    data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

void gemm_nn_int8_int32(int M, int N, int K, int8_t ALPHA,
                         int8_t *A, int lda, int8_t *B, int ldb,
                         int32_t *C, int ldc) {
    int32_t *c_tmp = (int32_t*)calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            int16_t A_PART = (int16_t)ALPHA * (int16_t)A[i * lda + k];
            for (j = 0; j < N; ++j) {
                c_tmp[j] += (int32_t)A_PART * (int32_t)B[k * ldb + j];
                if (abs(c_tmp[j]) > layer_max_accumulator)
                    layer_max_accumulator = abs(c_tmp[j]);
            }
        }
        for (j = 0; j < N; ++j) {
            C[i * ldc + j] += max_abs(c_tmp[j], MAX_VAL_32);
            c_tmp[j] = 0;
        }
    }
    free(c_tmp);
}

// ----------------------------------------------------------
// [HW 시뮬용] 레이어 입력 hex 3종 저장
//
// CONV{idx}_input.hex     : 8비트/줄,  planar 전체 (C×H×W)
// CONV{idx}_input_16b.hex : 16비트/줄, sign-extended
// CONV{idx}_input_32b.hex :
//   Ni=3 (CONV00, RGB) → [23:16]=B [15:8]=G [7:0]=R, 1픽셀/줄 (H×W줄)
//   Ni≥4              → TI=4 채널 패킹, [31:24]=ch3 [23:16]=ch2 [15:8]=ch1 [7:0]=ch0
// ----------------------------------------------------------
void save_input_hex(int layer_idx, int8_t *q, int C, int H, int W) {
    char path[256];
    int HW    = H * W;
    int total = C * HW;

    // 1) _input.hex : 8비트 1바이트/줄
    snprintf(path, sizeof(path), "%s/CONV%02d_input.hex", HEX_DIR, layer_idx);
    FILE *fp = fopen(path, "w");
    if (fp) {
        for (int z = 0; z < total; ++z)
            fprintf(fp, "%02x\n", (uint8_t)q[z]);
        fclose(fp);
        printf("[HEX] CONV%02d_input.hex      saved (%d lines)\n", layer_idx, total);
    }

    // 2) _input_16b.hex : 16비트 sign-extended 1개/줄
    snprintf(path, sizeof(path), "%s/CONV%02d_input_16b.hex", HEX_DIR, layer_idx);
    fp = fopen(path, "w");
    if (fp) {
        for (int z = 0; z < total; ++z) {
            int16_t v = (int16_t)q[z];
            fprintf(fp, "%04x\n", (uint16_t)v);
        }
        fclose(fp);
        printf("[HEX] CONV%02d_input_16b.hex  saved (%d lines)\n", layer_idx, total);
    }

    // 3) _input_32b.hex
    snprintf(path, sizeof(path), "%s/CONV%02d_input_32b.hex", HEX_DIR, layer_idx);
    fp = fopen(path, "w");
    if (fp) {
        if (C == 3) {
            // RGB 3채널: 픽셀 위치별 pack
            for (int i = 0; i < HW; ++i) {
                uint8_t r = (uint8_t)q[0 * HW + i];
                uint8_t g = (uint8_t)q[1 * HW + i];
                uint8_t b = (uint8_t)q[2 * HW + i];
                fprintf(fp, "%08x\n",
                        ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r);
            }
            printf("[HEX] CONV%02d_input_32b.hex  saved (%d lines, RGB pack)\n",
                   layer_idx, HW);
        } else {
            // feature map: 픽셀 위치 × TI=4 채널 pack
            int lines = 0;
            for (int i = 0; i < HW; ++i) {
                for (int c = 0; c < C; c += TI) {
                    uint8_t b0 = (uint8_t)q[(c+0)*HW + i];
                    uint8_t b1 = (c+1 < C) ? (uint8_t)q[(c+1)*HW + i] : 0;
                    uint8_t b2 = (c+2 < C) ? (uint8_t)q[(c+2)*HW + i] : 0;
                    uint8_t b3 = (c+3 < C) ? (uint8_t)q[(c+3)*HW + i] : 0;
                    fprintf(fp, "%08x\n",
                            ((uint32_t)b3 << 24) | ((uint32_t)b2 << 16) |
                            ((uint32_t)b1 <<  8) |  (uint32_t)b0);
                    lines++;
                }
            }
            printf("[HEX] CONV%02d_input_32b.hex  saved (%d lines, TI=4 pack)\n",
                   layer_idx, lines);
        }
        fclose(fp);
    }
}

// ----------------------------------------------------------
// Forward Convolutional Layer (Quantized)
// ----------------------------------------------------------
void forward_convolutional_layer_q(network net, layer l, network_state state) {
    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;
    int i, j;
    int const out_size = out_h * out_w;
    layer_max_accumulator = 0;

    typedef int32_t conv_t;
    conv_t *output_q = (conv_t*)calloc(l.outputs, sizeof(conv_t));
    state.input_uint8 = (int8_t*)calloc(l.inputs, sizeof(int8_t));

    // Input Quantization
    for (int z = 0; z < l.inputs; ++z) {
        float src_f = state.input[z] * l.input_quant_multiplier;
        state.input_uint8[z] = (int8_t)max_abs((int)src_f, MAX_VAL_8);
    }

    // [HW 시뮬용] 첫 번째 이미지에 대해서만 모든 CONV 레이어 입력 hex 저장
    // static hex_saved: CONV20(마지막) 저장 완료 시 1로 세팅 → 이후 이미지 스킵
    if (run_single_image_test) {
        static int hex_saved = 0;
        if (!hex_saved) {
            save_input_hex(state.index, state.input_uint8, l.c, l.h, l.w);
            if (state.index == 20) {
                hex_saved = 1;
                printf("[HEX] 첫 번째 이미지 hex 저장 완료 (이후 이미지 스킵)\n");
            }
        }
    }

    // Convolution
    int8_t *a = l.weights_int8;
    int8_t *b = (int8_t*)state.workspace;
    im2col_cpu_int8(state.input_uint8, l.c, l.h, l.w, l.size, l.stride, l.pad, b);

    #pragma omp parallel for
    for (int t = 0; t < l.n; ++t) {
        gemm_nn_int8_int32(1, out_size, l.size * l.size * l.c, 1,
                           a + t * l.size * l.size * l.c,
                           l.size * l.size * l.c,
                           b, out_size,
                           output_q + t * out_size, out_size);
    }

    // Bias + ReLU
    for (int fil = 0; fil < l.n; ++fil)
        for (j = 0; j < out_size; ++j)
            output_q[fil * out_size + j] += l.biases_quant[fil];

    if (l.activation == RELU)
        for (i = 0; i < l.n * out_size; ++i)
            output_q[i] = (output_q[i] > 0) ? output_q[i] : 0;

    // Descaling
    float ALPHA1 = 1.0f / (l.input_quant_multiplier * l.weights_quant_multiplier);
    for (i = 0; i < l.outputs; ++i)
        l.output[i] = (float)output_q[i] * ALPHA1;

    // 레이어 출력 hex 저장 (첫 번째 이미지만)
    if (run_single_image_test) {
        static int out_hex_saved = 0;
        if (!out_hex_saved) {
            char fpath[100];
            snprintf(fpath, sizeof(fpath), "./log_feamap/CONV%02d_output.hex", state.index);
            FILE *fp = fopen(fpath, "w");
            if (fp) {
                int next_m = 16;
                for (int z = state.index + 1; z < net.n; ++z) {
                    if (net.layers[z].type == CONVOLUTIONAL) {
                        next_m = net.layers[z].input_quant_multiplier;
                        break;
                    }
                }
                for (int chn = 0; chn < l.n; chn++) {
                    for (int idx = 0; idx < out_size; idx++) {
                        int i = chn * out_size + idx;
                        int16_t src = (int16_t)(l.output[i] * next_m);
                        fprintf(fp, "%02x\n", (uint8_t)max_abs(src, MAX_VAL_8));
                    }
                }
                fclose(fp);
            }
            if (state.index == 20)
                out_hex_saved = 1;
        }
    }

    free(state.input_uint8);
    free(output_q);
}

// ----------------------------------------------------------
// Quantization
// ----------------------------------------------------------
void do_quantization(network net) {
    int counter = 0;
    float weight_quant_multiplier[11] = {32, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512};
    float input_quant_multiplier[11]  = {128, 32, 32, 32, 32, 16, 32, 32, 64, 64, 16};

    for (int j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];
        if (l->type != CONVOLUTIONAL) continue;

        l->input_quant_multiplier   = (counter < 11) ? input_quant_multiplier[counter]  : 16;
        l->weights_quant_multiplier = (counter < 11) ? weight_quant_multiplier[counter] : 64;

        int total_w = l->size * l->size * l->c * l->n;
        for (int i = 0; i < total_w; ++i)
            l->weights_int8[i] = (int8_t)max_abs(
                (int)(l->weights[i] * l->weights_quant_multiplier), MAX_VAL_8);

        float b_mult = l->weights_quant_multiplier * l->input_quant_multiplier;
        for (int i = 0; i < l->n; ++i)
            l->biases_quant[i] = max_abs((int)(l->biases[i] * b_mult), MAX_VAL_16);

        counter++;
    }
}

// ----------------------------------------------------------
// Save Quantized Model Parameters
// ----------------------------------------------------------
void save_quantized_model(network net) {
    for (int j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];
        if (l->type != CONVOLUTIONAL) continue;

        char weightfile[100], biasfile[100], scalefile[100];
        sprintf(weightfile, "./log_param/CONV%02d_param_weight.hex", j);
        sprintf(biasfile,   "./log_param/CONV%02d_param_biases.hex", j);
        sprintf(scalefile,  "./log_param/CONV%02d_param_scales.hex", j);

        FILE *fp_w = fopen(weightfile, "w");
        FILE *fp_b = fopen(biasfile,   "w");
        FILE *fp_s = fopen(scalefile,  "w");
        if (!(fp_w && fp_b && fp_s)) continue;

        int filter_size = l->size * l->size * l->c;

        // weight: TI=4씩 패킹, [31:24]=i+3 [23:16]=i+2 [15:8]=i+1 [7:0]=i+0
        for (int f = 0; f < l->n; f++) {
            for (int i = 0; i < filter_size; i += TI) {
                uint8_t bw[4] = {0, 0, 0, 0};
                for (int k = 0; k < TI; k++)
                    if (i + k < filter_size)
                        bw[k] = (uint8_t)l->weights_int8[f * filter_size + i + k];
                fprintf(fp_w, "%02x%02x%02x%02x\n", bw[3], bw[2], bw[1], bw[0]);
            }
        }

        // bias: 16비트
        for (int f = 0; f < l->n; f++)
            fprintf(fp_b, "%04x\n", (uint16_t)l->biases_quant[f]);

        // scale = (in_m × w_m) / next_in_m
        int next_m = 1;
        for (int z = l->index + 1; z < net.n; ++z) {
            if (net.layers[z].type == CONVOLUTIONAL) {
                next_m = net.layers[z].input_quant_multiplier;
                break;
            }
        }
        if (next_m == 0) next_m = 1;
        int scale = (int)(l->input_quant_multiplier * l->weights_quant_multiplier) / next_m;
        for (int f = 0; f < l->n; f++)
            fprintf(fp_s, "%04x\n", (uint16_t)scale);

        fclose(fp_w); fclose(fp_b); fclose(fp_s);
    }
}

// ----------------------------------------------------------
// Network Forward
// ----------------------------------------------------------
void yolov2_forward_network_q(network net, network_state state) {
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];
        if      (l.type == CONVOLUTIONAL) forward_convolutional_layer_q(net, l, state);
        else if (l.type == MAXPOOL)       forward_maxpool_layer_cpu(l, state);
        else if (l.type == ROUTE)         forward_route_layer_cpu(l, state);
        else if (l.type == REORG)         forward_reorg_layer_cpu(l, state);
        else if (l.type == UPSAMPLE)      forward_upsample_layer_cpu(l, state);
        else if (l.type == SHORTCUT)      forward_shortcut_layer_cpu(l, state);
        else if (l.type == YOLO)          forward_yolo_layer_cpu(l, state);
        else if (l.type == REGION)        forward_region_layer_cpu(l, state);
        state.input = l.output;
    }
}

float *network_predict_quantized(network net, float *input) {
    network_state state;
    state.net = net; state.index = 0; state.input = input;
    state.truth = 0; state.train = 0; state.delta = 0;
    yolov2_forward_network_q(net, state);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;

    return net.layers[i].output;
}
