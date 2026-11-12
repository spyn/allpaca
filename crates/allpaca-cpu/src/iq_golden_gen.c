// Golden generator for iq_golden.txt: random IQ blocks dequantised by
// llama.cpp's own dequantize_row_* (ggml-quants.c). Output, one line per type:
//   <name> <block bytes hex> <f32 bit patterns, little-endian hex>
//
// Built on xeres against ggml-org/llama.cpp 3e7344670adf63ce28527a4d42f2d71eca27c41e
// with the ROCm 7.1 clang (MSVC/Windows Kits include dirs passed with -isystem):
//   clang -O1 -ffp-contract=off -c <llama>/ggml/src/ggml-quants.c \
//     -I<llama>/ggml/include -I<llama>/ggml/src -I<llama>/ggml/src/ggml-cpu
//   clang -O1 -ffp-contract=off -c iq_golden_gen.c -I<llama>/ggml/include -I<llama>/ggml/src
//   clang -fuse-ld=lld ggml-quants.o iq_golden_gen.o -o gen.exe && gen.exe > iq_golden.txt
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t next_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

// A finite half: sign, exponent 3..20 (about 2^-12 .. 2^5), random mantissa.
static uint16_t finite_half(void) {
    uint32_t r = next_u32();
    uint16_t sign = (r & 1) ? 0x8000 : 0;
    uint16_t exp = 3 + (r >> 1) % 18;
    uint16_t man = (r >> 8) & 0x3ff;
    return sign | (uint16_t)(exp << 10) | man;
}

static void hex(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) printf("%02x", p[i]);
}

typedef void (*dequant_fn)(const void *, float *, int64_t);

static void emit(const char *name, size_t block_bytes, size_t nblk, int d_at_front, dequant_fn f) {
    uint8_t *blocks = malloc(block_bytes * nblk);
    float *out = malloc(sizeof(float) * QK_K * nblk);
    for (size_t b = 0; b < nblk; ++b) {
        uint8_t *blk = blocks + b * block_bytes;
        for (size_t i = 0; i < block_bytes; ++i) blk[i] = (uint8_t)next_u32();
        uint16_t d = finite_half();
        if (d_at_front) {
            memcpy(blk, &d, 2);
        } else {
            // IQ1_M: the half scale lives in the top nibble of the four
            // 16-bit scale words (block bytes 48..56).
            uint16_t sc[4];
            memcpy(sc, blk + 48, 8);
            for (int k = 0; k < 4; ++k) {
                sc[k] = (uint16_t)((sc[k] & 0x0fff) | (((d >> (4 * k)) & 0xf) << 12));
            }
            memcpy(blk + 48, sc, 8);
        }
    }
    f(blocks, out, (int64_t)(QK_K * nblk));
    printf("%s ", name);
    hex(blocks, block_bytes * nblk);
    printf(" ");
    hex((const uint8_t *)out, sizeof(float) * QK_K * nblk);
    printf("\n");
    free(blocks);
    free(out);
}

// Stubs for symbols ggml-quants.c references but the dequant paths never call.
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void)type; (void)ne; abort(); }
size_t ggml_type_size(enum ggml_type type) { (void)type; abort(); }
const char *ggml_type_name(enum ggml_type type) { (void)type; return "?"; }
void ggml_abort(const char *file, int line, const char *fmt, ...) {
    (void)fmt;
    fprintf(stderr, "ggml_abort %s:%d\n", file, line);
    abort();
}

int main(void) {
    const size_t nblk = 4;
    emit("iq2_xxs", sizeof(block_iq2_xxs), nblk, 1, (dequant_fn)dequantize_row_iq2_xxs);
    emit("iq2_xs", sizeof(block_iq2_xs), nblk, 1, (dequant_fn)dequantize_row_iq2_xs);
    emit("iq2_s", sizeof(block_iq2_s), nblk, 1, (dequant_fn)dequantize_row_iq2_s);
    emit("iq3_xxs", sizeof(block_iq3_xxs), nblk, 1, (dequant_fn)dequantize_row_iq3_xxs);
    emit("iq3_s", sizeof(block_iq3_s), nblk, 1, (dequant_fn)dequantize_row_iq3_s);
    emit("iq1_s", sizeof(block_iq1_s), nblk, 1, (dequant_fn)dequantize_row_iq1_s);
    emit("iq1_m", sizeof(block_iq1_m), nblk, 0, (dequant_fn)dequantize_row_iq1_m);
    emit("iq4_xs", sizeof(block_iq4_xs), nblk, 1, (dequant_fn)dequantize_row_iq4_xs);
    return 0;
}
