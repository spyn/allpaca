// Golden generator for rocmfpx_golden.txt: random ROCmFPX blocks dequantised
// by the fork's own rocmfpx_dequantize_row_* (ggml/rocmfpx/rocmfpx.c).
// Output, one line per type:
//   <ggml type id> <block bytes hex> <f32 bit patterns, little-endian hex>
//
// Built on xeres against ROCmFPX/ROCmFPX 120141f1f9da0a82bb109135c82cd173107fac25
// (ggml/include/ggml.h, ggml/rocmfpx/rocmfpx.{h,c}) with the ROCm 7.1 clang:
//   clang -O1 -ffp-contract=off -c rocmfpx.c -I<dir with ggml.h, rocmfpx.h>
//   clang -O1 -ffp-contract=off -c rocmfpx_golden_gen.c -I<same>
//   clang -fuse-ld=lld rocmfpx.o rocmfpx_golden_gen.o -o gen.exe && gen.exe > rocmfpx_golden.txt
#include "rocmfpx.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng = 0xd1b54a32d192ed03ull;
static uint32_t next_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

// A UE4M3 scale byte: mostly valid (0x00..0x7e), now and then an invalid one
// (0x7f or a set sign bit), which the reference decodes as a zero scale.
static uint8_t scale_byte(void) {
    uint32_t r = next_u32();
    if (r % 16 == 0) {
        return (uint8_t)(0x7f + (r >> 8) % 0x81);
    }
    return (uint8_t)((r >> 8) % 0x7f);
}

typedef void (*dequant_fn)(const void *, float *, int64_t);

static void hex(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) printf("%02x", p[i]);
}

// `scales` scale bytes sit at the end of each `block_bytes` block.
static void emit(int id, size_t block_bytes, size_t scales, dequant_fn f) {
    const size_t nblk = 16;
    uint8_t *blocks = malloc(block_bytes * nblk);
    float *out = malloc(sizeof(float) * QK_ROCMFPX * nblk);
    for (size_t b = 0; b < nblk; ++b) {
        uint8_t *blk = blocks + b * block_bytes;
        for (size_t i = 0; i < block_bytes - scales; ++i) blk[i] = (uint8_t)next_u32();
        for (size_t s = 0; s < scales; ++s) blk[block_bytes - scales + s] = scale_byte();
    }
    f(blocks, out, (int64_t)(QK_ROCMFPX * nblk));
    printf("%d ", id);
    hex(blocks, block_bytes * nblk);
    printf(" ");
    hex((const uint8_t *)out, sizeof(float) * QK_ROCMFPX * nblk);
    printf("\n");
    free(blocks);
    free(out);
}

int main(void) {
    emit(102, sizeof(block_rocmfp6), 2, (dequant_fn)rocmfpx_dequantize_row_fp6);
    emit(103, sizeof(block_rocmfp8), 1, (dequant_fn)rocmfpx_dequantize_row_fp8);
    emit(104, sizeof(block_rocmfp3), 2, (dequant_fn)rocmfpx_dequantize_row_fp3);
    emit(109, sizeof(block_rocmfp5), 2, (dequant_fn)rocmfpx_dequantize_row_fp5);
    emit(110, sizeof(block_rocmfp7), 2, (dequant_fn)rocmfpx_dequantize_row_fp7);
    emit(111, sizeof(block_rocmfp2), 2, (dequant_fn)rocmfpx_dequantize_row_fp2);
    return 0;
}
