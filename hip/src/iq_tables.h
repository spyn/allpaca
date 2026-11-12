/* Lookup tables for the GGUF i-quant kernels.
 *
 * Copied from llama.cpp `ggml/src/ggml-common.h` at commit
 * 3e7344670adf63ce28527a4d42f2d71eca27c41e (MIT License, Copyright (c)
 * 2023-2026 The ggml authors), the commit `crates/allpaca-cpu/src/iq_grids.rs`
 * and the goldens in `iq_golden.txt` use. */
#ifndef ALLPACA_IQ_TABLES_H
#define ALLPACA_IQ_TABLES_H

#include <stdint.h>

/* `kvalues_iq4nl`: the 16 non-linear levels of IQ4_NL and IQ4_XS. */
constexpr int8_t k_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                1,    13,   25,  38,  53,  69,  89,  113};

/* Eight levels as the bytes of one word, so a kernel selects a level with a
 * shift and never reads the table from memory. */
constexpr uint64_t iq4nl_pack8(int first) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(static_cast<uint8_t>(k_iq4nl[first + i])) << (8 * i);
    }
    return v;
}
constexpr uint64_t K_IQ4NL_LO = iq4nl_pack8(0);
constexpr uint64_t K_IQ4NL_HI = iq4nl_pack8(8);

#endif
