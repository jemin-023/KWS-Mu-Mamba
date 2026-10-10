# ESP-IDF Component CMakeLists.txt for engine
idf_component_register(
    SRCS
        "src/mm_math.c"
        "src/mm_frontend.c"
        "src/mm_block_f32.c"
        "src/mm_block_q.c"
        "src/mm_model.c"
        "gen/mm_weights.c"
        "gen/mm_weights_q.c"
        "gen/mm_luts.c"
    INCLUDE_DIRS
        "include"
        "gen"
    REQUIRES
)
