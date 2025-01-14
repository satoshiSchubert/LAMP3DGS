/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// #define COLOR_CHANNEL (4)
// #define FEATURE_CHANNEL (35)
// #define EXTRA_CHANNEL (0)
// #define MEAN_RELATED_CHANNEL (COLOR_CHANNEL+FEATURE_CHANNEL)
// #define NUM_CHANNELS (COLOR_CHANNEL+FEATURE_CHANNEL+EXTRA_CHANNEL) // Default 3, RGB
// #define BLOCK_X 16
// #define BLOCK_Y 16

#define COLOR_CHANNEL (4)
#define FEATURE_CHANNEL (35)
#define EXTRA_CHANNEL (0)
#define MEAN_RELATED_CHANNEL (COLOR_CHANNEL+FEATURE_CHANNEL)
#define NUM_CHANNELS (COLOR_CHANNEL+FEATURE_CHANNEL+EXTRA_CHANNEL) // COLOR0-4, POSITION 4-7
#define BLOCK_X 16
#define BLOCK_Y 16

#endif