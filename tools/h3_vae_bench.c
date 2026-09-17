/* Times the video VAE decode alone, on a real latent, so decoder work does not
 * pay for a denoise between attempts (the 15 s preset's latent takes minutes to
 * produce and ~22 s to decode).
 *
 * Produce a latent with any generation:
 *   H3_DUMP_VIDEO_LATENT=latent.bin ./h3 -d "$H3_MODEL_ROOT" -p ... -o out.mp4
 * then:
 *   ./h3_vae_bench latent.bin [iterations] [--save rgb.u8] [--ref rgb.u8]
 *
 * The decoder is loaded once and kept, as in a session. Each decode prints its
 * wall time and a hash of the 8-bit frames (the same conversion the pipeline
 * muxes), so a variant that claims to be bit-identical can be checked. --ref
 * reports PSNR against a saved decode.
 *
 * Build: make -f Makefile.linux h3_vae_bench
 */

#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

static uint8_t *to_u8(const float *rgb, size_t count) {
    uint8_t *output = malloc(count);
    if (!output) return NULL;
    for (size_t index = 0; index < count; index++) {
        float scaled = rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        output[index] = (uint8_t)lrintf(scaled);
    }
    return output;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total) fprintf(stderr, "video VAE load: %d/%d\n",
                                    completed, total);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s latent.bin [iterations] [--save rgb.u8] "
                "[--ref rgb.u8]\n",
                argv[0]);
        return 2;
    }
    const char *latent_path = argv[1];
    int iterations = 3;
    const char *save_path = NULL, *ref_path = NULL;
    for (int arg = 2; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--save") && arg + 1 < argc)
            save_path = argv[++arg];
        else if (!strcmp(argv[arg], "--ref") && arg + 1 < argc)
            ref_path = argv[++arg];
        else
            iterations = atoi(argv[arg]);
    }
    if (iterations < 1) iterations = 1;

    const char *model_root = getenv("H3_MODEL_ROOT");
    if (!model_root) {
        fprintf(stderr, "set H3_MODEL_ROOT\n");
        return 2;
    }
    FILE *file = fopen(latent_path, "rb");
    char magic[8];
    int32_t header[4];
    if (!file || fread(magic, 8, 1, file) != 1 ||
        memcmp(magic, "H3VLAT1", 8) ||
        fread(header, sizeof(header), 1, file) != 1) {
        fprintf(stderr, "%s is not an H3_DUMP_VIDEO_LATENT file\n",
                latent_path);
        return 1;
    }
    int latent_t = header[0], latent_h = header[1], latent_w = header[2];
    size_t count = (size_t)header[3] * latent_t * latent_h * latent_w;
    float *latent = malloc(count * sizeof(*latent));
    if (!latent || fread(latent, sizeof(*latent), count, file) != count) {
        fprintf(stderr, "short latent file\n");
        return 1;
    }
    fclose(file);
    printf("latent %d x %d x %d x %d\n", header[3], latent_t, latent_h,
           latent_w);

    char weights[1024], error[512] = {0};
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source",
             model_root);
    double start = now_seconds();
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        weights, "h3_shaders.metal", latent_h, latent_w, progress, NULL,
        error, sizeof(error));
    if (!decoder) {
        fprintf(stderr, "load failed: %s\n", error);
        return 1;
    }
    printf("load %.3f s\n", now_seconds() - start);

    uint8_t *reference = NULL;
    size_t reference_bytes = 0;
    if (ref_path) {
        FILE *ref = fopen(ref_path, "rb");
        if (ref) {
            fseek(ref, 0, SEEK_END);
            reference_bytes = (size_t)ftell(ref);
            fseek(ref, 0, SEEK_SET);
            reference = malloc(reference_bytes);
            if (!reference ||
                fread(reference, 1, reference_bytes, ref) != reference_bytes)
                reference_bytes = 0;
            fclose(ref);
        }
        if (!reference_bytes) {
            fprintf(stderr, "could not read %s\n", ref_path);
            return 1;
        }
    }

    for (int iteration = 0; iteration < iterations; iteration++) {
        h3_video_frames frames;
        start = now_seconds();
        if (!h3_video_vae_decoder_decode(decoder, latent, latent_t, &frames,
                                         error, sizeof(error))) {
            fprintf(stderr, "decode failed: %s\n", error);
            return 1;
        }
        double wall = now_seconds() - start;
        size_t bytes = (size_t)frames.frames * frames.height * frames.width * 3;
        uint8_t *rgb = to_u8(frames.rgb, bytes);
        uint64_t hash = UINT64_C(1469598103934665603);
        for (size_t index = 0; index < bytes; index++) {
            hash ^= rgb[index];
            hash *= UINT64_C(1099511628211);
        }
        printf("decode %d: %.3f s  %d frames %dx%d  peak %.2f GiB  rgb %016llx",
               iteration, wall, frames.frames, frames.width, frames.height,
               (double)frames.gpu_stats.peak_live_bytes /
                   (1024.0 * 1024.0 * 1024.0),
               (unsigned long long)hash);
        if (reference) {
            if (reference_bytes != bytes) {
                printf("  ref size mismatch");
            } else {
                double squared = 0.0;
                for (size_t index = 0; index < bytes; index++) {
                    double diff = (double)rgb[index] - reference[index];
                    squared += diff * diff;
                }
                double mse = squared / (double)bytes;
                if (mse == 0.0)
                    printf("  identical to ref");
                else
                    printf("  PSNR %.2f dB", 10.0 * log10(255.0 * 255.0 / mse));
            }
        }
        printf("\n");
        fflush(stdout);
        if (save_path && iteration == iterations - 1) {
            FILE *out = fopen(save_path, "wb");
            if (!out || fwrite(rgb, 1, bytes, out) != bytes) {
                fprintf(stderr, "could not write %s\n", save_path);
                return 1;
            }
            fclose(out);
        }
        free(rgb);
        h3_video_frames_free(&frames);
    }
    h3_video_vae_decoder_free(decoder);
    free(latent);
    free(reference);
    return 0;
}
