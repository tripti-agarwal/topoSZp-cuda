// fstats <float32 file>
// Prints "FSTATS n=<values> min=<min> max=<max> range=<max-min> absmax=<max |v|> nan=<count>"
// for a raw little-endian float32 file (NaNs are skipped).
#include <cstdio>
#include <cmath>
#include <vector>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: fstats <float32 file>\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    std::vector<float> b(1 << 20);
    size_t k, n = 0, nan = 0;
    double mn = INFINITY, mx = -INFINITY;
    while ((k = fread(b.data(), sizeof(float), b.size(), f)) > 0) {
        for (size_t i = 0; i < k; i++) {
            float v = b[i];
            if (v != v) { nan++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        n += k;
    }
    fclose(f);
    if (!(mx >= mn)) { mn = 0; mx = 0; }
    double am = std::fmax(std::fabs(mn), std::fabs(mx));
    printf("FSTATS n=%zu min=%.9g max=%.9g range=%.9g absmax=%.9g nan=%zu\n", n, mn, mx, mx - mn, am, nan);
    return 0;
}
