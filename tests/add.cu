${cluda_preamble}

#define Real ${float_type}

KERNEL
void add(GLOBAL_MEM Real* results, GLOBAL_MEM Real* a, int n) {
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    results[i] = a[i] + ${b};
}
