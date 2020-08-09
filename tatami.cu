// tatami.cu (simplified single kernel)

#include <cuda.h>

#include <iostream>

const unsigned nMax(100000000);
const unsigned nMaxSqrt(sqrt(nMax));

__global__ void count(unsigned* v, unsigned base)
{
    const unsigned i = threadIdx.x + base + 7;
    const unsigned loop_increment = (i & 1) + 1;
    unsigned k2 = i + 3;
    unsigned k3 = i + i - 4;
    while ((i * k2 < nMax) && (k2 <= k3))
    {
        const unsigned k4 = min((nMax - 1) / i, k3);
        for (unsigned j = k2; j <= k4; j += loop_increment)
            atomicAdd(v + i * j, 1);
        k2 += i + 1;
        k3 += i - 1;
    }
}

unsigned Tatami(unsigned s)
{
    unsigned* v;

    cudaMalloc(&v, sizeof(unsigned) * nMax);
    cudaMemset(v, 0, sizeof(unsigned) * nMax);
    const unsigned max_group_size = 1024;

    unsigned iterations = nMaxSqrt - 7;
    for (unsigned iteration = 0; iteration < iterations;
         iteration += max_group_size)
        count<<<1, min(iterations - iteration, max_group_size)>>>(v, iteration);

    unsigned* vh = new unsigned[nMax];
    cudaMemcpy(vh, v, sizeof(unsigned) * nMax, cudaMemcpyDeviceToHost);

    for (unsigned i = 7; i < nMax; i++)
        if (vh[i] == s)
            return i;

    return 0;  // shouldn't happen
}

int main()
{
    const unsigned s = 200;
    std::cout << "T(" << Tatami(s) << ")=" << s << std::endl;
}
