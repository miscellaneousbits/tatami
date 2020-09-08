// tatami.cu (J.M.Cyr 2020)

#include <cuda.h>
#include <iostream>

const unsigned nMax(100000000);
const unsigned nMaxSqrt(sqrt(nMax));

__global__ void count(unsigned* v, unsigned base)
{
    const unsigned i(threadIdx.x + base);
    const unsigned loopIncrement((i & 1) + 1);
    unsigned k2(i + 3);
    unsigned k3(i + i - 4);
    unsigned k5(i * k2);
    while ((k5 < nMax) && (k2 <= k3))
    {
        const unsigned k4(min((nMax - 1) / i, k3));
        for (unsigned j(k2); j <= k4; j += loopIncrement)
            atomicAdd(v + ((i * j) / 2), 1);
        k3 += i - 1;
        k5 += (k2 += i + 1);
    }
}

unsigned Tatami(unsigned s)
{
    const unsigned sizeV(sizeof(unsigned) * nMax / 2);
    const unsigned groupSize(256);
    unsigned* v;

    cudaMalloc(&v, sizeV);
    cudaMemset(v, 0, sizeV);

    for (unsigned i(7); i < nMaxSqrt; i += groupSize)
        count<<<1, min(nMaxSqrt - i, groupSize)>>>(v, i);

    unsigned* vh = new unsigned[nMax / 2];
    cudaMemcpy(vh, v, sizeV, cudaMemcpyDeviceToHost);

    for (unsigned i(7); i < nMax / 2; i++)
        if (vh[i] == s)
            return i + i;

    abort();  // shouldn't happen
}

int main(int ac, char** av)
{
    unsigned s(200);
    if (ac >= 2)
        s = atoi(av[1]);
    std::cout << "T(" << Tatami(s) << ")=" << s << std::endl;
}
