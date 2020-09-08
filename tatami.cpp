
#include <math.h>

#include <iostream>
#include <thread>

const unsigned nMax = 100000000;
const unsigned nMaxSqrt = sqrt(nMax);
unsigned* v;

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

static void odd()
{
    for (unsigned i = 7; i < nMaxSqrt; i += 2)
    {
        unsigned k2 = i + 3;
        unsigned k3 = i + i - 4;
        unsigned k5 = i * k2;
        while ((k2 <= k3) && (k5 < nMax))
        {
            unsigned k4 = MIN((nMax - 1) / i, k3);
            for (unsigned j = k2; j <= k4; j += 2)
                ++v[(i * j) / 2];
            k3 += i - 1;
            k5 += (k2 += i + 1);
        }
    }
}

static void even()
{
    for (unsigned i = 8; i < nMaxSqrt; i += 2)
    {
        unsigned k2 = i + 3;
        unsigned k3 = i + i - 4;
        unsigned k5 = i * k2;
        while ((k2 <= k3) && (k5 < nMax))
        {
            unsigned k4 = MIN((nMax - 1) / i, k3);
            for (unsigned j = k2; j <= k4; ++j)
                ++v[(i * j) / 2];
            k3 += i - 1;
            k5 += (k2 += i + 1);
        }
    }
}

unsigned Tatami(unsigned s)
{
    v = new unsigned[nMax / 2];
    for (unsigned i = 0; i < (nMax / 2); i++)
        v[i] = 0;

    std::thread evenThread(even);
    std::thread oddThread(odd);
    evenThread.join();
    oddThread.join();

    for (unsigned i = 0; i < nMax / 2; ++i)
        if (v[i] == s)
            return i + i;
    abort();  // shouldn't happen
}

int main(int ac, char* av[])
{
    unsigned s = 200;
    if (ac > 1)
        s = atoi(av[1]);
    std::cout << "T(" << Tatami(s) << ") = " << s << std::endl;
}

