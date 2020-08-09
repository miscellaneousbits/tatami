#if defined(__TCC)
#include <tcclib.h>
#else
#include <math.h>
#include <stdio.h>
#endif

#define nMax 100000000  // default
//#define nMax 85765682        // for T(s) = 200
//#define nMax 1640000000      // for T(s) = 400

//#define nMaxSqrt sqrt(nMax)
#define nMaxSqrt 10000

unsigned v[nMax];

static void odd()
{
    for (int i = 7; i < nMaxSqrt; i += 2)
    {
        int k2 = i + 3;
        int k3 = i + i - 4;
        while ((k2 <= k3) && ((i * k2) < nMax))
        {
            int k4 = (nMax - 1) / i;
            if (k3 < k4)
                k4 = k3;
            for (int j = k2; j <= k4; j += 2)
                ++v[(i * j) / 2];
            k2 += i + 1;
            k3 += i - 1;
        }
    }
}

static void even()
{
    for (int i = 8; i < nMaxSqrt; i += 2)
    {
        int k2 = i + 3;
        int k3 = i + i - 4;
        while ((k2 <= k3) && ((i * k2) < nMax))
        {
            int k4 = (nMax - 1) / i;
            if (k3 < k4)
                k4 = k3;
            for (int j = k2; j <= k4; ++j)
                ++v[(i * j) / 2];
            k2 += i + 1;
            k3 += i - 1;
        }
    }
}

int Tatami(int s)
{
    for (unsigned i = 0; i < (nMax / 2); i++)
        v[i] = 0;
    odd();
    even();
    for (int i = 0; i < nMax / 2; ++i)
        if (v[i] == s)
            return i + i;
    return 0; // shouldn't happen
}

int main(int ac, char* av[])
{
    int s = 200;
    printf("T(%u)=%u\n", Tatami(s), s);
}

