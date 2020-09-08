#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define nMax 100000000
#define nMaxSqrt 10000

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

unsigned v[nMax / 2];

static void count()
{
    for (unsigned i = 7; i < nMaxSqrt; i++)
    {
        const unsigned inc = (i & 1) + 1;
        unsigned k2 = i + 3;
        unsigned k3 = i + i - 4;
        unsigned k5 = i * k2;

        while ((k2 <= k3) && (k5 < nMax))
        {
            unsigned k4 = MIN((nMax - 1) / i, k3);
            for (unsigned j = k2; j <= k4; j += inc)
                v[(i * j) / 2]++;
            k3 += i - 1;
            k5 += (k2 += i + 1);
        }
    }
}

unsigned Tatami(unsigned s)
{
    for (unsigned i = 0; i < (nMax / 2); i++)
        v[i] = 0;
    count();
    for (unsigned i = 0; i < nMax / 2; ++i)
        if (v[i] == s)
            return i + i;
    abort();  // shouldn't happen
}

int main(int ac, char* av[])
{
    int s = 200;
    if (ac > 1)
        s = atoi(av[1]);
    printf("T(%u)=%u\n", Tatami(s), s);
}

