/*  prune.c -- Compute T(s) from Project Euler Problem 256
    Written November 28, 2019 by Eric Olson */

#include <iostream>

const uint64_t smax = 100000000000ULL;
const uint32_t Pnum = 40000;
const uint32_t fnum = 20;

typedef struct
{
    uint64_t s;
    uint32_t fmax, i, p[fnum];
    uint8_t n[fnum];
} factors;

factors x;
uint32_t Pn, Tisn, in;
uint64_t P[Pnum], smin;
uint8_t z[fnum];

bool tfree(uint64_t k, uint64_t  l)
{
    uint64_t n = l / k;
    uint64_t lmin = (k + 1) * n + 2;
    uint64_t lmax = (k - 1) * (n + 1) - 2;
    return lmin <= l && l <= lmax;
}

bool isprime(uint64_t p)
{
    uint32_t i;
    for (i = 1; i < in; i++)
    {
        if (!(p % P[i]))
            return 0;
    }
    for (i = in; P[i] * P[i] <= p; i++)
    {
        if (!(p % P[i]))
            return 0;
    }
    in = i - 1;
    return 1;
}

void doinit()
{
    uint32_t i, p;
    uint64_t r;
    smin = smax;
    P[0] = 2;
    P[1] = 3;
    Pn = 2, in = 1;
    for (p = 5; Pn < Pnum; p += 2)
    {
        if (isprime(p))
            P[Pn++] = p;
    }
    if (p <= smax / p + 1)
    {
        std::cout << "The maximum prime " << p << " is too small!" << std::endl;
        exit(1);
    }
    r = 1;
    for (i = 0; i < fnum - 1; i++)
    {
        if (P[i] > smax / r + 1)
            return;
        r *= P[i];
    }
    std::cout << "Distinct primes " << fnum << " in factorisation too few!" << std::endl;
    exit(2);
}

uint64_t ppow(uint64_t p, uint32_t n)
{
    uint64_t r;
    if (!n)
        return 1;
    r = 1;
    for (;;)
    {
        if (n & 1)
            r *= p;
        n >>= 1;
        if (!n)
            return r;
        p *= p;
    }
}

uint32_t sigma()
{
    uint32_t i;
    uint32_t r = x.n[0];
    for (i = 1; i <= x.fmax; i++)
        r *= x.n[i] + 1;
    return r;
}

uint32_t T()
{
    uint32_t r, w;
    for (w = 0; w < fnum; w++)
        z[w] = 0;
    r = 0;
    for (;;)
    {
        uint32_t i;
        uint64_t k, l;
        for (i = 0; i <= x.fmax; i++)
        {
            if (z[i] < x.n[i])
            {
                z[i]++;
                break;
            }
            z[i] = 0;
        }
        if (i > x.fmax)
            break;
        k = 1;
        l = 1;
        for (i = 0; i <= x.fmax; i++)
        {
            k *= ppow(x.p[i], z[i]);
            l *= ppow(x.p[i], x.n[i] - z[i]);
        }
        if (k <= l)
            r += tfree(k, l) ? 1 : 0;
    }
    return r;
}

void Twork()
{
    uint64_t s, pmax;
    uint32_t fmax, i, p, r;
    s = x.s;
    r = sigma();
    if (r >= Tisn)
    {
        r = T();
        if (r == Tisn && s < smin)
            smin = s;
    }
    i = x.i;
    fmax = x.fmax;
    pmax = smin / s + 1;
    p = (uint32_t)P[i];
    if (p <= pmax)
    {
        x.n[fmax]++;
        x.s = s * p;
        Twork();
        x.n[fmax]--;
    }
    fmax++;
    x.n[fmax] = 1;
    for (i++; i < Pnum; i++)
    {
        p = (uint32_t)P[i];
        if (p > pmax)
            break;
        x.p[fmax] = p;
        x.s = s * p;
        x.i = i;
        x.fmax = fmax;
        Twork();
    }
    x.n[fmax] = 0;
}

uint64_t Tinv(uint32_t n)
{
    Tisn = n;
    x.p[0] = uint32_t(P[0]);
    x.n[0] = 1;
    x.i = 0;
    x.s = 2;
    x.fmax = 0;
    Twork();
    return smin < smax ? smin : -1;
}

int main()
{
    uint32_t n = 1000;
    doinit();
    std::cout << "T(" << Tinv(n) << ")=" << n << std::endl;
    return 0;
}
