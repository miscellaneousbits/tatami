/*  gplum.go -- Compute T(s) from Project Euler Problem 256
    Optimised from gprune.go December 28, 2019 by Eric Olson  */
#include <iostream>
#include <sstream>
#include <chrono>

using namespace std;

const int Pnum = 1300;
const int Smax = 100000000;
const int Fnum = 10;

typedef  struct {
    int s, fmax, i;
    int p[Fnum];
    int n[Fnum];
} factors;

int Tisn, smin, P[Pnum];

int in;

bool isprime(int p)
{
    int i;
    for (i = 0; i < in; i++)
    {
        if (p % P[i] == 0)
            return false;
    }
    for (i = in; P[i] * P[i] <= p; i++)
    {
        if (p % P[i] == 0)
            return false;
    }
    in = i - 1;
    return true;
}

void doinit() {
    int Pn;
    int p;
    smin = Smax;
    P[0] = 2;
    P[1] = 3;
    Pn = 2;
    in = 1;
    for (p = 5; Pn < Pnum; p += 2)
    {
        if (isprime(p))
            P[Pn++] = p;
    }
    if (p <= Smax / p + 1)
    {
        stringstream ss;
        ss << "The maximum prime " << p << " is too small!\n"; 
        throw runtime_error(ss.str());
    }
    int i, r = 1;
    for (i = 0; i < Fnum - 1; i++)
    {
        if (P[i] > Smax / r + 1)
            return;
        r *= P[i];
    }
    stringstream ss;
    ss << "Distinct primes " << Fnum << " in factorisation too few!\n";
    throw runtime_error(ss.str());
}

int tfree(int k, int l)
{
    int n = l / k;
    int lmin = (k + 1) * n + 2;
    int lmax = (k - 1) * (n + 1) - 2;
    if (lmin <= l && l <= lmax)
        return 1;
    return 0;
}

int ppow(int p, int n)
{
    int r = 1;
    for (; n; p *= p)
    {
        if ((n & 1) == 1)
            r *= p;
        n >>= 1;
    }
    return r;
}

int T(factors* x)
{
    int z[Fnum];
    for (int w = 0; w < Fnum; w++)
        z[w] = 0;
    int r = 0;
    for (;;)
    {
        int i;
        for (i = 0; i <= x->fmax; i++)
        {
            if (z[i] < x->n[i])
            {
                z[i]++;
                break;
            }
            z[i] = 0;
        }
        if (i > x->fmax)
            break;
        int k = 1, l = 1;
        for (i = 0; i <= x->fmax; i++)
            k *= ppow(x->p[i], z[i]);
        l = x->s / k;
        if (k <= l)
            r += tfree(k, l);
     }
    return r;
}

int sigma(factors* x)
{
    int r = x->n[0];
    for (int i = 1; i <= x->fmax; i++)
        r *= x->n[i] + 1;
    return r;
}

void Twork(factors* x)
{
    int s = x->s;
    int r = sigma(x);
    if (r >= Tisn)
    {
        r = T(x);   
        if (r == Tisn && s < smin)
            smin = s;
    }
    int i = x->i;   
    int fmax = x->fmax; 
    int pmax = smin / s + 1;
    int p = P[i];
    if (p <= pmax)
    {
        x->n[fmax]++;
        x->s = s * p;
        Twork(x);
        x->n[fmax]--;
    }
    fmax++;
    x->n[fmax] = 1;
    for (i++; i < Pnum; i++)
    {
        p = P[i];
        if (p > pmax)
            break;
        x->p[fmax] = p;
        x->s = s * p;
        x->i = i;
        x->fmax = fmax;
        Twork(x);
    }
    x->n[fmax] = 0;
}

int Tinv(int n) {
    Tisn = n;
    factors x;
    x.p[0] = P[0];
    x.n[0] = 1;
    x.i = 0;
    x.s = 2;
    x.fmax = 0;
    Twork(&x);
    if (smin < Smax)
        return smin;
    return -1;
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    int n = 200;
    doinit();
    int s = Tinv(n);
    cout << "T(" << s << ")=" << n << '\n';
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Duration: " << elapsed.count() << " seconds\n";
    return 0;
}

