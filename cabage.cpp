#include <future>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

using namespace std;

#if !defined(T_DEBUG)
#define T_DEBUG 0
#endif

#if !defined(TS)
#define TS 200
#endif

#if TS == 200

typedef uint32_t prime_t;
const uint32_t Pnum = 1300, Fnum = 10;
const prime_t Smax = 100000000;

#elif TS == 1000

typedef uint64_t prime_t;
const uint32_t Pnum = 40000, Fnum = 20;
const prime_t Smax = 100000000000ULL;

#else
#error "Unsupported TS"
#endif

const uint32_t meow = 5, bark = 30;

typedef struct
{
    prime_t s, smin, d, p[Fnum];
    uint32_t fmax, i, n[Fnum];
} factors_t;

typedef future<prime_t> future_t;
typedef vector<future_t> futures_t;

static prime_t P[Pnum];

static bool isprime(prime_t p)
{
    uint32_t i;
    static uint32_t in = 1;
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

static void doinit()
{
    prime_t p;
    uint32_t Pn;
    P[0] = 2;
    P[1] = 3;
    Pn = 2;
    for (p = 5; Pn < Pnum; p += 2)
    {
        if (isprime(p))
            P[Pn++] = p;
    }
#if T_DEBUG
    if (p <= Smax / p + 1)
    {
        stringstream ss;
        ss << "The maximum prime " << p << " is too small!";
        throw runtime_error(ss.str());
    }
    prime_t r = 1;
    for (uint32_t i = 0; i < Fnum - 1; i++)
    {
        if (P[i] > Smax / r + 1)
            return;
        r *= P[i];
    }
    stringstream ss;
    ss << "Distinct primes " << Fnum << " in factorisation too few!";
    throw runtime_error(ss.str());
#endif
}

static inline bool tfree(prime_t k, prime_t l)
{
    prime_t n = l / k;
    prime_t lmin = (k + 1) * n + 2;
    prime_t lmax = (k - 1) * (n + 1) - 2;
    return (lmin <= l && l <= lmax);
}

static inline prime_t ppow(prime_t p, uint32_t n)
{
    prime_t r = 1;
    for (; n; p *= p, n >>= 1)
        if (n & 1)
            r *= p;
    return r;
}

static uint32_t T(factors_t& x)
{
    uint32_t z[Fnum];
    for (uint32_t& i : z)
        i = 0;
    uint32_t r = 0;
    for (;;)
    {
        uint32_t i;
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
        prime_t k = 1;
        for (i = 0; i <= x.fmax; i++)
            k *= ppow(x.p[i], z[i]);
        prime_t l = x.s / k;
        if (k <= l)
            if (tfree(k, l))
                r++;
    }
    return r;
}

static inline prime_t sigma(factors_t& x)
{
    prime_t r = x.n[0];
    for (uint32_t i = 1; i <= x.fmax; i++)
        r *= x.n[i] + 1;
    return r;
}

static void Twork(factors_t& x)
{
    prime_t s = x.s;
    prime_t d = x.d;
    if (sigma(x) >= TS)
        if (T(x) == TS && s < x.smin)
            x.smin = s;
    uint32_t i = x.i;
    uint32_t fmax = x.fmax;
    prime_t pmax = x.smin / s + 1;
    prime_t p = P[i];
    if (p <= pmax)
    {
        x.n[fmax]++;
        x.s = s * p;
        x.d = d + p;
        Twork(x);
        x.n[fmax]--;
    }
    fmax++;
    x.n[fmax] = 1;
    for (i++; i < Pnum; i++)
    {
        p = P[i];
        if (p > pmax)
            break;
        x.p[fmax] = p;
        x.s = s * p;
        x.d = d + p;
        x.i = i;
        x.fmax = fmax;
        Twork(x);
    }
    x.n[fmax] = 0;
}

static prime_t Pwork(factors_t& x);

static void Twrap(factors_t& x, futures_t& f)
{
    if (x.d > bark)
        Twork(x);
    else if (x.d < meow)
    {
        prime_t t = Pwork(x);
        if (t < x.smin)
            x.smin = t;
    }
    else
    {
        f.emplace_back(
            async(std::launch::async, [](factors_t x) { return Pwork(x); }, x));
    }
}

static prime_t Pwork(factors_t& x)
{
    prime_t s = x.s;
    prime_t d = x.d;
    futures_t f;
    if (sigma(x) >= TS)
        if (T(x) == TS && s < x.smin)
            x.smin = s;
    uint32_t i = x.i;
    uint32_t fmax = x.fmax;
    prime_t pmax = x.smin / s + 1;
    prime_t p = P[i];
    if (p <= pmax)
    {
        x.n[fmax]++;
        x.s = s * p;
        x.d = d + p;
        Twrap(x, f);
        x.n[fmax]--;
    }
    fmax++;
    x.n[fmax] = 1;
    for (i++; i < Pnum; i++)
    {
        p = P[i];
        if (p > pmax)
            break;
        x.p[fmax] = p;
        x.s = s * p;
        x.d = d + p;
        x.i = i;
        x.fmax = fmax;
        Twrap(x, f);
    }
    x.n[fmax] = 0;
    for (future_t& future : f)
    {
        prime_t t = future.get();
        if (t < x.smin)
            x.smin = t;
    }
    return x.smin;
}

static prime_t Tinv()
{
    factors_t x;
    x.i = 0;
    x.s = 2;
    x.d = 2;
    x.fmax = 0;
    x.smin = Smax;
    x.p[0] = P[0];
    x.n[0] = 1;
    prime_t smin =
        async(launch::async, [](factors_t x) { return Pwork(x); }, x).get();
	if (smin >= Smax)
	{
		stringstream ss;
		ss << "T(s)=" << TS << " not found";
		throw runtime_error(ss.str());
	}
    return smin;
}

int main()
{
    doinit();
    cout << "T(" << Tinv() << ")=" << TS << endl;
    return 0;
}
