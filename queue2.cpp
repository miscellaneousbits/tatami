// queue.cpp -- Compute T(s) from Project Euler Problem 256
// Written December 7, 2019 by Eric Olson
// Translated to C++, 2019 by Jean M. Cyr

#if !defined(TDEBUG)
#define TDEBUG 0
#endif

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if TDEBUG
#include <sstream>
#endif

using namespace std;

#if !defined(Ts)
#define Ts 1000
#endif

#if Ts == 200
typedef uint32_t uintp_t;
const uintp_t sMax(100000000);
const uint32_t pNum(1300);
const uint32_t fNum(20);
#elif Ts == 1000
typedef uint64_t uintp_t;
const uintp_t sMax(100000000000ULL);
const uint32_t pNum(40000);
const uint32_t fNum(20);
#else
#error "Only Ts=200 and Ts=1000 are supported"
#endif

typedef struct
{
    uintp_t s, p[fNum];
    uint32_t fmax, i;
    uint8_t n[fNum];
} Factors;

class Threads
{
public:
    Threads(uint32_t);
    void enqueue(Factors& f);
    ~Threads();

private:
    vector<thread> workers;
    queue<Factors> tasks;
    mutex tMutex;
    condition_variable condition;
    bool stop;
};

static void TWork(Factors& f);

inline Threads::Threads(uint32_t threads) : stop(false)
{
    for (uint32_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            Factors f;
            for (;;)
            {
                {
                    unique_lock<mutex> lock(tMutex);
                    if (tasks.empty())
                    {
                        if (stop)
                            break;
                        else
                            condition.wait(lock);
                    }
                    f = tasks.front();
                    tasks.pop();
                }
                TWork(f);
            }
        });
}

inline void Threads::enqueue(Factors& f)
{
    {
        unique_lock<mutex> lock(tMutex);
        tasks.push(f);
    }
    condition.notify_one();
}

inline Threads::~Threads()
{
    {
        unique_lock<mutex> lock(tMutex);
        stop = true;
    }
    condition.notify_all();
    for (thread& worker : workers)
        worker.join();
}

static uint32_t numPrimes, iLim;
static uintp_t P[pNum];
static atomic<uintp_t> gMin(sMax);
static Threads* pool;

static inline bool IsPrime(uintp_t p)
{
    uint32_t i;
    for (i = 1; i < iLim; i++)
        if (!(p % P[i]))
            return false;
    for (i = iLim; P[i] * P[i] <= p; i++)
        if (!(p % P[i]))
            return false;
    iLim = i - 1;
    return true;
}

static void Primes()
{
    uint32_t p;

    P[0] = 2;
    P[1] = 3;
    numPrimes = 2;
    iLim = 1;
    for (p = 5; numPrimes < pNum; p += 2)
        if (IsPrime(p))
            P[numPrimes++] = p;
#if TDEBUG
    uint32_t i;
    uintp_t r;
    stringstream ss;
    if (p <= sMax / p + 1)
    {
        ss << "The maximum prime " << p << " is too small!";
        throw runtime_error(ss.str());
    }
    r = 1;
    for (i = 0; i < fNum - 1; i++)
    {
        if (P[i] > sMax / r + 1)
            return;
        r *= P[i];
    }
    ss << "Distinct Primes " << fNum << " in factorisation too few!";
    throw runtime_error(ss.str());
#endif
}

static inline uintp_t ppow(uintp_t p, uint32_t n)
{
    uintp_t r = 1;
    for (; n; p *= p)
    {
        if (n & 1)
            r *= p;
        n >>= 1;
    }
    return r;
}

static inline uintp_t sigma(Factors& x)
{
    uintp_t r = x.n[0];
    for (uint32_t i = 1; i <= x.fmax; i++)
        r *= x.n[i] + 1;
    return r;
}

static inline bool TFree(uintp_t k, uintp_t l)
{
    uintp_t n = l / k;
    uintp_t lmin = (k + 1) * n + 2;
    uintp_t lmax = (k - 1) * (n + 1) - 2;
    return lmin <= l && l <= lmax;
}

static uint32_t T(Factors& x)
{
    uint8_t z[fNum];
	for (uint32_t i = 0; i < fNum; i++)
		z[i] = 0;
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
        uintp_t k = 1;
        for (i = 0; i <= x.fmax; i++)
            k *= ppow(x.p[i], z[i]);
		uintp_t l = x.s / k;
        if ((k <= l) && TFree(k, l))
            r++;
    }
    return r;
}

static void TWork(Factors& x)
{
    uintp_t s = x.s;
    uintp_t r = sigma(x);
    if (r >= Ts)
    {
        r = T(x);   
        if (r == Ts && s < smin)
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

static void TQueue(Factors& f)
{
    uintp_t s(f.s);
    uintp_t pMax(gMin.load() / s + 1);
    uintp_t p(P[f.i]);
    if (p <= pMax)
    {
        uint32_t fmax(f.fmax);
        if ((pow(log(pMax), sqrt(2)) / log(p)) < 15)
        {
            pool->enqueue(f);
            return;
        }
        f.n[fmax]++;
        f.s = s * p;
        if ((sigma(f) >= Ts) && (T(f) == Ts))
		{
            uintp_t sMin(gMin.load());
            while ((f.s < sMin) && !(gMin.compare_exchange_weak(sMin, f.s)))
                sMin = gMin.load();
		}
        TQueue(f);
        f.s = s;
        f.n[fmax]--;
        if (f.i >= pNum - 1)
            return;
        f.i++;
        if (f.n[fmax])
            f.fmax++;
        f.p[f.fmax] = P[f.i];
        f.n[f.fmax] = 0;
        TQueue(f);
        f.fmax = fmax;
        f.i--;
    }
}

int main()
{
    Primes();
cout << "xxx\n";
    Factors f = {.s = 2, .p = {P[0]}, .fmax = 0, .i = 0, .n = {1}};
    pool = new Threads(thread::hardware_concurrency());
    TQueue(f);
    delete pool;
    cout << "T(" << gMin.load() << ")=" << Ts << endl;
    return 0;
}
