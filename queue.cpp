// queue.cpp -- Compute T(s) from Project Euler Problem 256
// Written December 7, 2019 by Eric Olson
// Translated to C++, 2019 by Jean M. Cyr

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

using namespace std;

#define restrict __restrict

#if !defined(Ts)
#define Ts 200
#endif

#if Ts == 200
typedef uint32_t uint_prime_t;
const uint_prime_t sMax(100000000);
const uint32_t pNum(1300);
const uint32_t fNum(10);
#elif Ts == 1000
typedef uint64_t uint_prime_t;
const uint_prime_t sMax(100000000000ULL);
const uint32_t pNum(40000);
const uint32_t fNum(20);
#else
#error "Only Ts=200 and Ts=1000 are supported"
#endif

const uint32_t fido = 15;

typedef struct
{
    uint_prime_t s;
    uint32_t fmax, i;
    uint8_t n[fNum];
    uint_prime_t p[fNum];
} Factors;

class Threads
{
public:
    Threads(uint32_t);
    void enqueue(Factors& restrict);
    ~Threads();

private:
    vector<thread> workers;
    queue<Factors> tasks;
    mutex mtx;
    condition_variable condition;
    bool stop;
};

static void TWork(Factors& restrict);

inline Threads::Threads(uint32_t threads) : stop(false)
{
    for (uint32_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            for (;;)
            {
                Factors task;
                {
                    unique_lock<mutex> lock(mtx);
                    while (tasks.empty())
                    {
                        if (stop)
                            return;
                        condition.wait(lock);
                    }
                    task = move(tasks.front());
                    tasks.pop();
                }
                TWork(task);
            }
        });
}

void Threads::enqueue(Factors& restrict f)
{
    {
        unique_lock<mutex> lock(mtx);
        tasks.emplace(f);
    }
    condition.notify_one();
}

inline Threads::~Threads()
{
    {
        unique_lock<mutex> lock(mtx);
        stop = true;
    }
    condition.notify_all();
    for (thread& worker : workers)
        worker.join();
}

static uint_prime_t P[pNum];
static uint32_t num_primes;
static atomic<uint_prime_t> gMin(sMax);
static Threads* pool;

static inline bool IsPrime(uint_prime_t p)
{
    static uint32_t limit(1);
    uint32_t i;
    for (i = 1; i < limit; i++)
    {
        if (!(p % P[i]))
            return false;
    }
    for (i = limit; P[i] * P[i] <= p; i++)
    {
        if (!(p % P[i]))
            return false;
    }
    limit = i - 1;
    return true;
}

static void Primes()
{
    uint32_t p;

    P[0] = 2;
    P[1] = 3;
    num_primes = 2;
    for (p = 5; num_primes <= pNum; p += 2)
    {
        if (IsPrime(p))
            P[num_primes++] = p;
    }
}

static inline uint_prime_t ppow(uint_prime_t p, uint32_t n)
{
    uint_prime_t r(1);
    for (; n; p *= p)
    {
        if (n & 1)
            r *= p;
        n >>= 1;
    }
    return r;
}

static inline uint_prime_t sigma(Factors& restrict f)
{
    uint_prime_t r(f.n[0]);
    for (uint32_t i = 1; i <= f.fmax; i++)
        r *= f.n[i] + 1;
    return r;
}

static inline bool TFree(uint_prime_t k, uint_prime_t l)
{
    uint_prime_t n(l / k);
    uint_prime_t lmin((k + 1) * n + 2);
    uint_prime_t lmax((k - 1) * (n + 1) - 2);
    return lmin <= l && l <= lmax;
}

static uint32_t T(Factors& restrict f)
{
    uint8_t z[fNum] = {};
    uint32_t r(0);
    for (;;)
    {
        uint32_t i;
        for (i = 0; i <= f.fmax; i++)
        {
            if (z[i] < f.n[i])
            {
                z[i]++;
                break;
            }
            z[i] = 0;
        }
        if (i > f.fmax)
            break;
        uint_prime_t k(1);
        for (i = 0; i <= f.fmax; i++)
            k *= ppow(f.p[i], z[i]);
        uint_prime_t l(f.s / k);
        if ((k <= l) && TFree(k, l))
            r++;
    }
    return r;
}

static void TWork(Factors& restrict f)
{
    uint_prime_t p(P[f.i]);
    uint_prime_t s(f.s);
    uint_prime_t pMax(gMin.load() / s + 1);
    if (p <= pMax)
    {
        uint32_t fmax(f.fmax);
        f.n[fmax]++;
        f.s = s * p;
        if ((sigma(f) >= Ts) && (T(f) == Ts))
        {
            uint_prime_t sMin = gMin.load();
            while (f.s < sMin)
            {
                gMin.compare_exchange_weak(sMin, f.s);
                sMin = gMin.load();
            }
        }
        TWork(f);
        f.s = s;
        f.n[fmax]--;
        if (f.i >= pNum - 1)
            return;
        f.i++;
        if (f.n[fmax])
            f.fmax++;
        f.p[f.fmax] = P[f.i];
        f.n[f.fmax] = 0;
        TWork(f);
        f.fmax = fmax;
        f.i--;
    }
}

static void TQueue(Factors& restrict f)
{
    uint_prime_t s(f.s);
    uint_prime_t pMax(gMin.load() / s + 1);
    uint_prime_t p(P[f.i]);
    if (p <= pMax)
    {
        uint32_t fmax(f.fmax);
        if ((pow(log(pMax), sqrt(2)) / log(p)) < fido)
        {
            pool->enqueue(f);
            return;
        }
        f.n[fmax]++;
        f.s = s * p;
        if ((sigma(f) >= Ts) && (T(f) == Ts))
        {
            uint_prime_t sMin = gMin.load();
            while (f.s < sMin)
            {
                gMin.compare_exchange_weak(sMin, f.s);
                sMin = gMin.load();
            }
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
    cout << "Compiled using " << ((Ts == 200) ? "32" : "64")
         << " bit arithmetic" << endl;
    auto start = std::chrono::high_resolution_clock::now();
    Primes();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Prime time:   " << setprecision(3) << elapsed.count() << " seconds"
         << endl;
    start = end;
    Factors f;
    f.s = 2;
    f.p[0] = P[0];
    f.fmax = 0;
    f.i = 0;
    f.n[0] = 1;
    pool = new Threads(thread::hardware_concurrency());
    TQueue(f);
    delete pool;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Tatami time:  " << setprecision(3) << elapsed.count() << " seconds"
         << endl;
    cout << "T(" << gMin.load() << ")=" << Ts << endl;
    return 0;
}
