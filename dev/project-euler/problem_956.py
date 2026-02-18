"""
Project Euler Problem 956: Super Duper Sum

Key insight: Use roots of unity filter.

D(n, m) = (1/m) × Σ_{k=0}^{m-1} σ(n, ω^k)

where ω is a primitive m-th root of unity and
σ(n, x) = Π_{p^e || n} (1 + p·x + p²·x² + ... + p^e·x^e)
"""

import time
import sys
from collections import Counter

MOD = 999_999_001
TIMEOUT = 120  # seconds


def modinv(a: int, mod: int) -> int:
    return pow(a, mod - 2, mod)


def primitive_root(mod: int) -> int:
    """Find a primitive root modulo mod (mod must be prime)."""
    phi = mod - 1
    factors = []
    n = phi
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if p * p > n:
            break
        if n % p == 0:
            factors.append(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.append(n)

    for g in range(2, mod):
        is_primitive = True
        for p in factors:
            if pow(g, phi // p, mod) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return -1


def get_primitive_mth_root(m: int, mod: int) -> int:
    g = primitive_root(mod)
    return pow(g, (mod - 1) // m, mod)


def sieve_primes(n: int) -> list[int]:
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]


def compute_superduperfactorial_exponent(n: int, p: int) -> int:
    """Compute exponent of prime p in n★."""
    f = [0] * (n + 1)
    for k in range(1, n + 1):
        f[k] = f[k-1]
        kk = k
        while kk % p == 0:
            f[k] += 1
            kk //= p

    total = 0
    for k in range(1, n + 1):
        total += f[k] * (n - k + 1)
    return total


def geometric_sum(p: int, e: int, x: int, mod: int) -> int:
    """Compute 1 + p*x + p²*x² + ... + p^e * x^e (mod mod)"""
    px = (p * x) % mod
    if px == 1:
        return (e + 1) % mod
    numerator = (pow(px, e + 1, mod) - 1) % mod
    denominator = (px - 1) % mod
    return (numerator * modinv(denominator, mod)) % mod


def compute_sigma(factorization: Counter, x: int, mod: int) -> int:
    """Compute σ(n, x) = Π_{p^e || n} (1 + p·x + ... + p^e·x^e) mod mod"""
    result = 1
    for p, e in factorization.items():
        result = (result * geometric_sum(p, e, x, mod)) % mod
    return result


def solve():
    start = time.time()
    n = 1000
    m = 1000

    print(f"Computing D({n}★, {m}) mod {MOD}")

    primes = sieve_primes(n)
    print(f"[{time.time()-start:.2f}s] Found {len(primes)} primes up to {n}")

    # Compute factorization of n★
    factorization = Counter()
    for i, p in enumerate(primes):
        if time.time() - start > TIMEOUT:
            print(f"TIMEOUT after {i}/{len(primes)} primes in factorization")
            sys.exit(1)
        exp = compute_superduperfactorial_exponent(n, p)
        if exp > 0:
            factorization[p] = exp

    print(f"[{time.time()-start:.2f}s] Factorization done. 2^{factorization[2]}, 3^{factorization[3]}, 5^{factorization[5]}")

    # Get primitive m-th root of unity
    omega = get_primitive_mth_root(m, MOD)
    print(f"[{time.time()-start:.2f}s] ω = {omega}")

    # Compute D using roots of unity filter
    total = 0
    omega_k = 1
    for k in range(m):
        if time.time() - start > TIMEOUT:
            print(f"TIMEOUT at k={k}/{m} in roots of unity sum")
            sys.exit(1)
        if k % 100 == 0:
            print(f"[{time.time()-start:.2f}s] k={k}/{m}")
        sigma_k = compute_sigma(factorization, omega_k, MOD)
        total = (total + sigma_k) % MOD
        omega_k = (omega_k * omega) % MOD

    result = (total * modinv(m, MOD)) % MOD
    print(f"\n[{time.time()-start:.2f}s] Answer: {result}")
    return result


def verify_small():
    """Quick verification with smaller example."""
    start = time.time()

    # Test D(24, 3) = 21
    factorization = Counter({2: 3, 3: 1})  # 24 = 2^3 * 3
    m = 3
    omega = get_primitive_mth_root(m, MOD)

    total = 0
    omega_k = 1
    for k in range(m):
        sigma_k = compute_sigma(factorization, omega_k, MOD)
        total = (total + sigma_k) % MOD
        omega_k = (omega_k * omega) % MOD

    result = (total * modinv(m, MOD)) % MOD
    expected = 21
    print(f"D(24, 3) = {result}, expected {expected}, match: {result == expected}")

    # Test 6★ factorization
    primes = sieve_primes(6)
    factorization = Counter()
    for p in primes:
        exp = compute_superduperfactorial_exponent(6, p)
        if exp > 0:
            factorization[p] = exp
    print(f"6★ factorization: {dict(factorization)}")

    # D(6★, 6) using roots of unity
    m = 6
    omega = get_primitive_mth_root(m, MOD)
    total = 0
    omega_k = 1
    for k in range(m):
        sigma_k = compute_sigma(factorization, omega_k, MOD)
        total = (total + sigma_k) % MOD
        omega_k = (omega_k * omega) % MOD

    result = (total * modinv(m, MOD)) % MOD
    expected = 6_368_195_719_791_280 % MOD
    print(f"D(6★, 6) mod {MOD} = {result}, expected {expected}, match: {result == expected}")
    print(f"[{time.time()-start:.2f}s] Verification done")

    return result == expected


if __name__ == "__main__":
    print("Quick verification...")
    if verify_small():
        print("\n" + "="*60)
        solve()
    else:
        print("Verification failed!")
