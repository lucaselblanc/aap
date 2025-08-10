#include "secp256k1.h"
#include <gmpxx.h>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <sys/sysinfo.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <unordered_map>
#include <array>
#include <vector>

/******************************************************************************************************
 * This file is part of the Pollard's Rho distribution: (https://github.com/lucaselblanc/pollardsrho) *
 * Copyright (c) 2024, 2025 Lucas Leblanc.                                                            *
 * Distributed under the MIT software license, see the accompanying.                                  *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.                               *
 ******************************************************************************************************/

/*****************************************
 * Pollard's Rho Algorithm for SECP256K1 *
 * Written by Lucas Leblanc              *
 * Optimized Implementation with Fixes   *
******************************************/

using namespace boost::multiprecision;

ECPoint G, H;
mpz_t P, GX, GY, N;

void init_secp256k1() {
    mpz_inits(P, N, GX, GY, NULL);

    mpz_set_str(P, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16);
    mpz_set_str(N, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
    mpz_set_str(GX, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16);
    mpz_set_str(GY, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16);

    point_init(&G);
    mpz_set(G.x, GX);
    mpz_set(G.y, GY);
    G.infinity = 0;

    point_init(&H);
    H.infinity = 0;
}

struct DPEntry {
    std::array<uint8_t, 33> compressed;
    uint256_t a, b;

    DPEntry() : a(0), b(0) {
        compressed.fill(0);
    }

    DPEntry(const std::array<uint8_t, 33>& comp, uint256_t a_val, uint256_t b_val) 
        : compressed(comp), a(a_val), b(b_val) {}
};

struct CompressedPointHash {
    std::size_t operator()(const std::array<uint8_t, 33>& key) const {
        std::size_t result = 0;
        for (int i = 0; i < 8; ++i) {
            result = (result << 8) | key[i];
        }
        return result;
    }
};

// Shards System
class ShardedDPTable {
    private:
    static constexpr int NUM_SHARDS = 64;
    std::vector<std::unordered_map<std::array<uint8_t, 33>, DPEntry, CompressedPointHash>> shards;
    std::vector<std::mutex> shard_mutexes;

    int get_shard_index(const std::array<uint8_t, 33>& key) const {
        CompressedPointHash hasher;
        return hasher(key) % NUM_SHARDS;
    }

    public:
    ShardedDPTable() : shards(NUM_SHARDS), shard_mutexes(NUM_SHARDS) {
        for (auto& shard : shards) {
            shard.reserve(10000);
        }
    }

    bool insert_or_find_collision(const std::array<uint8_t, 33>& key, const DPEntry& entry, DPEntry& existing) {
        int shard_idx = get_shard_index(key);
        std::lock_guard<std::mutex> lock(shard_mutexes[shard_idx]);

        auto& shard = shards[shard_idx];
        auto it = shard.find(key);

        if (it != shard.end()) {
            existing = it->second;
            return true;
        } else {
            shard[key] = entry;
            return false;
        }
    }

    size_t total_size() const {
        size_t total = 0;
        for (const auto& shard : shards) {
            total += shard.size();
        }
        return total;
    }
};

// Decompress Point (Tonelli-Shanks)
bool decompress_point(ECPoint* point, const std::array<uint8_t, 33>& compressed) {
    if (compressed[0] != 0x02 && compressed[0] != 0x03) {
        return false;
    }

    mpz_import(point->x, 32, 1, 1, 0, 0, &compressed[1]);

    mpz_t x3, y_squared, seven;
    mpz_inits(x3, y_squared, seven, NULL);
    mpz_powm_ui(x3, point->x, 3, P);
    mpz_set_ui(seven, 7);
    mpz_add(y_squared, x3, seven);
    mpz_mod(y_squared, y_squared, P);

    mpz_t exp, y_candidate;
    mpz_inits(exp, y_candidate, NULL);
    mpz_add_ui(exp, P, 1);
    mpz_fdiv_q_ui(exp, exp, 4);
    mpz_powm(y_candidate, y_squared, exp, P);

    mpz_t verify;
    mpz_init(verify);
    mpz_powm_ui(verify, y_candidate, 2, P);

    bool is_valid = (mpz_cmp(verify, y_squared) == 0);

    if (is_valid) {
        int y_parity = mpz_tstbit(y_candidate, 0);
        int expected_parity = (compressed[0] == 0x03) ? 1 : 0;

        if (y_parity != expected_parity) {
            mpz_sub(y_candidate, P, y_candidate);
        }

        mpz_set(point->y, y_candidate);
        point->infinity = 0;
    }

    mpz_clears(x3, y_squared, seven, exp, y_candidate, verify, NULL);
    return is_valid;
}

uint256_t uint256_to_mpz(mpz_t private_key, uint256_t value) {

    const int limb_bits = sizeof(mp_limb_t) * 8;
    const int limb_bytes = sizeof(mp_limb_t);

    std::vector<uint8_t> bytes;
    export_bits(value, std::back_inserter(bytes), 8);

    size_t num_limbs = (bytes.size() + limb_bytes - 1) / limb_bytes;

    mpz_init2(private_key, 256);

    mp_limb_t* limbs = mpz_limbs_write(private_key, num_limbs);

    mp_limb_t carry = 0;
    size_t byte_index = 0;

    for (size_t i = 0; i < num_limbs; ++i) {

        mp_limb_t temp = carry;

        for (size_t j = 0; j < limb_bytes && byte_index < bytes.size(); ++j, ++byte_index) {
            temp = (temp << 8) | bytes[byte_index];
        }

        limbs[i] = temp;
        carry = temp >> (limb_bits - 8);
    }

    if (carry) {
        limbs[num_limbs - 1] = carry;
        num_limbs++;
    }

    private_key->_mp_size = num_limbs;
};

uint256_t mpz_to_uint256(const mpz_t value) {

    if (mpz_cmp_ui(value, 0) == 0) {
        return uint256_t(0);
    }

    const int limb_bits = sizeof(mp_limb_t) * 8;
    const int limb_bytes = sizeof(mp_limb_t);

    size_t num_limbs = mpz_size(value);
    const mp_limb_t* limbs = mpz_limbs_read(value);

    std::vector<uint8_t> bytes;
    bytes.reserve(num_limbs * limb_bytes);

    for (size_t i = 0; i < num_limbs; ++i) {
        mp_limb_t limb = limbs[i];
        for (size_t j = 0; j < limb_bytes; ++j) {
            bytes.push_back(static_cast<uint8_t>(limb & 0xFF));
            limb >>= 8;
        }
    }

    while (!bytes.empty() && bytes.back() == 0) {
        bytes.pop_back();
    }

    if (bytes.empty()) {
        return uint256_t(0);
    }

    uint256_t result = 0;
    for (int i = static_cast<int>(bytes.size()) - 1; i >= 0 && (bytes.size() - 1 - i) < 32; --i) {
        result = (result << 8) | bytes[i];
    }

    return result;
}

bool is_distinguished_point(const ECPoint& point, int dp_bits = 20) {

    for (int i = 0; i < dp_bits; ++i) {
        if (mpz_tstbit(point.x, i)) {
            return false;
        }
    }
    return true;
}

std::array<uint8_t, 33> point_to_compressed_array(const ECPoint& point) {
    std::array<uint8_t, 33> result;
    get_compressed_public_key(result.data(), &point);
    return result;
}

struct PrecomputedJump {
    ECPoint point;
    uint256_t c, d;

    PrecomputedJump() {
        point_init(&point);
        c = 0;
        d = 0;
    }

    ~PrecomputedJump() {
        point_clear(&point);
    }

    PrecomputedJump(const PrecomputedJump& other) {
        point_init(&point);
        mpz_set(point.x, other.point.x);
        mpz_set(point.y, other.point.y);
        point.infinity = other.point.infinity;
        c = other.c;
        d = other.d;
    }

    PrecomputedJump& operator=(const PrecomputedJump& other) {
        if (this != &other) {
            mpz_set(point.x, other.point.x);
            mpz_set(point.y, other.point.y);
            point.infinity = other.point.infinity;
            c = other.c;
            d = other.d;
        }
        return *this;
    }
};

class OptimizedRNG {
private:
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint64_t> dist;

    public: OptimizedRNG() : gen(std::random_device{}()), dist(0, UINT64_MAX) {}

    uint256_t generate_in_range(const uint256_t& min_val, const uint256_t& max_val) {

        uint256_t range = max_val - min_val + 1;
        uint256_t max_valid = (uint256_t(1) << 256) - ((uint256_t(1) << 256) % range);
        uint256_t result;

        do {
            result = 0;
            for (int i = 0; i < 4; ++i) {
                result = (result << 64) | dist(gen);
            }
        } while (result >= max_valid);

        return min_val + (result % range);
    }

    int get_partition(const ECPoint& point, int num_partitions = 32) {

        uint64_t hash = 0;
        for (int i = 0; i < 8; ++i) {
            if (mpz_tstbit(point.x, i)) {
                hash |= (1ULL << i);
            }
        }
        return hash % num_partitions;
    }
};

std::vector<PrecomputedJump> precompute_jumps(int num_jumps, const uint256_t& range_size) {
    std::vector<PrecomputedJump> jumps(num_jumps);
    OptimizedRNG rng;

    std::cout << "Precomputando " << num_jumps << " saltos..." << std::flush;

    for (int i = 0; i < num_jumps; ++i) {

        jumps[i].c = rng.generate_in_range(1, range_size / 100);
        jumps[i].d = rng.generate_in_range(1, range_size / 100);

        mpz_t c_mpz, d_mpz;
        mpz_inits(c_mpz, d_mpz, NULL);
        uint256_to_mpz(c_mpz, jumps[i].c);
        uint256_to_mpz(d_mpz, jumps[i].d);

        ECPoint temp1, temp2;
        point_init(&temp1);
        point_init(&temp2);
        scalar_mult(&temp1, c_mpz, &G, P);
        scalar_mult(&temp2, d_mpz, &H, P);
        point_add(&jumps[i].point, &temp1, &temp2, P);
        point_clear(&temp1);
        point_clear(&temp2);
        mpz_clears(c_mpz, d_mpz, NULL);

        if ((i + 1) % (num_jumps / 10) == 0) {
            std::cout << "." << std::flush;
        }
    }

    std::cout << " concluido!" << std::endl;
    return jumps;
}

void iteration_function(ECPoint& R, uint256_t& a, uint256_t& b, OptimizedRNG& rng, const std::vector<PrecomputedJump>& jumps, const uint256_t& range_size) {

    int partition = rng.get_partition(R, jumps.size());
    const PrecomputedJump& jump = jumps[partition];

    point_add(&R, &R, &jump.point, P);
    a = (a + jump.c) % range_size;
    b = (b + jump.d) % range_size;
}

uint256_t solve_dlog_collision(const DPEntry& dp1, const DPEntry& dp2) {

    mpz_t n_mpz, da_mpz, db_mpz, db_inv, k_mpz;
    mpz_inits(n_mpz, da_mpz, db_mpz, db_inv, k_mpz, NULL);
    mpz_set(n_mpz, N);

    if (dp1.a >= dp2.a) {
        uint256_to_mpz(da_mpz, dp1.a - dp2.a);
    } else {
        uint256_t temp = mpz_to_uint256(N) - (dp2.a - dp1.a);
        uint256_to_mpz(da_mpz, temp);
    }

    if (dp2.b >= dp1.b) {
        uint256_to_mpz(db_mpz, dp2.b - dp1.b);
    } else {
        uint256_t temp = mpz_to_uint256(N) - (dp1.b - dp2.b);
        uint256_to_mpz(db_mpz, temp);
    }

    if (mpz_cmp_ui(db_mpz, 0) == 0) {
        mpz_clears(n_mpz, da_mpz, db_mpz, db_inv, k_mpz, NULL);
        return uint256_t(0);
    }

    uint256_t result = 0;
    if (mpz_invert(db_inv, db_mpz, n_mpz)) {
        mpz_mul(k_mpz, da_mpz, db_inv);
        mpz_mod(k_mpz, k_mpz, n_mpz);
        result = mpz_to_uint256(k_mpz);
    }

    mpz_clears(n_mpz, da_mpz, db_mpz, db_inv, k_mpz, NULL);
    return result;
}

uint256_t prho(const std::string& target_pubkey_hex, int key_range, int num_threads = 0) {
    auto start_time = std::chrono::system_clock::now();

    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    std::cout << "Usando " << num_threads << " threads" << std::endl;
    std::cout << "Range: " << key_range << " bits" << std::endl;

    auto hex_to_bytes = [](const std::string& hex) -> std::vector<unsigned char> {
        std::vector<unsigned char> bytes;
        for (size_t i = 0; i < hex.length(); i += 2) {
            unsigned char byte = (unsigned char) std::stoi(hex.substr(i, 2), nullptr, 16);
            bytes.push_back(byte);
        }
        return bytes;
    };

    auto target_bytes = hex_to_bytes(target_pubkey_hex);

    if (target_bytes.size() != 33) {
        std::cerr << "Erro: chave publica deve ter 33 bytes (formato comprimido)" << std::endl;
        return uint256_t(0);
    }

    std::array<uint8_t, 33> compressed_target;
    std::copy(target_bytes.begin(), target_bytes.end(), compressed_target.begin());

    if (!decompress_point(&H, compressed_target)) {
        std::cerr << "Erro: falha ao descomprimir chave publica alvo" << std::endl;
        return uint256_t(0);
    }

    uint256_t min_scalar = (uint256_t(1) << (key_range - 1));  
    uint256_t max_scalar = (uint256_t(1) << key_range) - 1;
    uint256_t range_size = max_scalar - min_scalar + 1;

    auto jumps = precompute_jumps(32, range_size);

    ShardedDPTable dp_table;

    std::atomic<bool> found{false};
    std::atomic<uint64_t> total_iterations{0};
    uint256_t result = 0;

    std::thread logger_thread([&]() {
        while (!found.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            uint64_t its = total_iterations.load();
            size_t dps = dp_table.total_size();
            std::cout << "\r Iteracoes: " << its << " | DPs: " << dps << std::flush;
        }
    });

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        OptimizedRNG rng;
        ECPoint current_point;
        point_init(&current_point);
        uint256_t a, b;
        uint64_t local_iterations = 0;
        try {
            while (!found.load()) {
                a = rng.generate_in_range(min_scalar, max_scalar);
                b = rng.generate_in_range(min_scalar, max_scalar);

                mpz_t a_mpz, b_mpz;
                mpz_inits(a_mpz, b_mpz, NULL);
                uint256_to_mpz(a_mpz, a);
                uint256_to_mpz(b_mpz, b);
                ECPoint temp1, temp2;
                point_init(&temp1);
                point_init(&temp2);
                scalar_mult(&temp1, a_mpz, &G, P);
                scalar_mult(&temp2, b_mpz, &H, P);
                point_add(&current_point, &temp1, &temp2, P);
                point_clear(&temp1);
                point_clear(&temp2);
                mpz_clears(a_mpz, b_mpz, NULL);

                const int max_walk = 2000000;
                int walk_length = 0;

                while (walk_length < max_walk && !found.load()) {
                    iteration_function(current_point, a, b, rng, jumps, range_size);
                    walk_length++;
                    local_iterations++;

                    if (local_iterations % 1000 == 0) {
                        total_iterations.fetch_add(1000);
                        local_iterations = 0;
                    }

                    if (is_distinguished_point(current_point)) {
                        auto compressed = point_to_compressed_array(current_point);
                        DPEntry new_entry(compressed, a, b);
                        DPEntry existing_entry;

                        if (dp_table.insert_or_find_collision(compressed, new_entry, existing_entry)) {
                            uint256_t private_key = solve_dlog_collision(existing_entry, new_entry);

                            if (private_key != 0 && private_key >= min_scalar && private_key <= max_scalar) {
                                #pragma omp critical
                                {
                                    if (!found.load()) {
                                        result = private_key;
                                        found.store(true);
                                        std::cout << "\n Chave privada encontrada: " << private_key << std::endl;
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "\nErro na thread: " << e.what() << std::endl;
        }

        total_iterations.fetch_add(local_iterations);
        point_clear(&current_point);
    }

    logger_thread.join();

    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTempo total: " << duration.count() << " segundos" << std::endl;
    std::cout << "Total de iteracoes: " << total_iterations.load() << std::endl;
    std::cout << "Pontos distinguidos: " << dp_table.total_size() << std::endl;

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Uso: " << argv[0] << " <Compressed Public Key> <Key Range> [num_threads]" << std::endl;
        return 1;
    }

    init_secp256k1();

    std::string pub_key_hex(argv[1]);
    int key_range = std::stoi(argv[2]);
    int num_threads = (argc == 4) ? std::stoi(argv[3]) : 0;

    uint256_t found_key = prho(pub_key_hex, key_range, num_threads);

    if (found_key != 0) {
        std::cout << "Sucesso! Chave privada: " << std::hex << found_key << std::dec << std::endl;
    } else {
        std::cout << "Chave nao encontrada." << std::endl;
    }

    mpz_clears(P, N, GX, GY, NULL);
    point_clear(&G);
    point_clear(&H);

    return 0;
}
