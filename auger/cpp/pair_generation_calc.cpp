// pair_generation_calc.cpp
// =============================================================================
// Standalone C++ calculator for Auger exact-kpoint and pair-table generation.
//
// Input is the binary file produced by auger.cpp.prepare_pair_generation_input.
// The numerical loops mirror auger.pairs.PairGenerator:
//   - exact-kpoint table generation for Brute_Force and Max_Heap
//   - nearest-kpoint pair generation for Brute_Force and Max_Heap
//   - exact-kpoint pair-table construction from resolved exact-kpoint rows
//
// Build:
//   g++ -O3 -std=c++17 -o pair_generation_calc pair_generation_calc.cpp
//
// Usage:
//   ./pair_generation_calc pair_generation_input.bin output_base.csv [start_chunk_index]
//
// Output is always chunked as output_base_1.csv, output_base_2.csv, ...
// with 1,000,000 rows per chunk.
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double K_B_EV = 8.617333262145e-5;
constexpr std::size_t CHUNK_SIZE = 1000000ULL;

enum class Task : int32_t { ExactKpoints = 0, Pairs = 1 };
enum class AugerType : int32_t { EEH = 0, EHH = 1 };
enum class Approach : int32_t { NearestKpoint = 0, ExactKpoint = 1 };
enum class SearchMode : int32_t { BruteForce = 0, MaxHeap = 1 };

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct State {
    int32_t band_index = 0;
    int32_t k_index = 0;
    double energy = 0.0;
    Vec3 k;
    double kw = 0.0;
    double P = 0.0;
};

struct ExactRow {
    std::string partial_pair_id;
    int32_t E_index[4]{-1, -1, -1, -1};
    int32_t k_index[4]{-1, -1, -1, -1};
    int32_t wc_index[4]{-1, -1, -1, -1};
    int32_t nscf_index[4]{-1, -1, -1, -1};
    double E[4]{0.0, 0.0, 0.0, 0.0};
    Vec3 k[4];
    double kw[4]{0.0, 0.0, 0.0, 0.0};
    Vec3 mapped;
};

struct ExactOutputRow {
    std::string partial_pair_id;
    double prob = 0.0;
    int32_t E_index[4]{-1, -1, -1, -1};
    int32_t k_index[4]{-1, -1, -1, -1};
    double E[4]{0.0, 0.0, 0.0, 0.0};
    Vec3 k[4];
    double kw[4]{0.0, 0.0, 0.0, 0.0};
    Vec3 k_frac[4];
    Vec3 target_cart;
    Vec3 target_frac;
    Vec3 target_frac_mapped;
    Vec3 target_cart_mapped;
};

struct PairOutputRow {
    std::string pair_id;
    double probability = 0.0;
    int32_t E_index[4]{-1, -1, -1, -1};
    int32_t k_index[4]{-1, -1, -1, -1};
    double E[4]{0.0, 0.0, 0.0, 0.0};
    Vec3 k[4];
    double kw[4]{0.0, 0.0, 0.0, 0.0};
    Vec3 mapped;
    int32_t mapped_slot = 0;  // 2 for eeh, 4 for ehh
    bool has_exact_indices = false;
    int32_t wc_index[4]{-1, -1, -1, -1};
    int32_t nscf_index[4]{-1, -1, -1, -1};
};

struct NearestResult {
    int32_t band_index = 0;
    int32_t k_index = 0;
    double energy = 0.0;
    Vec3 nearest_k;
    double nearest_kw = 0.0;
    double Px = 0.0;
    Vec3 target_cart;
    Vec3 target_frac;
    Vec3 target_frac_mapped;
    Vec3 target_cart_mapped;
};

struct InputData {
    Task task = Task::Pairs;
    AugerType auger_type = AugerType::EEH;
    Approach approach = Approach::NearestKpoint;
    SearchMode search_mode = SearchMode::BruteForce;
    int64_t desired_total = -1;   // total requested, including previous files
    int64_t previous_count = 0;   // rows already present in continuation files
    int32_t multiplier = 1;
    int32_t firstCB_index = 0;
    int32_t num_bands = 0;
    int64_t num_kpoints = 0;
    double T = 300.0;
    double Efn = 0.0;
    double Efp = 0.0;
    double reciprocal_lattice[9]{};
    double reciprocal_lattice_inv[9]{};
    std::vector<Vec3> kpoints;
    std::vector<double> kpoint_weights;
    std::vector<double> energies;  // row-major [band][kpoint]
    std::vector<State> E1;
    std::vector<State> E2;
    std::vector<State> E3;
    std::vector<State> E4;
    std::unordered_set<std::string> skip_partial_ids;
    std::unordered_set<std::string> skip_pair_ids;
    std::vector<ExactRow> exact_rows;
};

struct HeapNode {
    double probability = 0.0;
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    NearestResult nearest;
    bool operator<(const HeapNode& other) const {
        return probability < other.probability;
    }
};

struct ExactHeapNode {
    double probability = 0.0;
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    bool operator<(const ExactHeapNode& other) const {
        return probability < other.probability;
    }
};

struct TripleIndex {
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    bool operator==(const TripleIndex& other) const {
        return i == other.i && j == other.j && k == other.k;
    }
};

struct TripleHash {
    std::size_t operator()(const TripleIndex& t) const noexcept {
        std::size_t h = std::hash<std::size_t>{}(t.i);
        h ^= std::hash<std::size_t>{}(t.j) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<std::size_t>{}(t.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double norm2(Vec3 a) { return a.x * a.x + a.y * a.y + a.z * a.z; }

double fermi_dirac(double E, double Ef, double T) {
    const double x = (Ef - E) / (K_B_EV * T);
    if (x > 50.0) return 1.0;
    if (x < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

void invert3x3(const double m[9], double inv[9]) {
    const double det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::abs(det) < 1e-30) {
        throw std::runtime_error("reciprocal_lattice is singular");
    }
    const double id = 1.0 / det;
    inv[0] =  (m[4] * m[8] - m[5] * m[7]) * id;
    inv[1] = -(m[1] * m[8] - m[2] * m[7]) * id;
    inv[2] =  (m[1] * m[5] - m[2] * m[4]) * id;
    inv[3] = -(m[3] * m[8] - m[5] * m[6]) * id;
    inv[4] =  (m[0] * m[8] - m[2] * m[6]) * id;
    inv[5] = -(m[0] * m[5] - m[2] * m[3]) * id;
    inv[6] =  (m[3] * m[7] - m[4] * m[6]) * id;
    inv[7] = -(m[0] * m[7] - m[1] * m[6]) * id;
    inv[8] =  (m[0] * m[4] - m[1] * m[3]) * id;
}

Vec3 row_times_matrix(Vec3 v, const double m[9]) {
    return {
        v.x * m[0] + v.y * m[3] + v.z * m[6],
        v.x * m[1] + v.y * m[4] + v.z * m[7],
        v.x * m[2] + v.y * m[5] + v.z * m[8],
    };
}

Vec3 to_fractional(Vec3 cart, const InputData& d) {
    return row_times_matrix(cart, d.reciprocal_lattice_inv);
}

Vec3 to_cartesian(Vec3 frac, const InputData& d) {
    return row_times_matrix(frac, d.reciprocal_lattice);
}

Vec3 fold_vasp_centered(Vec3 k) {
    auto fold_one = [](double x) {
        double y = x - std::floor(x);
        if (y > 0.5) y -= 1.0;
        if (std::abs(y + 0.5) < 1e-10) y = 0.5;
        return y;
    };
    return {fold_one(k.x), fold_one(k.y), fold_one(k.z)};
}

std::string format_double(double x) {
    std::ostringstream oss;
    oss << std::setprecision(17) << x;
    return oss.str();
}

std::string format_vec(Vec3 v) {
    return "[" + format_double(v.x) + ", " + format_double(v.y) + ", " + format_double(v.z) + "]";
}

std::string csv_escape(const std::string& s) {
    bool quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            quote = true;
            break;
        }
    }
    if (!quote) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("unexpected end of binary input");
    return value;
}

std::string read_string(std::ifstream& in) {
    const int32_t n = read_pod<int32_t>(in);
    if (n < 0) throw std::runtime_error("negative string length in binary input");
    std::string s(static_cast<std::size_t>(n), '\0');
    if (n > 0) {
        in.read(&s[0], n);
        if (!in) throw std::runtime_error("unexpected end while reading string");
    }
    return s;
}

Vec3 read_vec3(std::ifstream& in) {
    Vec3 v;
    v.x = read_pod<double>(in);
    v.y = read_pod<double>(in);
    v.z = read_pod<double>(in);
    return v;
}

State read_state(std::ifstream& in) {
    State s;
    s.band_index = read_pod<int32_t>(in);
    s.k_index = read_pod<int32_t>(in);
    s.energy = read_pod<double>(in);
    s.k = read_vec3(in);
    s.kw = read_pod<double>(in);
    s.P = read_pod<double>(in);
    return s;
}

std::vector<State> read_states(std::ifstream& in) {
    const int64_t n = read_pod<int64_t>(in);
    if (n < 0) throw std::runtime_error("negative state count");
    std::vector<State> states;
    states.reserve(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        states.push_back(read_state(in));
    }
    return states;
}

std::unordered_set<std::string> read_string_set(std::ifstream& in) {
    const int64_t n = read_pod<int64_t>(in);
    if (n < 0) throw std::runtime_error("negative string set count");
    std::unordered_set<std::string> values;
    values.reserve(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        values.insert(read_string(in));
    }
    return values;
}

ExactRow read_exact_row(std::ifstream& in) {
    ExactRow r;
    r.partial_pair_id = read_string(in);
    for (int i = 0; i < 4; ++i) r.E_index[i] = read_pod<int32_t>(in);
    for (int i = 0; i < 4; ++i) r.k_index[i] = read_pod<int32_t>(in);
    for (int i = 0; i < 4; ++i) r.wc_index[i] = read_pod<int32_t>(in);
    for (int i = 0; i < 4; ++i) r.nscf_index[i] = read_pod<int32_t>(in);
    for (int i = 0; i < 4; ++i) r.E[i] = read_pod<double>(in);
    for (int i = 0; i < 4; ++i) r.k[i] = read_vec3(in);
    for (int i = 0; i < 4; ++i) r.kw[i] = read_pod<double>(in);
    r.mapped = read_vec3(in);
    return r;
}

InputData read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input binary: " + path);

    char magic[8];
    in.read(magic, 8);
    if (!in || std::string(magic, 8) != "AUGPAIR1") {
        throw std::runtime_error("invalid pair-generation binary magic");
    }
    const int32_t version = read_pod<int32_t>(in);
    if (version != 1) {
        throw std::runtime_error("unsupported pair-generation binary version");
    }

    InputData d;
    d.task = static_cast<Task>(read_pod<int32_t>(in));
    d.auger_type = static_cast<AugerType>(read_pod<int32_t>(in));
    d.approach = static_cast<Approach>(read_pod<int32_t>(in));
    d.search_mode = static_cast<SearchMode>(read_pod<int32_t>(in));
    d.desired_total = read_pod<int64_t>(in);
    d.previous_count = read_pod<int64_t>(in);
    d.multiplier = read_pod<int32_t>(in);
    d.firstCB_index = read_pod<int32_t>(in);
    d.num_bands = read_pod<int32_t>(in);
    d.num_kpoints = read_pod<int64_t>(in);
    d.T = read_pod<double>(in);
    d.Efn = read_pod<double>(in);
    d.Efp = read_pod<double>(in);
    for (double& x : d.reciprocal_lattice) x = read_pod<double>(in);
    invert3x3(d.reciprocal_lattice, d.reciprocal_lattice_inv);

    const int64_t nk = read_pod<int64_t>(in);
    if (nk < 0) throw std::runtime_error("negative k-point count");
    d.kpoints.reserve(static_cast<std::size_t>(nk));
    d.kpoint_weights.reserve(static_cast<std::size_t>(nk));
    for (int64_t i = 0; i < nk; ++i) {
        d.kpoints.push_back(read_vec3(in));
        d.kpoint_weights.push_back(read_pod<double>(in));
    }

    const int32_t nb = read_pod<int32_t>(in);
    const int64_t nek = read_pod<int64_t>(in);
    if (nb < 0 || nek < 0) throw std::runtime_error("negative energy dimensions");
    d.energies.resize(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nek));
    for (double& e : d.energies) e = read_pod<double>(in);
    if (d.num_bands == 0) d.num_bands = nb;
    if (d.num_kpoints == 0) d.num_kpoints = nek;

    d.E1 = read_states(in);
    d.E2 = read_states(in);
    d.E3 = read_states(in);
    d.E4 = read_states(in);
    d.skip_partial_ids = read_string_set(in);
    d.skip_pair_ids = read_string_set(in);

    const int64_t nr = read_pod<int64_t>(in);
    if (nr < 0) throw std::runtime_error("negative exact-row count");
    d.exact_rows.reserve(static_cast<std::size_t>(nr));
    for (int64_t i = 0; i < nr; ++i) {
        d.exact_rows.push_back(read_exact_row(in));
    }
    return d;
}

double energy_at(const InputData& d, int32_t band, int32_t k) {
    return d.energies[static_cast<std::size_t>(band) * static_cast<std::size_t>(d.num_kpoints)
                      + static_cast<std::size_t>(k)];
}

int64_t new_target(const InputData& d) {
    if (d.desired_total < 0) return -1;
    return std::max<int64_t>(0, d.desired_total - d.previous_count);
}

NearestResult nearest_kpoint(const InputData& d, Vec3 k_diff, double E_diff) {
    NearestResult r;
    r.target_cart = k_diff;
    r.target_frac = to_fractional(k_diff, d);
    r.target_frac_mapped = fold_vasp_centered(r.target_frac);
    r.target_cart_mapped = to_cartesian(r.target_frac_mapped, d);

    double best_dist = std::numeric_limits<double>::infinity();
    int32_t best_k = 0;
    for (std::size_t ki = 0; ki < d.kpoints.size(); ++ki) {
        const double dist = norm2(sub(d.kpoints[ki], r.target_cart_mapped));
        if (dist < best_dist) {
            best_dist = dist;
            best_k = static_cast<int32_t>(ki);
        }
    }
    r.k_index = best_k;
    r.nearest_k = d.kpoints[static_cast<std::size_t>(best_k)];
    r.nearest_kw = d.kpoint_weights[static_cast<std::size_t>(best_k)];

    const int32_t lo = d.auger_type == AugerType::EEH ? d.firstCB_index : 0;
    const int32_t hi = d.auger_type == AugerType::EEH ? d.num_bands : d.firstCB_index;
    double best_ediff = std::numeric_limits<double>::infinity();
    for (int32_t bi = lo; bi < hi; ++bi) {
        const double e = energy_at(d, bi, best_k);
        const double diff = std::abs(e - E_diff);
        if (diff < best_ediff) {
            best_ediff = diff;
            r.band_index = bi;
            r.energy = e;
        }
    }
    if (d.auger_type == AugerType::EEH) {
        r.Px = 1.0 - fermi_dirac(r.energy, d.Efn, d.T);
    } else {
        r.Px = fermi_dirac(r.energy, d.Efp, d.T);
    }
    return r;
}

ExactOutputRow make_exact_eeh(const InputData& d, const State& e1, const State& e3, const State& e4) {
    ExactOutputRow row;
    row.partial_pair_id =
        std::to_string(e1.band_index) + "-X-" + std::to_string(e3.band_index) + "-" +
        std::to_string(e4.band_index) + "-" + std::to_string(e1.k_index) + "-X-" +
        std::to_string(e3.k_index) + "-" + std::to_string(e4.k_index);
    row.prob = e1.P * e3.P * e4.P;
    row.E_index[0] = e1.band_index;
    row.E_index[2] = e3.band_index;
    row.E_index[3] = e4.band_index;
    row.k_index[0] = e1.k_index;
    row.k_index[2] = e3.k_index;
    row.k_index[3] = e4.k_index;
    row.E[0] = e1.energy;
    row.E[2] = e3.energy;
    row.E[3] = e4.energy;
    row.k[0] = e1.k;
    row.k[2] = e3.k;
    row.k[3] = e4.k;
    row.kw[0] = e1.kw;
    row.kw[2] = e3.kw;
    row.kw[3] = e4.kw;
    row.k_frac[0] = to_fractional(e1.k, d);
    row.k_frac[2] = to_fractional(e3.k, d);
    row.k_frac[3] = to_fractional(e4.k, d);
    row.target_cart = add(sub(e3.k, e4.k), e1.k);
    row.target_frac = to_fractional(row.target_cart, d);
    row.target_frac_mapped = fold_vasp_centered(row.target_frac);
    row.target_cart_mapped = to_cartesian(row.target_frac_mapped, d);
    return row;
}

ExactOutputRow make_exact_ehh(const InputData& d, const State& e1, const State& e2, const State& e3) {
    ExactOutputRow row;
    row.partial_pair_id =
        std::to_string(e1.band_index) + "-" + std::to_string(e2.band_index) + "-" +
        std::to_string(e3.band_index) + "-X-" + std::to_string(e1.k_index) + "-" +
        std::to_string(e2.k_index) + "-" + std::to_string(e3.k_index) + "-X";
    row.prob = e1.P * e2.P * e3.P;
    row.E_index[0] = e1.band_index;
    row.E_index[1] = e2.band_index;
    row.E_index[2] = e3.band_index;
    row.k_index[0] = e1.k_index;
    row.k_index[1] = e2.k_index;
    row.k_index[2] = e3.k_index;
    row.E[0] = e1.energy;
    row.E[1] = e2.energy;
    row.E[2] = e3.energy;
    row.k[0] = e1.k;
    row.k[1] = e2.k;
    row.k[2] = e3.k;
    row.kw[0] = e1.kw;
    row.kw[1] = e2.kw;
    row.kw[2] = e3.kw;
    row.k_frac[0] = to_fractional(e1.k, d);
    row.k_frac[1] = to_fractional(e2.k, d);
    row.k_frac[2] = to_fractional(e3.k, d);
    row.target_cart = add(sub(e3.k, e1.k), e2.k);
    row.target_frac = to_fractional(row.target_cart, d);
    row.target_frac_mapped = fold_vasp_centered(row.target_frac);
    row.target_cart_mapped = to_cartesian(row.target_frac_mapped, d);
    return row;
}

PairOutputRow make_nearest_pair_eeh(
    const State& e1,
    const NearestResult& e2,
    const State& e3,
    const State& e4
) {
    PairOutputRow row;
    row.E_index[0] = e1.band_index;
    row.E_index[1] = e2.band_index;
    row.E_index[2] = e3.band_index;
    row.E_index[3] = e4.band_index;
    row.k_index[0] = e1.k_index;
    row.k_index[1] = e2.k_index;
    row.k_index[2] = e3.k_index;
    row.k_index[3] = e4.k_index;
    row.E[0] = e1.energy;
    row.E[1] = e2.energy;
    row.E[2] = e3.energy;
    row.E[3] = e4.energy;
    row.k[0] = e1.k;
    row.k[1] = e2.nearest_k;
    row.k[2] = e3.k;
    row.k[3] = e4.k;
    row.kw[0] = e1.kw;
    row.kw[1] = e2.nearest_kw;
    row.kw[2] = e3.kw;
    row.kw[3] = e4.kw;
    row.mapped = e2.target_cart_mapped;
    row.mapped_slot = 2;
    row.probability = e1.P * e2.Px * e3.P * e4.P;
    row.pair_id =
        std::to_string(row.E_index[0]) + "-" + std::to_string(row.E_index[1]) + "-" +
        std::to_string(row.E_index[2]) + "-" + std::to_string(row.E_index[3]) + "-" +
        std::to_string(row.k_index[0]) + "-" + std::to_string(row.k_index[1]) + "-" +
        std::to_string(row.k_index[2]) + "-" + std::to_string(row.k_index[3]);
    return row;
}

PairOutputRow make_nearest_pair_ehh(
    const State& e1,
    const State& e2,
    const State& e3,
    const NearestResult& e4
) {
    PairOutputRow row;
    row.E_index[0] = e1.band_index;
    row.E_index[1] = e2.band_index;
    row.E_index[2] = e3.band_index;
    row.E_index[3] = e4.band_index;
    row.k_index[0] = e1.k_index;
    row.k_index[1] = e2.k_index;
    row.k_index[2] = e3.k_index;
    row.k_index[3] = e4.k_index;
    row.E[0] = e1.energy;
    row.E[1] = e2.energy;
    row.E[2] = e3.energy;
    row.E[3] = e4.energy;
    row.k[0] = e1.k;
    row.k[1] = e2.k;
    row.k[2] = e3.k;
    row.k[3] = e4.nearest_k;
    row.kw[0] = e1.kw;
    row.kw[1] = e2.kw;
    row.kw[2] = e3.kw;
    row.kw[3] = e4.nearest_kw;
    row.mapped = e4.target_cart_mapped;
    row.mapped_slot = 4;
    row.probability = e1.P * e2.P * e3.P * e4.Px;
    row.pair_id =
        std::to_string(row.E_index[0]) + "-" + std::to_string(row.E_index[1]) + "-" +
        std::to_string(row.E_index[2]) + "-" + std::to_string(row.E_index[3]) + "-" +
        std::to_string(row.k_index[0]) + "-" + std::to_string(row.k_index[1]) + "-" +
        std::to_string(row.k_index[2]) + "-" + std::to_string(row.k_index[3]);
    return row;
}

PairOutputRow make_exact_pair_eeh(const InputData& d, const ExactRow& r) {
    PairOutputRow row;
    for (int i = 0; i < 4; ++i) {
        row.E_index[i] = r.E_index[i];
        row.k_index[i] = r.k_index[i];
        row.E[i] = r.E[i];
        row.k[i] = r.k[i];
        row.kw[i] = r.kw[i];
        row.wc_index[i] = r.wc_index[i];
        row.nscf_index[i] = r.nscf_index[i];
    }
    const double P1 = fermi_dirac(row.E[0], d.Efn, d.T);
    const double P2 = 1.0 - fermi_dirac(row.E[1], d.Efn, d.T);
    const double P3 = fermi_dirac(row.E[2], d.Efn, d.T);
    const double P4 = 1.0 - fermi_dirac(row.E[3], d.Efp, d.T);
    row.probability = P1 * P2 * P3 * P4;
    row.mapped = r.mapped;
    row.mapped_slot = 2;
    row.has_exact_indices = true;
    row.pair_id =
        std::to_string(row.E_index[0]) + "-" + std::to_string(row.E_index[1]) + "-" +
        std::to_string(row.E_index[2]) + "-" + std::to_string(row.E_index[3]) +
        "-w" + std::to_string(row.wc_index[0]) + ":" + std::to_string(row.nscf_index[0]) +
        "-w" + std::to_string(row.wc_index[1]) + ":" + std::to_string(row.nscf_index[1]) +
        "-w" + std::to_string(row.wc_index[2]) + ":" + std::to_string(row.nscf_index[2]) +
        "-w" + std::to_string(row.wc_index[3]) + ":" + std::to_string(row.nscf_index[3]);
    return row;
}

PairOutputRow make_exact_pair_ehh(const InputData& d, const ExactRow& r) {
    PairOutputRow row;
    for (int i = 0; i < 4; ++i) {
        row.E_index[i] = r.E_index[i];
        row.k_index[i] = r.k_index[i];
        row.E[i] = r.E[i];
        row.k[i] = r.k[i];
        row.kw[i] = r.kw[i];
        row.wc_index[i] = r.wc_index[i];
        row.nscf_index[i] = r.nscf_index[i];
    }
    const double P1 = fermi_dirac(row.E[0], d.Efn, d.T);
    const double P2 = 1.0 - fermi_dirac(row.E[1], d.Efp, d.T);
    const double P3 = 1.0 - fermi_dirac(row.E[2], d.Efp, d.T);
    const double P4 = fermi_dirac(row.E[3], d.Efp, d.T);
    row.probability = P1 * P2 * P3 * P4;
    row.mapped = r.mapped;
    row.mapped_slot = 4;
    row.has_exact_indices = true;
    row.pair_id =
        std::to_string(row.E_index[0]) + "-" + std::to_string(row.E_index[1]) + "-" +
        std::to_string(row.E_index[2]) + "-" + std::to_string(row.E_index[3]) +
        "-w" + std::to_string(row.wc_index[0]) + ":" + std::to_string(row.nscf_index[0]) +
        "-w" + std::to_string(row.wc_index[1]) + ":" + std::to_string(row.nscf_index[1]) +
        "-w" + std::to_string(row.wc_index[2]) + ":" + std::to_string(row.nscf_index[2]) +
        "-w" + std::to_string(row.wc_index[3]) + ":" + std::to_string(row.nscf_index[3]);
    return row;
}

void maybe_add_pair(
    std::vector<PairOutputRow>& rows,
    std::unordered_set<std::string>& seen,
    PairOutputRow row,
    int64_t target_new
) {
    if (target_new >= 0 && static_cast<int64_t>(rows.size()) >= target_new) return;
    if (seen.find(row.pair_id) != seen.end()) return;
    seen.insert(row.pair_id);
    rows.push_back(std::move(row));
}

std::vector<ExactOutputRow> generate_exact_brute(const InputData& d) {
    std::vector<ExactOutputRow> rows;
    const int64_t target = new_target(d);
    const bool finite = target >= 0;
    if (finite && target == 0) return rows;

    if (d.auger_type == AugerType::EEH) {
        for (const auto& e1 : d.E1) {
            for (const auto& e3 : d.E3) {
                for (const auto& e4 : d.E4) {
                    auto row = make_exact_eeh(d, e1, e3, e4);
                    if (d.skip_partial_ids.find(row.partial_pair_id) != d.skip_partial_ids.end()) continue;
                    rows.push_back(std::move(row));
                }
            }
        }
    } else {
        for (const auto& e1 : d.E1) {
            for (const auto& e2 : d.E2) {
                for (const auto& e3 : d.E3) {
                    auto row = make_exact_ehh(d, e1, e2, e3);
                    if (d.skip_partial_ids.find(row.partial_pair_id) != d.skip_partial_ids.end()) continue;
                    rows.push_back(std::move(row));
                }
            }
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.prob > b.prob;
    });
    if (finite && static_cast<int64_t>(rows.size()) > target) {
        rows.resize(static_cast<std::size_t>(target));
    }
    return rows;
}

std::vector<ExactOutputRow> generate_exact_heap(const InputData& d) {
    std::vector<ExactOutputRow> rows;
    const int64_t target = new_target(d);
    if (target == 0) return rows;

    const auto& A = d.E1;
    const auto& B = d.auger_type == AugerType::EEH ? d.E3 : d.E2;
    const auto& C = d.auger_type == AugerType::EEH ? d.E4 : d.E3;
    const std::size_t total = A.size() * B.size() * C.size();
    const std::size_t limit = target < 0 ? total : static_cast<std::size_t>(std::min<int64_t>(target, total));

    std::priority_queue<ExactHeapNode> heap;
    std::unordered_set<TripleIndex, TripleHash> visited;

    auto push = [&](std::size_t i, std::size_t j, std::size_t k) {
        if (i >= A.size() || j >= B.size() || k >= C.size()) return;
        TripleIndex key{i, j, k};
        if (visited.find(key) != visited.end()) return;
        visited.insert(key);
        heap.push({A[i].P * B[j].P * C[k].P, i, j, k});
    };

    push(0, 0, 0);
    while (!heap.empty() && rows.size() < limit) {
        const auto node = heap.top();
        heap.pop();
        ExactOutputRow row = d.auger_type == AugerType::EEH
            ? make_exact_eeh(d, A[node.i], B[node.j], C[node.k])
            : make_exact_ehh(d, A[node.i], B[node.j], C[node.k]);
        if (d.skip_partial_ids.find(row.partial_pair_id) == d.skip_partial_ids.end()) {
            rows.push_back(std::move(row));
        }
        push(node.i + 1, node.j, node.k);
        push(node.i, node.j + 1, node.k);
        push(node.i, node.j, node.k + 1);
    }
    return rows;
}

std::vector<PairOutputRow> generate_nearest_brute(const InputData& d) {
    std::vector<PairOutputRow> rows;
    std::unordered_set<std::string> seen = d.skip_pair_ids;
    const int64_t target = new_target(d);
    if (target == 0) return rows;

    if (d.auger_type == AugerType::EEH) {
        for (const auto& e1 : d.E1) {
            for (const auto& e3 : d.E3) {
                for (const auto& e4 : d.E4) {
                    const double E_diff = e3.energy - e4.energy + e1.energy;
                    const Vec3 k_diff = add(sub(e3.k, e4.k), e1.k);
                    auto res = nearest_kpoint(d, k_diff, E_diff);
                    maybe_add_pair(rows, seen, make_nearest_pair_eeh(e1, res, e3, e4), target);
                    if (target >= 0 && static_cast<int64_t>(rows.size()) >= target) goto done;
                }
            }
        }
    } else {
        for (const auto& e1 : d.E1) {
            for (const auto& e2 : d.E2) {
                for (const auto& e3 : d.E3) {
                    const double E_diff = e3.energy - e1.energy + e2.energy;
                    const Vec3 k_diff = add(sub(e3.k, e1.k), e2.k);
                    auto res = nearest_kpoint(d, k_diff, E_diff);
                    maybe_add_pair(rows, seen, make_nearest_pair_ehh(e1, e2, e3, res), target);
                    if (target >= 0 && static_cast<int64_t>(rows.size()) >= target) goto done;
                }
            }
        }
    }
done:
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.probability > b.probability;
    });
    return rows;
}

std::vector<PairOutputRow> generate_nearest_heap(const InputData& d) {
    std::vector<PairOutputRow> rows;
    std::unordered_set<std::string> seen = d.skip_pair_ids;
    const int64_t target_new = new_target(d);
    if (target_new == 0) return rows;

    const auto& A = d.E1;
    const auto& B = d.auger_type == AugerType::EEH ? d.E3 : d.E2;
    const auto& C = d.auger_type == AugerType::EEH ? d.E4 : d.E3;
    const std::size_t total = A.size() * B.size() * C.size();
    const std::size_t candidate_budget = target_new < 0
        ? total
        : static_cast<std::size_t>(std::min<int64_t>(
              d.desired_total * std::max<int32_t>(1, d.multiplier),
              static_cast<int64_t>(total)));

    std::priority_queue<HeapNode> heap;
    std::unordered_set<TripleIndex, TripleHash> visited;

    auto push = [&](std::size_t i, std::size_t j, std::size_t k) {
        if (i >= A.size() || j >= B.size() || k >= C.size()) return;
        TripleIndex key{i, j, k};
        if (visited.find(key) != visited.end()) return;
        visited.insert(key);
        double E_diff = 0.0;
        Vec3 k_diff;
        if (d.auger_type == AugerType::EEH) {
            E_diff = B[j].energy - C[k].energy + A[i].energy;
            k_diff = add(sub(B[j].k, C[k].k), A[i].k);
        } else {
            E_diff = C[k].energy - A[i].energy + B[j].energy;
            k_diff = add(sub(C[k].k, A[i].k), B[j].k);
        }
        auto nearest = nearest_kpoint(d, k_diff, E_diff);
        const double prob = A[i].P * B[j].P * C[k].P * nearest.Px;
        heap.push({prob, i, j, k, nearest});
    };

    push(0, 0, 0);
    std::size_t popped = 0;
    while (!heap.empty() && popped < candidate_budget &&
           (target_new < 0 || static_cast<int64_t>(rows.size()) < target_new)) {
        const auto node = heap.top();
        heap.pop();
        ++popped;
        PairOutputRow row = d.auger_type == AugerType::EEH
            ? make_nearest_pair_eeh(A[node.i], node.nearest, B[node.j], C[node.k])
            : make_nearest_pair_ehh(A[node.i], B[node.j], C[node.k], node.nearest);
        maybe_add_pair(rows, seen, std::move(row), target_new);
        push(node.i + 1, node.j, node.k);
        push(node.i, node.j + 1, node.k);
        push(node.i, node.j, node.k + 1);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.probability > b.probability;
    });
    return rows;
}

std::vector<PairOutputRow> generate_exact_pairs(const InputData& d) {
    std::vector<PairOutputRow> rows;
    std::unordered_set<std::string> seen = d.skip_pair_ids;
    const int64_t target_new = new_target(d);
    if (target_new == 0) return rows;

    std::vector<ExactRow> sorted_rows = d.exact_rows;
    std::sort(sorted_rows.begin(), sorted_rows.end(), [&](const auto& a, const auto& b) {
        auto pa = d.auger_type == AugerType::EEH ? make_exact_pair_eeh(d, a).probability
                                                 : make_exact_pair_ehh(d, a).probability;
        auto pb = d.auger_type == AugerType::EEH ? make_exact_pair_eeh(d, b).probability
                                                 : make_exact_pair_ehh(d, b).probability;
        return pa > pb;
    });

    for (const auto& r : sorted_rows) {
        PairOutputRow row = d.auger_type == AugerType::EEH
            ? make_exact_pair_eeh(d, r)
            : make_exact_pair_ehh(d, r);
        maybe_add_pair(rows, seen, std::move(row), target_new);
        if (target_new >= 0 && static_cast<int64_t>(rows.size()) >= target_new) break;
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.probability > b.probability;
    });
    return rows;
}

std::pair<std::string, std::string> split_stem_ext(const std::string& output_path) {
    const std::size_t dot = output_path.find_last_of('.');
    if (dot == std::string::npos) return {output_path, ".csv"};
    return {output_path.substr(0, dot), output_path.substr(dot)};
}

template <typename Writer>
void write_chunked_csv(
    std::size_t nrows,
    const std::string& output_path,
    const std::string& header,
    std::size_t start_chunk_index,
    Writer writer
) {
    if (nrows == 0) {
        std::cout << "  No new rows to write.\n";
        return;
    }
    const auto [stem, ext] = split_stem_ext(output_path);
    const std::size_t parts = (nrows + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (std::size_t part = 0; part < parts; ++part) {
        const std::size_t start = part * CHUNK_SIZE;
        const std::size_t end = std::min(start + CHUNK_SIZE, nrows);
        const std::string path = stem + "_" + std::to_string(start_chunk_index + part) + ext;
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write output CSV: " + path);
        out << header << "\n";
        for (std::size_t i = start; i < end; ++i) {
            writer(out, i);
        }
        std::cout << "  Wrote " << (end - start) << " row(s): " << path << "\n";
    }
}

void write_exact_rows(
    const std::vector<ExactOutputRow>& rows,
    const InputData& d,
    const std::string& output_path,
    std::size_t start_chunk_index
) {
    std::string header;
    if (d.auger_type == AugerType::EEH) {
        header = "partial_pair_id,P_134,E1_index,E3_index,E4_index,k1_index,k3_index,k4_index,"
                 "E1,E3,E4,k1,k3,k4,kw1,kw3,kw4,k1_frac,k3_frac,k4_frac,"
                 "k2_target_cart,k2_target_frac,k2_target_frac_mapped,k2_target_cart_mapped";
    } else {
        header = "partial_pair_id,P_123,E1_index,E2_index,E3_index,k1_index,k2_index,k3_index,"
                 "E1,E2,E3,k1,k2,k3,kw1,kw2,kw3,k1_frac,k2_frac,k3_frac,"
                 "k4_target_cart,k4_target_frac,k4_target_frac_mapped,k4_target_cart_mapped";
    }

    write_chunked_csv(rows.size(), output_path, header, start_chunk_index, [&](std::ofstream& out, std::size_t i) {
        const auto& r = rows[i];
        if (d.auger_type == AugerType::EEH) {
            out << csv_escape(r.partial_pair_id) << "," << format_double(r.prob) << ","
                << r.E_index[0] << "," << r.E_index[2] << "," << r.E_index[3] << ","
                << r.k_index[0] << "," << r.k_index[2] << "," << r.k_index[3] << ","
                << format_double(r.E[0]) << "," << format_double(r.E[2]) << "," << format_double(r.E[3]) << ","
                << csv_escape(format_vec(r.k[0])) << "," << csv_escape(format_vec(r.k[2])) << "," << csv_escape(format_vec(r.k[3])) << ","
                << format_double(r.kw[0]) << "," << format_double(r.kw[2]) << "," << format_double(r.kw[3]) << ","
                << csv_escape(format_vec(r.k_frac[0])) << "," << csv_escape(format_vec(r.k_frac[2])) << "," << csv_escape(format_vec(r.k_frac[3])) << ","
                << csv_escape(format_vec(r.target_cart)) << "," << csv_escape(format_vec(r.target_frac)) << ","
                << csv_escape(format_vec(r.target_frac_mapped)) << "," << csv_escape(format_vec(r.target_cart_mapped)) << "\n";
        } else {
            out << csv_escape(r.partial_pair_id) << "," << format_double(r.prob) << ","
                << r.E_index[0] << "," << r.E_index[1] << "," << r.E_index[2] << ","
                << r.k_index[0] << "," << r.k_index[1] << "," << r.k_index[2] << ","
                << format_double(r.E[0]) << "," << format_double(r.E[1]) << "," << format_double(r.E[2]) << ","
                << csv_escape(format_vec(r.k[0])) << "," << csv_escape(format_vec(r.k[1])) << "," << csv_escape(format_vec(r.k[2])) << ","
                << format_double(r.kw[0]) << "," << format_double(r.kw[1]) << "," << format_double(r.kw[2]) << ","
                << csv_escape(format_vec(r.k_frac[0])) << "," << csv_escape(format_vec(r.k_frac[1])) << "," << csv_escape(format_vec(r.k_frac[2])) << ","
                << csv_escape(format_vec(r.target_cart)) << "," << csv_escape(format_vec(r.target_frac)) << ","
                << csv_escape(format_vec(r.target_frac_mapped)) << "," << csv_escape(format_vec(r.target_cart_mapped)) << "\n";
        }
    });
}

void write_pair_rows(
    const std::vector<PairOutputRow>& rows,
    const InputData& d,
    const std::string& output_path,
    std::size_t start_chunk_index
) {
    std::string header = "pair_id,pair_type,E1_index,E2_index,E3_index,E4_index,"
                         "k1_index,k2_index,k3_index,k4_index,E1,E2,E3,E4,"
                         "k1,k2,k3,k4,kw1,kw2,kw3,kw4,";
    header += d.auger_type == AugerType::EEH ? "k2_mapped" : "k4_mapped";
    header += ",probability";
    if (d.approach == Approach::ExactKpoint) {
        header += ",k1_nscf_index,k2_nscf_index,k3_nscf_index,k4_nscf_index,"
                  "k1_wc_index,k2_wc_index,k3_wc_index,k4_wc_index";
    }

    const std::string type = d.auger_type == AugerType::EEH ? "eeh" : "ehh";
    write_chunked_csv(rows.size(), output_path, header, start_chunk_index, [&](std::ofstream& out, std::size_t i) {
        const auto& r = rows[i];
        out << csv_escape(r.pair_id) << "," << type << ","
            << r.E_index[0] << "," << r.E_index[1] << "," << r.E_index[2] << "," << r.E_index[3] << ","
            << r.k_index[0] << "," << r.k_index[1] << "," << r.k_index[2] << "," << r.k_index[3] << ","
            << format_double(r.E[0]) << "," << format_double(r.E[1]) << "," << format_double(r.E[2]) << "," << format_double(r.E[3]) << ","
            << csv_escape(format_vec(r.k[0])) << "," << csv_escape(format_vec(r.k[1])) << ","
            << csv_escape(format_vec(r.k[2])) << "," << csv_escape(format_vec(r.k[3])) << ","
            << format_double(r.kw[0]) << "," << format_double(r.kw[1]) << ","
            << format_double(r.kw[2]) << "," << format_double(r.kw[3]) << ","
            << csv_escape(format_vec(r.mapped)) << "," << format_double(r.probability);
        if (d.approach == Approach::ExactKpoint) {
            out << "," << r.nscf_index[0] << "," << r.nscf_index[1] << "," << r.nscf_index[2] << "," << r.nscf_index[3]
                << "," << r.wc_index[0] << "," << r.wc_index[1] << "," << r.wc_index[2] << "," << r.wc_index[3];
        }
        out << "\n";
    });
}

void print_summary(const InputData& d) {
    std::cout << "Standalone C++ Auger pair generation\n";
    std::cout << "  task: " << (d.task == Task::ExactKpoints ? "exact_kpoints" : "pairs") << "\n";
    std::cout << "  auger_type: " << (d.auger_type == AugerType::EEH ? "eeh" : "ehh") << "\n";
    std::cout << "  approach: " << (d.approach == Approach::NearestKpoint ? "nearest_kpoint" : "exact_kpoint") << "\n";
    std::cout << "  search_mode: " << (d.search_mode == SearchMode::BruteForce ? "Brute_Force" : "Max_Heap") << "\n";
    std::cout << "  desired_total: " << (d.desired_total < 0 ? std::string("all") : std::to_string(d.desired_total)) << "\n";
    std::cout << "  previous_count: " << d.previous_count << "\n";
    std::cout << "  new rows requested: " << (new_target(d) < 0 ? std::string("all") : std::to_string(new_target(d))) << "\n";
    std::cout << "  states: E1=" << d.E1.size() << " E2=" << d.E2.size()
              << " E3=" << d.E3.size() << " E4=" << d.E4.size() << "\n";
    std::cout << "  exact rows supplied: " << d.exact_rows.size() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) {
            std::cerr << "Usage: " << argv[0]
                      << " pair_generation_input.bin output_base.csv [start_chunk_index]\n";
            return 2;
        }
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        std::size_t start_chunk_index = 1;
        if (argc == 4) {
            const long long parsed = std::stoll(argv[3]);
            if (parsed < 1) {
                throw std::runtime_error("start_chunk_index must be >= 1");
            }
            start_chunk_index = static_cast<std::size_t>(parsed);
        }

        InputData d = read_input(input_path);
        print_summary(d);

        if (d.task == Task::ExactKpoints) {
            auto rows = d.search_mode == SearchMode::MaxHeap
                ? generate_exact_heap(d)
                : generate_exact_brute(d);
            std::cout << "  generated exact-kpoint rows: " << rows.size() << "\n";
            write_exact_rows(rows, d, output_path, start_chunk_index);
        } else {
            std::vector<PairOutputRow> rows;
            if (d.approach == Approach::ExactKpoint) {
                rows = generate_exact_pairs(d);
            } else if (d.search_mode == SearchMode::MaxHeap) {
                rows = generate_nearest_heap(d);
            } else {
                rows = generate_nearest_brute(d);
            }
            std::cout << "  generated pair rows: " << rows.size() << "\n";
            write_pair_rows(rows, d, output_path, start_chunk_index);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
}
