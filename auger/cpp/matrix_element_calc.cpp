// matrix_element_calc.cpp
// =============================================================================
// Standalone C++ / OpenMP calculator for Auger Coulomb matrix elements.
//
// Reads the binary input produced by auger.cpp.prepare_cpp_input, computes the
// same fields returned by auger.matrix_elements._calc_matrix_element, and writes
// JSONL that AugerCalculator.read_matrix_elements() can read directly. Scalar
// dielectric input follows the existing scalar/Penn path; tensor dielectric
// input uses the same q-dependent model with directional epsilon_L(qhat).
// All matrix-element fields are written in eV^2, including direct/exchange
// diagnostic components.
//
// Build:
//   g++ -O3 -fopenmp -std=c++17 -o matrix_element_calc matrix_element_calc.cpp
//
// Usage:
//   ./matrix_element_calc input.bin output_1.jsonl [num_threads]
//   ./matrix_element_calc input.bin output_1.jsonl [num_threads] --resume
//   ./matrix_element_calc --config cpp_matrix_elements_config.json
// =============================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using cd = std::complex<double>;

struct Triple {
    int x, y, z;
    bool operator==(const Triple& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct TripleHash {
    std::size_t operator()(const Triple& k) const noexcept {
        std::size_t h = std::hash<int>{}(k.x);
        h ^= std::hash<int>{}(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

using GMap = std::unordered_map<Triple, cd, TripleHash>;

struct WfcEntry {
    int nG = 0;
    std::vector<int> G;
    std::vector<cd> C;
    GMap dict;
};

struct PairRecord {
    std::string pair_id;
    double k1[3], k2[3], k3[3], k4[3];
    int wfc_idx[4];
    int G_prime[3];
};

struct Params {
    int auger_type = 0;       // 0 = eeh, 1 = ehh
    double true_Bcell[9]{};   // row-major 3x3
    double a_fit = 0.0;
    double b_fit = 0.0;
    double c_fit = 0.0;
    double inv_debye = 0.0;
    double matrix_factor = 0.0;
    double V_m3 = 0.0;
    double eV_const = 0.0;
    int dielectric_mode = 0;  // 0 = scalar/Penn, 1 = tensor/directional Penn
    double dielectric_scalar = 0.0;
    double dielectric_tensor[9]{};
};

struct PairResult {
    std::string pair_id;
    double M2 = 0.0;
    double M2_0 = 0.0;
    double Md2 = 0.0;
    double Mx2 = 0.0;
    double Md2_0 = 0.0;
    double Mx2_0 = 0.0;
    bool has_error = false;
    std::string error_msg;
};

struct RunConfig {
    std::string input_binary;
    std::string output_jsonl;
    int num_threads = 0;
    bool num_threads_set = false;
    bool overwrite = false;
    bool append = false;
    bool resume = false;
    int progress_interval = 100;
    std::string log_file;
    bool log_file_set = false;
};

// -----------------------------------------------------------------------------
// Minimal JSON reader for the small control/config file.
// -----------------------------------------------------------------------------

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Object, Array };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::unordered_map<std::string, JsonValue> object;
    std::vector<JsonValue> array;
};

class JsonParser {
public:
    explicit JsonParser(std::string src) : source_(std::move(src)) {}

    JsonValue parse() {
        skip_ws();
        JsonValue value = parse_value();
        skip_ws();
        if (pos_ != source_.size()) {
            throw std::runtime_error("unexpected trailing content in JSON");
        }
        return value;
    }

private:
    std::string source_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[pos_]))) {
            ++pos_;
        }
    }

    char peek() const {
        if (pos_ >= source_.size()) return '\0';
        return source_[pos_];
    }

    char get() {
        if (pos_ >= source_.size()) {
            throw std::runtime_error("unexpected end of JSON");
        }
        return source_[pos_++];
    }

    void expect(char ch) {
        if (get() != ch) {
            std::ostringstream oss;
            oss << "expected '" << ch << "'";
            throw std::runtime_error(oss.str());
        }
    }

    JsonValue parse_value() {
        skip_ws();
        const char ch = peek();
        if (ch == '"') return parse_string_value();
        if (ch == '{') return parse_object();
        if (ch == '[') return parse_array();
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) return parse_number();
        if (source_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.boolean = true;
            return v;
        }
        if (source_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.boolean = false;
            return v;
        }
        if (source_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            JsonValue v;
            v.type = JsonValue::Type::Null;
            return v;
        }
        throw std::runtime_error("invalid JSON value");
    }

    JsonValue parse_string_value() {
        JsonValue v;
        v.type = JsonValue::Type::String;
        v.text = parse_string();
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            const char ch = get();
            if (ch == '"') break;
            if (ch == '\\') {
                const char esc = get();
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        int code = 0;
                        for (int i = 0; i < 4; ++i) {
                            const char h = get();
                            code <<= 4;
                            if (h >= '0' && h <= '9') code += h - '0';
                            else if (h >= 'a' && h <= 'f') code += 10 + h - 'a';
                            else if (h >= 'A' && h <= 'F') code += 10 + h - 'A';
                            else throw std::runtime_error("invalid unicode escape");
                        }
                        out.push_back(code >= 0 && code < 128 ? static_cast<char>(code) : '?');
                        break;
                    }
                    default:
                        throw std::runtime_error("invalid string escape");
                }
            } else {
                out.push_back(ch);
            }
        }
        return out;
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        if (peek() == '.') {
            ++pos_;
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.number = std::stod(source_.substr(start, pos_ - start));
        return v;
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::Type::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') throw std::runtime_error("expected object key string");
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.object[key] = parse_value();
            skip_ws();
            const char ch = get();
            if (ch == '}') break;
            if (ch != ',') throw std::runtime_error("expected ',' or '}' in object");
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::Type::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v.array.push_back(parse_value());
            skip_ws();
            const char ch = get();
            if (ch == ']') break;
            if (ch != ',') throw std::runtime_error("expected ',' or ']' in array");
        }
        return v;
    }
};

// -----------------------------------------------------------------------------
// Binary reader helpers
// -----------------------------------------------------------------------------

static bool read_exact(std::ifstream& f, void* dst, std::size_t n) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    return f.good();
}

template <typename T>
static T read_val(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

// -----------------------------------------------------------------------------
// Physics kernels
// -----------------------------------------------------------------------------

static inline cd I_ab(int gx, int gy, int gz,
                      const int* Ga, int Na,
                      const GMap& da, const GMap& db)
{
    cd total(0.0, 0.0);
    for (int i = 0; i < Na; ++i) {
        const int a0 = Ga[3 * i];
        const int a1 = Ga[3 * i + 1];
        const int a2 = Ga[3 * i + 2];
        auto ita = da.find({a0, a1, a2});
        if (ita == da.end()) continue;
        auto itb = db.find({a0 - gx, a1 - gy, a2 - gz});
        if (itb == db.end()) continue;
        total += std::conj(ita->second) * itb->second;
    }
    return total;
}

static inline double eps_penn(double q_mag, double a, double b, double c) {
    const double q_m = q_mag * 1e10;
    return 1.0 + 1.0 / (a + b * q_mag * q_mag + c * q_m * q_m * q_m * q_m);
}

static inline double W_screened(double q_mag, double eps, double lam) {
    return (1.0 / eps) * (1.0 / (q_mag * q_mag + lam * lam));
}

static inline double directional_epsilon(double qx, double qy, double qz, const double eps_tensor[9]) {
    const double q_mag = std::sqrt(qx * qx + qy * qy + qz * qz);
    if (q_mag <= 1e-14) {
        return (eps_tensor[0] + eps_tensor[4] + eps_tensor[8]) / 3.0;
    }
    const double hx = qx / q_mag;
    const double hy = qy / q_mag;
    const double hz = qz / q_mag;
    const double ex = eps_tensor[0] * hx + eps_tensor[1] * hy + eps_tensor[2] * hz;
    const double ey = eps_tensor[3] * hx + eps_tensor[4] * hy + eps_tensor[5] * hz;
    const double ez = eps_tensor[6] * hx + eps_tensor[7] * hy + eps_tensor[8] * hz;
    return hx * ex + hy * ey + hz * ez;
}

static inline double eps_directional_penn(double qx, double qy, double qz, const Params& P) {
    const double eps_l = directional_epsilon(qx, qy, qz, P.dielectric_tensor);
    if (!std::isfinite(eps_l) || eps_l <= 0.0) {
        throw std::runtime_error("directional dielectric epsilon_L is non-positive");
    }
    if (std::abs(eps_l - 1.0) <= 1e-14) {
        return 1.0;
    }
    const double q_mag = std::sqrt(qx * qx + qy * qy + qz * qz);
    const double a_dir = 1.0 / (eps_l - 1.0);
    return eps_penn(q_mag, a_dir, P.b_fit, P.c_fit);
}

static inline double W_directional(double qx, double qy, double qz, const Params& P) {
    const double q_mag = std::sqrt(qx * qx + qy * qy + qz * qz);
    const double eps_q = eps_directional_penn(qx, qy, qz, P);
    return W_screened(q_mag, eps_q, P.inv_debye);
}

static std::vector<Triple> build_common_G(
    const WfcEntry& w1, const WfcEntry& w2,
    const WfcEntry& w3, const WfcEntry& w4)
{
    std::unordered_set<Triple, TripleHash> seen;
    auto insert_all = [&](const WfcEntry& w) {
        for (int i = 0; i < w.nG; ++i) {
            seen.insert({w.G[3 * i], w.G[3 * i + 1], w.G[3 * i + 2]});
        }
    };
    insert_all(w1);
    insert_all(w2);
    insert_all(w3);
    insert_all(w4);
    std::vector<Triple> out{seen.begin(), seen.end()};
    std::sort(out.begin(), out.end(), [](const Triple& a, const Triple& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });
    return out;
}

static PairResult compute_pair(
    const PairRecord& pr,
    const std::vector<WfcEntry>& wfcs,
    const Params& P)
{
    PairResult res;
    res.pair_id = pr.pair_id;

    try {
        for (int idx : pr.wfc_idx) {
            if (idx < 0 || idx >= static_cast<int>(wfcs.size())) {
                throw std::runtime_error("pair references a wavefunction index outside the input table");
            }
        }

        const WfcEntry& w1 = wfcs[pr.wfc_idx[0]];
        const WfcEntry& w2 = wfcs[pr.wfc_idx[1]];
        const WfcEntry& w3 = wfcs[pr.wfc_idx[2]];
        const WfcEntry& w4 = wfcs[pr.wfc_idx[3]];

        std::vector<Triple> common_G = build_common_G(w1, w2, w3, w4);
        const double* B = P.true_Bcell;
        const int Gpx = pr.G_prime[0];
        const int Gpy = pr.G_prime[1];
        const int Gpz = pr.G_prime[2];

        cd Md_sum(0.0, 0.0);
        cd Mx_sum(0.0, 0.0);
        cd Md_G0(0.0, 0.0);
        cd Mx_G0(0.0, 0.0);

        const int* G1p = w1.G.data();
        const int* G2p = w2.G.data();
        const int* G3p = w3.G.data();
        const int N1 = w1.nG;
        const int N2 = w2.nG;
        const int N3 = w3.nG;

        const GMap& d1 = w1.dict;
        const GMap& d2 = w2.dict;
        const GMap& d3 = w3.dict;
        const GMap& d4 = w4.dict;

        for (const Triple& g : common_G) {
            const int gx = g.x;
            const int gy = g.y;
            const int gz = g.z;

            // Python uses np.dot(G, true_Bcell), i.e. row-vector times matrix.
            const double Gb0 = gx * B[0] + gy * B[3] + gz * B[6];
            const double Gb1 = gx * B[1] + gy * B[4] + gz * B[7];
            const double Gb2 = gx * B[2] + gy * B[5] + gz * B[8];

            const int pgx = Gpx - gx;
            const int pgy = Gpy - gy;
            const int pgz = Gpz - gz;

            cd i34, i12, i32, i14;
            double qdx, qdy, qdz;
            double qxx, qxy, qxz;

            if (P.auger_type == 0) {
                i34 = I_ab(gx, gy, gz, G3p, N3, d3, d4);
                i12 = I_ab(pgx, pgy, pgz, G1p, N1, d1, d2);
                i32 = I_ab(gx, gy, gz, G3p, N3, d3, d2);
                i14 = I_ab(pgx, pgy, pgz, G1p, N1, d1, d4);
                qdx = pr.k3[0] - pr.k4[0] + Gb0;
                qdy = pr.k3[1] - pr.k4[1] + Gb1;
                qdz = pr.k3[2] - pr.k4[2] + Gb2;
                qxx = pr.k3[0] - pr.k2[0] + Gb0;
                qxy = pr.k3[1] - pr.k2[1] + Gb1;
                qxz = pr.k3[2] - pr.k2[2] + Gb2;
            } else {
                i34 = I_ab(gx, gy, gz, G2p, N2, d2, d1);
                i12 = I_ab(pgx, pgy, pgz, G3p, N3, d3, d4);
                i32 = I_ab(gx, gy, gz, G2p, N2, d2, d4);
                i14 = I_ab(pgx, pgy, pgz, G3p, N3, d3, d1);
                qdx = pr.k2[0] - pr.k1[0] + Gb0;
                qdy = pr.k2[1] - pr.k1[1] + Gb1;
                qdz = pr.k2[2] - pr.k1[2] + Gb2;
                qxx = pr.k2[0] - pr.k4[0] + Gb0;
                qxy = pr.k2[1] - pr.k4[1] + Gb1;
                qxz = pr.k2[2] - pr.k4[2] + Gb2;
            }

            const double arg_d = std::sqrt(qdx * qdx + qdy * qdy + qdz * qdz);
            const double arg_x = std::sqrt(qxx * qxx + qxy * qxy + qxz * qxz);

            double Wd;
            double Wx;
            if (P.dielectric_mode == 1) {
                Wd = W_directional(qdx, qdy, qdz, P);
                Wx = W_directional(qxx, qxy, qxz, P);
            } else {
                const double eps_d = eps_penn(arg_d, P.a_fit, P.b_fit, P.c_fit);
                const double eps_x = eps_penn(arg_x, P.a_fit, P.b_fit, P.c_fit);
                Wd = W_screened(arg_d, eps_d, P.inv_debye);
                Wx = W_screened(arg_x, eps_x, P.inv_debye);
            }

            const cd md_term = i34 * i12 * Wd;
            const cd mx_term = i32 * i14 * Wx;

            Md_sum += md_term;
            Mx_sum += mx_term;

            if (gx == 0 && gy == 0 && gz == 0) {
                Md_G0 = md_term;
                Mx_G0 = mx_term;
            }
        }

        const double prefactor = (P.matrix_factor * P.matrix_factor)
                               / (P.V_m3 * P.V_m3 * P.eV_const * P.eV_const);

        const double Md2_raw = std::norm(Md_sum);
        const double Mx2_raw = std::norm(Mx_sum);
        const double Mdx2_raw = std::norm(Md_sum - Mx_sum);
        res.M2 = (Md2_raw + Mx2_raw + Mdx2_raw) * prefactor;
        res.Md2 = Md2_raw * prefactor;
        res.Mx2 = Mx2_raw * prefactor;

        const double Md2_0_raw = std::norm(Md_G0);
        const double Mx2_0_raw = std::norm(Mx_G0);
        const double Mdx2_0_raw = std::norm(Md_G0 - Mx_G0);
        res.M2_0 = (Md2_0_raw + Mx2_0_raw + Mdx2_0_raw) * prefactor;
        res.Md2_0 = Md2_0_raw * prefactor;
        res.Mx2_0 = Mx2_0_raw * prefactor;
    } catch (const std::exception& e) {
        res.has_error = true;
        res.error_msg = e.what();
    }

    return res;
}

// -----------------------------------------------------------------------------
// Input loading
// -----------------------------------------------------------------------------

static bool load_input(const std::string& path,
                       Params& params,
                       std::vector<WfcEntry>& wfcs,
                       std::vector<PairRecord>& pairs)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open input binary: " << path << "\n";
        return false;
    }

    char magic[8];
    if (!read_exact(f, magic, 8) || std::memcmp(magic, "AUGERCPP", 8) != 0) {
        std::cerr << "Error: invalid AUGERCPP binary header in " << path << "\n";
        return false;
    }

    const int version = read_val<int32_t>(f);
    if (version != 1 && version != 2) {
        std::cerr << "Error: unsupported AUGERCPP binary version " << version << "\n";
        return false;
    }

    params.auger_type = read_val<int32_t>(f);
    read_exact(f, params.true_Bcell, 9 * sizeof(double));
    params.a_fit = read_val<double>(f);
    params.b_fit = read_val<double>(f);
    params.c_fit = read_val<double>(f);
    params.inv_debye = read_val<double>(f);
    params.matrix_factor = read_val<double>(f);
    params.V_m3 = read_val<double>(f);
    params.eV_const = read_val<double>(f);
    if (version >= 2) {
        params.dielectric_mode = read_val<int32_t>(f);
        params.dielectric_scalar = read_val<double>(f);
        read_exact(f, params.dielectric_tensor, 9 * sizeof(double));
        if (params.dielectric_mode != 0 && params.dielectric_mode != 1) {
            std::cerr << "Error: invalid dielectric_mode " << params.dielectric_mode << "\n";
            return false;
        }
        if (params.dielectric_mode == 1) {
            const double avg = (params.dielectric_tensor[0] +
                                params.dielectric_tensor[4] +
                                params.dielectric_tensor[8]) / 3.0;
            if (!std::isfinite(avg) || avg <= 0.0) {
                std::cerr << "Error: dielectric tensor has non-positive trace average\n";
                return false;
            }
            std::cout << "  Dielectric tensor mode enabled; Debye scalar = "
                      << params.dielectric_scalar << "\n";
        }
    } else {
        params.dielectric_mode = 0;
        params.dielectric_scalar = 0.0;
        for (double& x : params.dielectric_tensor) x = 0.0;
    }

    const int num_wfc = read_val<int32_t>(f);
    if (num_wfc < 0) {
        std::cerr << "Error: negative wavefunction entry count in input\n";
        return false;
    }
    wfcs.resize(static_cast<std::size_t>(num_wfc));
    std::cout << "  Loading " << num_wfc << " wavefunction states ...\n";
    for (int i = 0; i < num_wfc; ++i) {
        WfcEntry& w = wfcs[static_cast<std::size_t>(i)];
        w.nG = read_val<int32_t>(f);
        if (w.nG < 0) {
            std::cerr << "Error: negative G-vector count for wavefunction entry " << i << "\n";
            return false;
        }
        w.G.resize(static_cast<std::size_t>(w.nG) * 3);
        if (!read_exact(f, w.G.data(), static_cast<std::size_t>(w.nG) * 3 * sizeof(int32_t))) {
            std::cerr << "Error: failed reading G-vectors for wavefunction entry " << i << "\n";
            return false;
        }

        std::vector<double> interleaved(static_cast<std::size_t>(w.nG) * 2);
        if (!read_exact(f, interleaved.data(), static_cast<std::size_t>(w.nG) * 2 * sizeof(double))) {
            std::cerr << "Error: failed reading coefficients for wavefunction entry " << i << "\n";
            return false;
        }

        w.C.resize(static_cast<std::size_t>(w.nG));
        w.dict.reserve(static_cast<std::size_t>(w.nG) * 2);
        for (int j = 0; j < w.nG; ++j) {
            w.C[static_cast<std::size_t>(j)] =
                cd(interleaved[static_cast<std::size_t>(2 * j)],
                   interleaved[static_cast<std::size_t>(2 * j + 1)]);
            w.dict[{w.G[3 * j], w.G[3 * j + 1], w.G[3 * j + 2]}] =
                w.C[static_cast<std::size_t>(j)];
        }

        if ((i + 1) % 200 == 0 || (i + 1) == num_wfc) {
            std::cout << "    " << (i + 1) << "/" << num_wfc << "\r" << std::flush;
        }
    }
    std::cout << "\n";

    const int num_pairs = read_val<int32_t>(f);
    if (num_pairs < 0) {
        std::cerr << "Error: negative pair count in input\n";
        return false;
    }
    pairs.resize(static_cast<std::size_t>(num_pairs));
    std::cout << "  Loading " << num_pairs << " pairs ...\n";
    for (int i = 0; i < num_pairs; ++i) {
        PairRecord& p = pairs[static_cast<std::size_t>(i)];
        const int id_len = read_val<int32_t>(f);
        if (id_len < 0) {
            std::cerr << "Error: negative pair_id length for pair " << i << "\n";
            return false;
        }
        p.pair_id.resize(static_cast<std::size_t>(id_len));
        if (!read_exact(f, p.pair_id.data(), static_cast<std::size_t>(id_len))) {
            std::cerr << "Error: failed reading pair_id for pair " << i << "\n";
            return false;
        }
        read_exact(f, p.k1, 3 * sizeof(double));
        read_exact(f, p.k2, 3 * sizeof(double));
        read_exact(f, p.k3, 3 * sizeof(double));
        read_exact(f, p.k4, 3 * sizeof(double));
        read_exact(f, p.wfc_idx, 4 * sizeof(int32_t));
        read_exact(f, p.G_prime, 3 * sizeof(int32_t));
    }

    if (!f.good() && !f.eof()) {
        std::cerr << "Error: stream error while reading " << path << "\n";
        return false;
    }

    std::cout << "  Input loaded: " << num_wfc << " wavefunction states, "
              << num_pairs << " pairs, auger_type="
              << (params.auger_type == 0 ? "eeh" : "ehh") << "\n";
    return true;
}

// -----------------------------------------------------------------------------
// Config and output helpers
// -----------------------------------------------------------------------------

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static bool read_text_file(const std::string& path, std::string& text, std::string& error) {
    std::ifstream f(path);
    if (!f.is_open()) {
        error = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    text = ss.str();
    return true;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    std::ostringstream oss;
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch));
                    out += oss.str();
                } else {
                    out.push_back(ch);
                }
        }
    }
    return out;
}

static bool parse_positive_int(const std::string& s, int& out) {
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v <= 0 || v > 2147483647L) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static const JsonValue* object_member(const JsonValue& obj, const std::string& key) {
    if (obj.type != JsonValue::Type::Object) return nullptr;
    const auto it = obj.object.find(key);
    if (it == obj.object.end()) return nullptr;
    return &it->second;
}

static bool get_string_if_present(const JsonValue* obj,
                                  const std::string& key,
                                  const std::string& label,
                                  std::string& value,
                                  bool& found,
                                  std::string& error,
                                  const std::string& selector = "")
{
    found = false;
    if (obj == nullptr || obj->type != JsonValue::Type::Object) return true;
    const auto it = obj->object.find(key);
    if (it == obj->object.end() || it->second.type == JsonValue::Type::Null) return true;
    found = true;

    const JsonValue* field = &it->second;
    if (field->type == JsonValue::Type::Object && !selector.empty()) {
        const auto selected = field->object.find(selector);
        if (selected == field->object.end() || selected->second.type == JsonValue::Type::Null) {
            error = "config field '" + label + "' is an object but has no entry for '" + selector + "'";
            return false;
        }
        field = &selected->second;
    }

    if (field->type != JsonValue::Type::String || field->text.empty()) {
        error = "config field '" + label + "' must be a non-empty string";
        return false;
    }
    value = field->text;
    return true;
}

static bool get_bool_if_present(const JsonValue* obj,
                                const std::string& key,
                                const std::string& label,
                                bool& value,
                                bool& found,
                                std::string& error)
{
    found = false;
    if (obj == nullptr || obj->type != JsonValue::Type::Object) return true;
    const auto it = obj->object.find(key);
    if (it == obj->object.end() || it->second.type == JsonValue::Type::Null) return true;
    found = true;
    if (it->second.type != JsonValue::Type::Bool) {
        error = "config field '" + label + "' must be true or false";
        return false;
    }
    value = it->second.boolean;
    return true;
}

static bool get_int_if_present(const JsonValue* obj,
                               const std::string& key,
                               const std::string& label,
                               int& value,
                               bool& found,
                               std::string& error)
{
    found = false;
    if (obj == nullptr || obj->type != JsonValue::Type::Object) return true;
    const auto it = obj->object.find(key);
    if (it == obj->object.end() || it->second.type == JsonValue::Type::Null) return true;
    found = true;
    if (it->second.type != JsonValue::Type::Number) {
        error = "config field '" + label + "' must be a number";
        return false;
    }
    const int iv = static_cast<int>(it->second.number);
    if (iv <= 0 || std::fabs(it->second.number - static_cast<double>(iv)) > 1e-9) {
        error = "config field '" + label + "' must be a positive integer";
        return false;
    }
    value = iv;
    return true;
}

static bool get_required_string_any(const JsonValue& root,
                                    const JsonValue* matrix_elements,
                                    const std::string& flat_key,
                                    const std::string& nested_key,
                                    const std::string& selector,
                                    std::string& value,
                                    std::string& error)
{
    bool found = false;
    if (!get_string_if_present(&root, flat_key, flat_key, value, found, error, selector)) return false;
    if (found) return true;
    const std::string nested_label = "matrix_elements." + nested_key;
    if (!get_string_if_present(matrix_elements, nested_key, nested_label, value, found, error, selector)) return false;
    if (found) return true;
    error = "config is missing required field '" + flat_key + "' or '" + nested_label + "'";
    return false;
}

static bool get_optional_string_any(const JsonValue& root,
                                    const JsonValue* matrix_elements,
                                    const std::string& flat_key,
                                    const std::string& nested_key,
                                    const std::string& selector,
                                    std::string& value,
                                    bool& was_set,
                                    std::string& error)
{
    bool found = false;
    if (!get_string_if_present(&root, flat_key, flat_key, value, found, error, selector)) return false;
    if (found) {
        was_set = true;
        return true;
    }
    const std::string nested_label = "matrix_elements." + nested_key;
    if (!get_string_if_present(matrix_elements, nested_key, nested_label, value, found, error, selector)) return false;
    if (found) was_set = true;
    return true;
}

static bool get_optional_bool_any(const JsonValue& root,
                                  const JsonValue* matrix_elements,
                                  const std::string& flat_key,
                                  const std::string& nested_key,
                                  bool& value,
                                  std::string& error)
{
    bool found = false;
    if (!get_bool_if_present(&root, flat_key, flat_key, value, found, error)) return false;
    if (found) return true;
    const std::string nested_label = "matrix_elements." + nested_key;
    return get_bool_if_present(matrix_elements, nested_key, nested_label, value, found, error);
}

static bool get_optional_int_any(const JsonValue& root,
                                 const JsonValue* matrix_elements,
                                 const std::string& flat_key,
                                 const std::string& nested_key,
                                 int& value,
                                 bool& was_set,
                                 std::string& error)
{
    bool found = false;
    if (!flat_key.empty()) {
        if (!get_int_if_present(&root, flat_key, flat_key, value, found, error)) return false;
        if (found) {
            was_set = true;
            return true;
        }
    }
    const std::string nested_label = "matrix_elements." + nested_key;
    if (!get_int_if_present(matrix_elements, nested_key, nested_label, value, found, error)) return false;
    if (found) was_set = true;
    return true;
}

static bool validate_output_flags(const RunConfig& cfg, std::string& error) {
    if (cfg.overwrite && cfg.append) {
        error = "'overwrite' and 'append' cannot both be true";
        return false;
    }
    if (cfg.overwrite && cfg.resume) {
        error = "'overwrite' and 'resume' cannot both be true";
        return false;
    }
    if (cfg.append && cfg.resume) {
        error = "'append' and 'resume' cannot both be true; use resume to append only missing pairs";
        return false;
    }
    return true;
}

static bool load_json_config(const std::string& path, RunConfig& cfg, std::string& error) {
    std::string text;
    if (!read_text_file(path, text, error)) return false;

    JsonValue root;
    try {
        root = JsonParser(text).parse();
    } catch (const std::exception& e) {
        error = "invalid JSON in " + path + ": " + e.what();
        return false;
    }

    if (root.type != JsonValue::Type::Object) {
        error = "config root must be a JSON object";
        return false;
    }

    const JsonValue* matrix_elements = object_member(root, "matrix_elements");
    std::string selected_auger_type = "eeh";
    bool auger_type_found = false;
    if (!get_string_if_present(&root, "auger_type", "auger_type",
                               selected_auger_type, auger_type_found, error)) return false;
    if (!auger_type_found) {
        if (!get_string_if_present(matrix_elements, "cpp_auger_type",
                                   "matrix_elements.cpp_auger_type",
                                   selected_auger_type, auger_type_found, error)) return false;
    }
    if (!auger_type_found) {
        if (!get_string_if_present(matrix_elements, "auger_type",
                                   "matrix_elements.auger_type",
                                   selected_auger_type, auger_type_found, error)) return false;
    }

    if (!get_required_string_any(root, matrix_elements, "input_binary", "cpp_input_binary",
                                 selected_auger_type, cfg.input_binary, error)) return false;
    if (!get_required_string_any(root, matrix_elements, "output_jsonl", "cpp_output_jsonl",
                                 selected_auger_type, cfg.output_jsonl, error)) return false;

    if (!get_optional_int_any(root, matrix_elements, "num_threads", "cpp_num_threads",
                              cfg.num_threads, cfg.num_threads_set, error)) return false;
    if (!cfg.num_threads_set) {
        if (!get_optional_int_any(root, matrix_elements, "", "num_threads",
                                  cfg.num_threads, cfg.num_threads_set, error)) return false;
    }

    if (!get_optional_bool_any(root, matrix_elements, "overwrite", "cpp_overwrite",
                               cfg.overwrite, error)) return false;
    if (!get_optional_bool_any(root, matrix_elements, "append", "cpp_append",
                               cfg.append, error)) return false;
    if (!get_optional_bool_any(root, matrix_elements, "resume", "cpp_resume",
                               cfg.resume, error)) return false;

    bool progress_set = false;
    if (!get_optional_int_any(root, matrix_elements, "progress_interval", "cpp_progress_interval",
                              cfg.progress_interval, progress_set, error)) return false;
    if (!get_optional_string_any(root, matrix_elements, "log_file", "cpp_log_file", selected_auger_type,
                                 cfg.log_file, cfg.log_file_set, error)) return false;

    return validate_output_flags(cfg, error);
}

static void print_usage(const char* exe) {
    std::cerr
        << "Usage:\n"
        << "  " << exe << " input.bin output_1.jsonl [num_threads] [--overwrite|--append|--resume]\n"
        << "  " << exe << " -- input.bin output_1.jsonl [num_threads] [--overwrite|--append|--resume]\n"
        << "  " << exe << " --config cpp_matrix_elements_config.json\n"
        << "  " << exe << " --config auger_config_template.json\n\n"
        << "Options:\n"
        << "  --num_threads N        Set OpenMP thread count for this run.\n"
        << "  --overwrite            Replace existing output chunks.\n"
        << "  --append               Append all results to existing output chunks.\n"
        << "  --resume               Append only pair_ids not already completed in output chunks.\n"
        << "  --progress_interval N  Print progress every N completed pairs.\n"
        << "  --log_file PATH        Also write a short run summary to PATH.\n";
}

static bool parse_cli(int argc, char* argv[], RunConfig& cfg, std::string& error) {
    if (argc < 2) {
        print_usage(argv[0]);
        error = "missing arguments";
        return false;
    }

    int i = 1;
    if (std::string(argv[i]) == "--") ++i;

    if (i < argc && (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")) {
        print_usage(argv[0]);
        std::exit(0);
    }

    if (i < argc && (std::string(argv[i]) == "--config" || std::string(argv[i]) == "-c")) {
        if (i + 1 >= argc) {
            error = "--config requires a JSON file path";
            return false;
        }
        const std::string config_path = argv[i + 1];
        if (!load_json_config(config_path, cfg, error)) return false;
        i += 2;
    } else {
        if (i + 2 >= argc) {
            print_usage(argv[0]);
            error = "positional mode requires input.bin and output_1.jsonl";
            return false;
        }
        cfg.input_binary = argv[i++];
        cfg.output_jsonl = argv[i++];
    }

    while (i < argc) {
        const std::string arg = argv[i++];
        if (arg == "--num_threads") {
            if (i >= argc || !parse_positive_int(argv[i], cfg.num_threads)) {
                error = "--num_threads requires a positive integer";
                return false;
            }
            cfg.num_threads_set = true;
            ++i;
        } else if (arg == "--overwrite") {
            cfg.overwrite = true;
        } else if (arg == "--append") {
            cfg.append = true;
        } else if (arg == "--resume") {
            cfg.resume = true;
        } else if (arg == "--progress_interval") {
            if (i >= argc || !parse_positive_int(argv[i], cfg.progress_interval)) {
                error = "--progress_interval requires a positive integer";
                return false;
            }
            ++i;
        } else if (arg == "--log_file") {
            if (i >= argc) {
                error = "--log_file requires a path";
                return false;
            }
            cfg.log_file = argv[i++];
            cfg.log_file_set = true;
        } else if (!cfg.num_threads_set) {
            int threads = 0;
            if (!parse_positive_int(arg, threads)) {
                error = "unknown argument: " + arg;
                return false;
            }
            cfg.num_threads = threads;
            cfg.num_threads_set = true;
        } else {
            error = "unknown argument: " + arg;
            return false;
        }
    }

    if (cfg.input_binary.empty() || cfg.output_jsonl.empty()) {
        error = "input_binary and output_jsonl must be set";
        return false;
    }
    return validate_output_flags(cfg, error);
}

static bool extract_pair_id(const std::string& line, std::string& pair_id) {
    const std::string key = "\"pair_id\"";
    std::size_t pos = line.find(key);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + key.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    ++pos;

    std::string out;
    while (pos < line.size()) {
        const char ch = line[pos++];
        if (ch == '"') {
            pair_id = out;
            return true;
        }
        if (ch == '\\' && pos < line.size()) {
            out.push_back(line[pos++]);
        } else {
            out.push_back(ch);
        }
    }
    return false;
}

static std::unordered_set<std::string> read_completed_pair_ids(const std::string& path) {
    std::unordered_set<std::string> done;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("\"error\"") != std::string::npos) continue;
        std::string pid;
        if (extract_pair_id(line, pid)) done.insert(pid);
    }
    return done;
}

static void merge_completed_pair_ids(const std::string& path,
                                     std::unordered_set<std::string>& done) {
    const auto cur = read_completed_pair_ids(path);
    done.insert(cur.begin(), cur.end());
}

static bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size()
        && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string strip_jsonl_suffix(const std::string& path) {
    return ends_with(path, ".jsonl") ? path.substr(0, path.size() - 6) : path;
}

static std::string chunk_base_from_output(const std::string& output_jsonl) {
    std::string base = strip_jsonl_suffix(output_jsonl);
    if (ends_with(base, "_1")) {
        base = base.substr(0, base.size() - 2);
    }
    return base;
}

static std::string chunk_path(const std::string& base, int part) {
    std::ostringstream oss;
    oss << base << "_" << part << ".jsonl";
    return oss.str();
}

static long count_lines(const std::string& path) {
    std::ifstream f(path);
    long n = 0;
    std::string line;
    while (std::getline(f, line)) ++n;
    return n;
}

static std::vector<std::string> existing_output_paths(const std::string& output_jsonl,
                                                      const std::string& base) {
    std::vector<std::string> paths;
    const std::string first_chunk = chunk_path(base, 1);
    if (output_jsonl != first_chunk && file_exists(output_jsonl)) {
        paths.push_back(output_jsonl);
    }
    for (int part = 1; ; ++part) {
        const std::string path = chunk_path(base, part);
        if (!file_exists(path)) break;
        paths.push_back(path);
    }
    return paths;
}

struct JsonlChunkWriter {
    std::string base;
    int part = 1;
    long rows_in_part = 0;
    FILE* out = nullptr;
    std::vector<std::string> written_paths;
};

static constexpr long JSONL_CHUNK_SIZE = 1000000L;

static bool open_chunk(JsonlChunkWriter& writer, const char* mode) {
    const std::string path = chunk_path(writer.base, writer.part);
    writer.out = std::fopen(path.c_str(), mode);
    if (!writer.out) {
        std::cerr << "Error: cannot open output JSONL for writing: " << path << "\n";
        return false;
    }
    if (std::find(writer.written_paths.begin(), writer.written_paths.end(), path)
        == writer.written_paths.end()) {
        writer.written_paths.push_back(path);
    }
    return true;
}

static bool advance_chunk(JsonlChunkWriter& writer) {
    if (writer.out) {
        std::fclose(writer.out);
        writer.out = nullptr;
    }
    ++writer.part;
    writer.rows_in_part = 0;
    return open_chunk(writer, "a");
}

static bool open_output_chunks(const RunConfig& cfg,
                               JsonlChunkWriter& writer,
                               std::unordered_set<std::string>& completed_ids)
{
    writer.base = chunk_base_from_output(cfg.output_jsonl);
    const auto existing = existing_output_paths(cfg.output_jsonl, writer.base);
    const bool exists = !existing.empty();

    if (exists) {
        if (cfg.overwrite) {
            for (const auto& path : existing) {
                std::remove(path.c_str());
            }
            writer.part = 1;
            writer.rows_in_part = 0;
            return open_chunk(writer, "w");
        }

        if (cfg.resume) {
            for (const auto& path : existing) {
                merge_completed_pair_ids(path, completed_ids);
            }
            std::cout << "  Resume enabled: " << completed_ids.size()
                      << " completed pair_ids found in existing output chunk(s).\n";
        } else if (cfg.append) {
            std::cout << "  Append enabled: existing output chunk(s) will be kept.\n";
        } else {
            std::cerr
                << "Error: output file/chunks already exist for: " << cfg.output_jsonl << "\n"
                << "Use --overwrite to replace them, --resume to skip completed pair_ids, "
                << "or --append to append all results.\n";
            return false;
        }
    }

    writer.part = 1;
    writer.rows_in_part = count_lines(chunk_path(writer.base, writer.part));
    while (writer.rows_in_part >= JSONL_CHUNK_SIZE) {
        ++writer.part;
        writer.rows_in_part = count_lines(chunk_path(writer.base, writer.part));
    }
    return open_chunk(writer, "a");
}

static void close_output_chunks(JsonlChunkWriter& writer) {
    if (writer.out) {
        std::fclose(writer.out);
        writer.out = nullptr;
    }
}

static void write_json_result(FILE* out, const PairResult& r) {
    const std::string pid = json_escape(r.pair_id);
    if (r.has_error) {
        const std::string err = json_escape(r.error_msg);
        std::fprintf(out, "{\"pair_id\":\"%s\",\"error\":\"%s\"}\n",
                     pid.c_str(), err.c_str());
        return;
    }

    std::fprintf(out,
        "{\"pair_id\":\"%s\",\"|M|^2\":%.17g,\"|M(G=0)|^2\":%.17g,"
        "\"|Md|^2\":%.17g,\"|Mx|^2\":%.17g,"
        "\"|Md(G=0)|^2\":%.17g,\"|Mx(G=0)|^2\":%.17g}\n",
        pid.c_str(), r.M2, r.M2_0, r.Md2, r.Mx2, r.Md2_0, r.Mx2_0);
}

static bool write_json_result(JsonlChunkWriter& writer, const PairResult& r) {
    if (!writer.out) {
        if (!open_chunk(writer, "a")) return false;
    }
    if (writer.rows_in_part >= JSONL_CHUNK_SIZE) {
        if (!advance_chunk(writer)) return false;
    }
    write_json_result(writer.out, r);
    std::fflush(writer.out);
    ++writer.rows_in_part;
    return true;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    RunConfig cfg;
    std::string error;
    if (!parse_cli(argc, argv, cfg, error)) {
        std::cerr << "Error: " << error << "\n";
        return 2;
    }

    std::ofstream log;
    if (cfg.log_file_set) {
        log.open(cfg.log_file, std::ios::app);
        if (!log.is_open()) {
            std::cerr << "Error: cannot open log_file: " << cfg.log_file << "\n";
            return 2;
        }
    }

#ifdef _OPENMP
    if (cfg.num_threads_set) {
        omp_set_num_threads(cfg.num_threads);
    }
    omp_set_nested(0);
    omp_set_dynamic(0);
    const int max_threads = omp_get_max_threads();
    std::cout << "  OpenMP threads: " << max_threads << "\n";
#else
    const int max_threads = 1;
    std::cout << "  WARNING: compiled without OpenMP, running single-threaded.\n";
#endif

    if (log.is_open()) {
        log << "input_binary=" << cfg.input_binary << "\n"
            << "output_jsonl=" << cfg.output_jsonl << "\n"
            << "threads=" << max_threads << "\n";
    }

    Params params;
    std::vector<WfcEntry> wfcs;
    std::vector<PairRecord> pairs;
    if (!load_input(cfg.input_binary, params, wfcs, pairs)) {
        return 1;
    }

    std::unordered_set<std::string> completed_ids;
    JsonlChunkWriter writer;
    if (!open_output_chunks(cfg, writer, completed_ids)) return 1;

    std::vector<int> work_indices;
    work_indices.reserve(pairs.size());
    int skipped = 0;
    for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
        if (cfg.resume && completed_ids.find(pairs[static_cast<std::size_t>(i)].pair_id) != completed_ids.end()) {
            ++skipped;
            continue;
        }
        work_indices.push_back(i);
    }

    const int total_work = static_cast<int>(work_indices.size());
    std::cout << "\n  Computing " << total_work << " matrix elements";
    if (skipped > 0) std::cout << " (" << skipped << " skipped by resume)";
    std::cout << " ...\n";

    int done_count = 0;
    int n_errors = 0;
    std::mutex io_mutex;

    if (total_work > 0) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int wi = 0; wi < total_work; ++wi) {
            const int pair_index = work_indices[static_cast<std::size_t>(wi)];
            PairResult r = compute_pair(pairs[static_cast<std::size_t>(pair_index)], wfcs, params);

            {
                std::lock_guard<std::mutex> lock(io_mutex);
                if (!write_json_result(writer, r)) {
                    ++n_errors;
                    continue;
                }

                if (r.has_error) ++n_errors;
                ++done_count;
                if (cfg.progress_interval > 0 &&
                    (done_count % cfg.progress_interval == 0 || done_count == total_work)) {
                    const double pct = 100.0 * done_count / total_work;
                    std::printf("    [%6d/%d] %5.1f%%\r", done_count, total_work, pct);
                    std::fflush(stdout);
                }
            }
        }
    }

    close_output_chunks(writer);
    std::cout << "\n";

    std::cout << "  Done. " << total_work << " pairs computed";
    if (skipped > 0) std::cout << ", " << skipped << " skipped";
    if (n_errors > 0) std::cout << " (" << n_errors << " errors)";
    std::cout << ".\n  Output chunk(s):";
    if (writer.written_paths.empty()) {
        std::cout << " " << chunk_path(writer.base, writer.part);
    } else {
        for (const auto& path : writer.written_paths) {
            std::cout << "\n    " << path;
        }
    }
    std::cout << "\n";

    if (log.is_open()) {
        log << "computed=" << total_work << "\n"
            << "skipped=" << skipped << "\n"
            << "errors=" << n_errors << "\n";
    }

    return n_errors == 0 ? 0 : 1;
}
