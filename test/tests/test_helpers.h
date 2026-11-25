#ifndef GCTA_TEST_HELPERS_H
#define GCTA_TEST_HELPERS_H

#include "test_config.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <zlib.h>
#include <sqlite3.h>

namespace gcta_test {

inline void RemoveTree(const std::string& path) {
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        DIR* dir = opendir(path.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
                    continue;
                }
                RemoveTree(path + "/" + entry->d_name);
            }
            closedir(dir);
        }
        rmdir(path.c_str());
    } else {
        unlink(path.c_str());
    }
}

class TempDir {
public:
    explicit TempDir(const std::string& prefix = "gcta_ut_") {
        std::string tmpl = "/tmp/" + prefix + "XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        char* res = mkdtemp(buf.data());
        if (!res) {
            throw std::runtime_error("mkdtemp failed");
        }
        path_ = res;
    }

    ~TempDir() {
        if (!path_.empty()) {
            RemoveTree(path_);
        }
    }

    const std::string& path() const { return path_; }
    std::string file(const std::string& name) const { return path_ + "/" + name; }

private:
    std::string path_;
};

// Runs fn() in an isolated child process so static option maps (Pheno::options,
// Marker::options, Geno::options, Covar::options, Logger) start fresh and
// Logger::e() exit(1) does not abort the parent test runner.
inline int RunIsolated(const std::function<void()>& fn) {
    std::cout.flush();
    std::cerr.flush();
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork() failed");
    }
    if (pid == 0) {
        ::testing::CurrentTestFailures() = 0;
        try {
            fn();
        } catch (const ::testing::FatalAssertionFailure&) {
            _exit(2);
        } catch (const std::exception& ex) {
            std::cerr << "Child exception: " << ex.what() << std::endl;
            _exit(3);
        } catch (...) {
            _exit(4);
        }
        _exit(::testing::CurrentTestFailures() == 0 ? 0 : 1);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {}
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 1);
}

struct CliResult {
    int exit_code = -1;
    std::string output;
};

inline CliResult RunGctaCli(const std::vector<std::string>& args) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        throw std::runtime_error("pipe() failed");
    }
    std::cout.flush();
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        throw std::runtime_error("fork() failed");
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        std::vector<char*> c_args;
        c_args.reserve(args.size() + 2);
        c_args.push_back(const_cast<char*>(GCTA_BIN_PATH.c_str()));
        for (const auto& a : args) {
            c_args.push_back(const_cast<char*>(a.c_str()));
        }
        c_args.push_back(nullptr);
        execv(GCTA_BIN_PATH.c_str(), c_args.data());
        _exit(127);
    }

    close(pipefd[1]);
    std::string out;
    char buf[4096];
    ssize_t n = 0;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<std::size_t>(n));
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {}
    CliResult res;
    res.output = std::move(out);
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        res.exit_code = 128 + WTERMSIG(status);
    }
    return res;
}

struct SampleSpec {
    std::string fid;
    std::string iid;
    int sex = 1;       // 1 = male, 2 = female, 0 = unknown
    double pheno = -9; // -9 = missing
};

struct SnpSpec {
    std::string chr = "1";
    std::string id;
    uint32_t pos = 1000;
    std::string a1 = "A"; // effect allele (dosage counts copies of a1: 0, 1, 2; -1 = missing)
    std::string a2 = "G"; // other/ref allele
};

// geno_m_by_n[m][i] in {0, 1, 2} (copies of SnpSpec::a1), or -1 for missing.
inline void WritePlinkBedDataset(const std::string& prefix,
                                 const std::vector<SampleSpec>& samples,
                                 const std::vector<SnpSpec>& snps,
                                 const std::vector<std::vector<int>>& geno_m_by_n) {
    const std::size_t N = samples.size();
    const std::size_t M = snps.size();

    {
        std::ofstream fam(prefix + ".fam");
        for (const auto& s : samples) {
            fam << s.fid << "\t" << s.iid << "\t0\t0\t" << s.sex << "\t" << s.pheno << "\n";
        }
    }
    {
        std::ofstream bim(prefix + ".bim");
        for (const auto& v : snps) {
            bim << v.chr << "\t" << v.id << "\t0\t" << v.pos << "\t" << v.a1 << "\t" << v.a2 << "\n";
        }
    }
    {
        std::ofstream bed(prefix + ".bed", std::ios::binary);
        const uint8_t hdr[3] = {0x6c, 0x1b, 0x01};
        bed.write(reinterpret_cast<const char*>(hdr), 3);
        const std::size_t bytes_per_snp = (N + 3) / 4;
        std::vector<uint8_t> row(bytes_per_snp, 0);
        for (std::size_t m = 0; m < M; ++m) {
            std::fill(row.begin(), row.end(), 0);
            for (std::size_t b = 0; b < bytes_per_snp; ++b) {
                uint8_t byte_val = 0;
                for (std::size_t k = 0; k < 4; ++k) {
                    const std::size_t idx = b * 4 + k;
                    uint8_t code = 0x03; // pad unused trailing slots
                    if (idx < N) {
                        const int g = geno_m_by_n[m][idx];
                        if (g == 2) code = 0x00;      // hom A1
                        else if (g == 1) code = 0x02; // het
                        else if (g == 0) code = 0x03; // hom A2
                        else code = 0x01;             // missing
                    }
                    byte_val |= static_cast<uint8_t>(code << (2 * k));
                }
                row[b] = byte_val;
            }
            bed.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        }
    }
}

// Writes PLINK2 mode 0x02 .pgen + .pvar + .psam with identical allele semantics
// (ALT = SnpSpec::a1, REF = SnpSpec::a2; geno in {0, 1, 2} copies of ALT, -1 missing).
inline void WritePlink2PgenDataset(const std::string& prefix,
                                   const std::vector<SampleSpec>& samples,
                                   const std::vector<SnpSpec>& snps,
                                   const std::vector<std::vector<int>>& geno_m_by_n) {
    const uint32_t N = static_cast<uint32_t>(samples.size());
    const uint32_t M = static_cast<uint32_t>(snps.size());

    {
        std::ofstream psam(prefix + ".psam");
        psam << "#FID\tIID\tSEX\n";
        for (const auto& s : samples) {
            psam << s.fid << "\t" << s.iid << "\t" << s.sex << "\n";
        }
    }
    {
        std::ofstream pvar(prefix + ".pvar");
        pvar << "#CHROM\tPOS\tID\tREF\tALT\n";
        for (const auto& v : snps) {
            pvar << v.chr << "\t" << v.pos << "\t" << v.id << "\t" << v.a2 << "\t" << v.a1 << "\n";
        }
    }
    {
        std::ofstream pgen(prefix + ".pgen", std::ios::binary);
        uint8_t hdr[12];
        hdr[0] = 0x6c;
        hdr[1] = 0x1b;
        hdr[2] = 0x02; // basic fixed-width hardcall mode
        std::memcpy(hdr + 3, &M, 4);
        std::memcpy(hdr + 7, &N, 4);
        hdr[11] = 0x00;
        pgen.write(reinterpret_cast<const char*>(hdr), 12);

        const std::size_t bytes_per_snp = (N + 3) / 4;
        std::vector<uint8_t> row(bytes_per_snp, 0);
        for (uint32_t m = 0; m < M; ++m) {
            std::fill(row.begin(), row.end(), 0);
            for (std::size_t b = 0; b < bytes_per_snp; ++b) {
                uint8_t byte_val = 0;
                for (std::size_t k = 0; k < 4; ++k) {
                    const std::size_t idx = b * 4 + k;
                    uint8_t code = 0x00;
                    if (idx < N) {
                        const int g = geno_m_by_n[m][idx];
                        if (g == 0) code = 0x00;      // hom REF (A2)
                        else if (g == 1) code = 0x01; // het
                        else if (g == 2) code = 0x02; // hom ALT (A1)
                        else code = 0x03;             // missing
                    }
                    byte_val |= static_cast<uint8_t>(code << (2 * k));
                }
                row[b] = byte_val;
            }
            pgen.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        }
    }
}

namespace detail {
inline void AppendU16(std::vector<uint8_t>& out, uint16_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + 2);
}
inline void AppendU32(std::vector<uint8_t>& out, uint32_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + 4);
}
inline void AppendStr16(std::vector<uint8_t>& out, const std::string& s) {
    AppendU16(out, static_cast<uint16_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}
inline void AppendStr32(std::vector<uint8_t>& out, const std::string& s) {
    AppendU32(out, static_cast<uint32_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}
} // namespace detail

// Writes Oxford BGEN v1.2 (zlib-compressed, 8-bit unphased probabilities) + .sample + .bgen.bgi SQLite index.
inline void WriteBgenV12Dataset(const std::string& prefix,
                                const std::vector<SampleSpec>& samples,
                                const std::vector<SnpSpec>& snps,
                                const std::vector<std::vector<int>>& geno_m_by_n) {
    const uint32_t N = static_cast<uint32_t>(samples.size());
    const uint32_t M = static_cast<uint32_t>(snps.size());
    const std::string bgen_path = prefix + ".bgen";
    const std::string sample_path = prefix + ".sample";
    const std::string bgi_path = bgen_path + ".bgi";

    {
        std::ofstream sf(sample_path);
        sf << "ID_1 ID_2 missing sex\n";
        sf << "0 0 0 D\n";
        for (const auto& s : samples) {
            sf << s.fid << " " << s.iid << " 0 " << s.sex << "\n";
        }
    }

    std::vector<uint8_t> file_bytes;
    detail::AppendU32(file_bytes, 20); // offset to first variant data block (after 4-byte offset field)
    detail::AppendU32(file_bytes, 20); // header length = 20 bytes
    detail::AppendU32(file_bytes, M);
    detail::AppendU32(file_bytes, N);
    file_bytes.insert(file_bytes.end(), {'b', 'g', 'e', 'n'});
    // flags: compress=1 (zlib), layout=2 (v1.2), has_sample=0 -> (2 << 2) | 1 = 9
    detail::AppendU32(file_bytes, (2u << 2) | 1u);

    struct BgiRow {
        std::string chr;
        uint32_t pos;
        std::string rsid;
        std::string a1;
        std::string a2;
        uint64_t start_pos;
        uint64_t size_bytes;
    };
    std::vector<BgiRow> bgi_rows;
    bgi_rows.reserve(M);

    for (uint32_t m = 0; m < M; ++m) {
        const uint64_t var_start = file_bytes.size();
        const auto& v = snps[m];
        detail::AppendStr16(file_bytes, v.id);  // variant id
        detail::AppendStr16(file_bytes, v.id);  // rsid
        detail::AppendStr16(file_bytes, v.chr); // chr
        detail::AppendU32(file_bytes, v.pos);   // pos
        detail::AppendU16(file_bytes, 2);       // K = 2 alleles
        detail::AppendStr32(file_bytes, v.a1);  // allele1 (effect allele)
        detail::AppendStr32(file_bytes, v.a2);  // allele2

        // Build uncompressed probability payload (BGEN v1.2 layout=2, B=8 bits)
        // Note: GCTA reads uint64_t via memcpy from &X_prob[start_bits/8], so pad 8 trailing 0 bytes in decomp buffer
        std::vector<uint8_t> raw_payload;
        detail::AppendU32(raw_payload, N);
        detail::AppendU16(raw_payload, 2); // 2 alleles
        raw_payload.push_back(2);          // min ploidy = 2
        raw_payload.push_back(2);          // max ploidy = 2
        for (uint32_t i = 0; i < N; ++i) {
            const int g = geno_m_by_n[m][i];
            if (g < 0) {
                raw_payload.push_back(static_cast<uint8_t>(0x80u | 2u)); // missing flag + ploidy 2
            } else {
                raw_payload.push_back(2u);
            }
        }
        raw_payload.push_back(0); // unphased
        raw_payload.push_back(8); // 8 bits per probability (2 bytes per diploid sample: P(A1A1), P(A1A2))
        for (uint32_t i = 0; i < N; ++i) {
            const int g = geno_m_by_n[m][i];
            if (g == 2) {
                raw_payload.push_back(255); // P(A1A1) = 1
                raw_payload.push_back(0);   // P(A1A2) = 0
            } else if (g == 1) {
                raw_payload.push_back(0);   // P(A1A1) = 0
                raw_payload.push_back(255); // P(A1A2) = 1
            } else {
                raw_payload.push_back(0);   // P(A1A1) = 0
                raw_payload.push_back(0);   // P(A1A2) = 0 -> P(A2A2) = 1 (or missing)
            }
        }

        const uint32_t D = static_cast<uint32_t>(raw_payload.size());
        uLongf comp_bound = compressBound(D);
        std::vector<uint8_t> comp_buf(comp_bound);
        if (compress2(comp_buf.data(), &comp_bound, raw_payload.data(), D, Z_DEFAULT_COMPRESSION) != Z_OK) {
            throw std::runtime_error("zlib compress2 failed while building test BGEN");
        }
        const uint32_t C = static_cast<uint32_t>(comp_bound) + 4u;
        detail::AppendU32(file_bytes, C);
        detail::AppendU32(file_bytes, D);
        file_bytes.insert(file_bytes.end(), comp_buf.begin(), comp_buf.begin() + comp_bound);

        const uint64_t var_size = file_bytes.size() - var_start;
        bgi_rows.push_back({v.chr, v.pos, v.id, v.a1, v.a2, var_start, var_size});
    }

    {
        std::ofstream bf(bgen_path, std::ios::binary);
        bf.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
    }

    unlink(bgi_path.c_str());
    sqlite3* db = nullptr;
    if (sqlite3_open(bgi_path.c_str(), &db) != SQLITE_OK) {
        throw std::runtime_error("sqlite3_open failed for .bgi");
    }
    const char* ddl =
        "CREATE TABLE Metadata (filename TEXT, file_size INT, last_write_time INT, first_1000_bytes BLOB, index_creation_time INT);"
        "CREATE TABLE Variant (chromosome TEXT, position INT, rsid TEXT, number_of_alleles INT, allele1 TEXT, allele2 TEXT, file_start_position INT, size_in_bytes INT);";
    if (sqlite3_exec(db, ddl, nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        throw std::runtime_error("sqlite3_exec DDL failed");
    }

    sqlite3_stmt* meta_stmt = nullptr;
    sqlite3_prepare_v2(db, "INSERT INTO Metadata VALUES (?1, ?2, 0, ?3, 0);", -1, &meta_stmt, nullptr);
    sqlite3_bind_text(meta_stmt, 1, bgen_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(meta_stmt, 2, static_cast<sqlite3_int64>(file_bytes.size()));
    const int first_n = static_cast<int>(std::min<std::size_t>(1000, file_bytes.size()));
    sqlite3_bind_blob(meta_stmt, 3, file_bytes.data(), first_n, SQLITE_TRANSIENT);
    sqlite3_step(meta_stmt);
    sqlite3_finalize(meta_stmt);

    sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
    sqlite3_stmt* var_stmt = nullptr;
    sqlite3_prepare_v2(db, "INSERT INTO Variant VALUES (?1, ?2, ?3, 2, ?4, ?5, ?6, ?7);", -1, &var_stmt, nullptr);
    for (const auto& r : bgi_rows) {
        sqlite3_reset(var_stmt);
        sqlite3_bind_text(var_stmt, 1, r.chr.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(var_stmt, 2, static_cast<int>(r.pos));
        sqlite3_bind_text(var_stmt, 3, r.rsid.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(var_stmt, 4, r.a1.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(var_stmt, 5, r.a2.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(var_stmt, 6, static_cast<sqlite3_int64>(r.start_pos));
        sqlite3_bind_int64(var_stmt, 7, static_cast<sqlite3_int64>(r.size_bytes));
        sqlite3_step(var_stmt);
    }
    sqlite3_finalize(var_stmt);
    sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
    sqlite3_close(db);
}

inline std::vector<float> ReadBinaryFloats(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open float file: " + path);
    const std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<float> vals(static_cast<std::size_t>(bytes) / sizeof(float));
    if (!vals.empty()) {
        in.read(reinterpret_cast<char*>(vals.data()), bytes);
    }
    return vals;
}

inline std::vector<std::string> ReadLines(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open text file: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

inline std::vector<std::string> SplitTabsOrSpaces(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> toks;
    std::string tok;
    while (iss >> tok) toks.push_back(tok);
    return toks;
}

} // namespace gcta_test

#endif // GCTA_TEST_HELPERS_H
