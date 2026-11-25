/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * Implementations of functions for estimating the genetic relationship matrix
 *
 * 2010 by Jian Yang <jian.yang.qt@gmail.com>
 *
 * This file is distributed under the GNU General Public
 * License, Version 3.  Please see the file LICENSE for more
 * details
 */

#include "gcta.h"
#include "cpu_f77blas.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <omp.h>
#include <unordered_set>
#include <zlib.h>

#if GCTA_CPU_ARM || defined(OPENBLAS_CONST)
extern "C" {
int openblas_get_num_threads(void);
void openblas_set_num_threads(int num_threads);
}
#endif

namespace {

static inline uint32_t popcount_u64(uint64_t x) {
#if defined(_MSC_VER) && defined(_WIN64)
    return static_cast<uint32_t>(__popcnt64(x));
#elif defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt(static_cast<uint32_t>(x)) + __popcnt(static_cast<uint32_t>(x >> 32)));
#else
    return static_cast<uint32_t>(__builtin_popcountll(x));
#endif
}

struct ScopedSingleThreadBLASV1 {
    int prev_omp;
#if GCTA_CPU_ARM || defined(OPENBLAS_CONST)
    int prev_threads;
    ScopedSingleThreadBLASV1()
        : prev_omp(omp_get_max_threads()),
          prev_threads(openblas_get_num_threads()) {
        openblas_set_num_threads(1);
        omp_set_num_threads(prev_omp);
    }
    ~ScopedSingleThreadBLASV1() {
        openblas_set_num_threads(prev_threads > 0 ? prev_threads : prev_omp);
        omp_set_num_threads(prev_omp);
    }
#else
    ScopedSingleThreadBLASV1() : prev_omp(omp_get_max_threads()) {}
    ~ScopedSingleThreadBLASV1() {}
#endif
    struct ThreadLocal {
        ThreadLocal() {
#if GCTA_CPU_x86
            mkl_set_num_threads_local(1);
#endif
        }
        ~ThreadLocal() {
#if GCTA_CPU_x86
            mkl_set_num_threads_local(0);
#endif
        }
    };
};

// 2D cache-tiled OpenMP lower-triangular C = A * A^T (column-major float, n x k)
static void parallelTiledSyrkFloatLower(int n, int k, const float *A, float *C) {
    if (n <= 0 || k <= 0) return;
    ScopedSingleThreadBLASV1 blas_guard;
    char uplo = 'L', notrans = 'N', trans = 'T';
    float alpha = 1.0f, beta = 0.0f;
    const int nthreads = omp_get_max_threads();
    if (nthreads <= 1 || n <= 256) {
        ScopedSingleThreadBLASV1::ThreadLocal tl_guard;
#if GCTA_CPU_x86
        ssyrk(&uplo, &notrans, &n, &k, &alpha, A, &n, &beta, C, &n);
#else
        ssyrk_(&uplo, &notrans, &n, &k, &alpha, const_cast<float *>(A), &n, &beta, C, &n);
#endif
        return;
    }

    int nb = static_cast<int>(std::sqrt((0.5 * static_cast<double>(n) * static_cast<double>(n)) / (8.0 * nthreads)));
    nb = ((nb + 32) / 64) * 64;
    if (nb < 128) nb = 128;
    if (nb > 768) nb = 768;

    const int nblk = (n + nb - 1) / nb;
    const int ntiles = nblk * (nblk + 1) / 2;

    #pragma omp parallel
    {
        ScopedSingleThreadBLASV1::ThreadLocal tl_guard;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntiles; ++t) {
            int bi = static_cast<int>((std::sqrt(8.0 * static_cast<double>(t) + 1.0) - 1.0) * 0.5);
            while ((bi + 1) * (bi + 2) / 2 <= t) ++bi;
            while (bi * (bi + 1) / 2 > t) --bi;
            const int bj = t - bi * (bi + 1) / 2;

            const int i0 = bi * nb;
            const int j0 = bj * nb;
            int tm = std::min(nb, n - i0);
            int tn = std::min(nb, n - j0);
            float *c_ptr = C + static_cast<size_t>(j0) * static_cast<size_t>(n) + i0;
            if (bi == bj) {
#if GCTA_CPU_x86
                ssyrk(&uplo, &notrans, &tm, &k, &alpha, A + i0, &n, &beta, c_ptr, &n);
#else
                ssyrk_(&uplo, &notrans, &tm, &k, &alpha, const_cast<float *>(A + i0), &n, &beta, c_ptr, &n);
#endif
            } else {
#if GCTA_CPU_x86
                sgemm(&notrans, &trans, &tm, &tn, &k, &alpha, A + i0, &n, A + j0, &n, &beta, c_ptr, &n);
#else
                sgemm_(&notrans, &trans, &tm, &tn, &k, &alpha, const_cast<float *>(A + i0), &n, const_cast<float *>(A + j0), &n, &beta, c_ptr, &n);
#endif
            }
        }
    }
}

} // namespace

void gcta::enable_grm_bin_flag() {
    _grm_bin_flag = true;
}

void gcta::check_autosome() {
    for (int i = 0; i < _include.size(); i++) {
        if (_chr[_include[i]] > _autosome_num) LOGGER.e(0, "this option is for the autosomal SNPs only. Please check the option --autosome.");
    }
}

void gcta::check_chrX() {
    for (int i = 0; i < _include.size(); i++) {
        if (_chr[_include[i]] != (_autosome_num + 1)) LOGGER.e(0, "this option is for SNPs on the X chromosome only.");
    }
}

void gcta::check_sex() {
    for (int i = 0; i < _keep.size(); i++) {
        if (_sex[_keep[i]] != 1 && _sex[_keep[i]] != 2) LOGGER.e(0, "sex information of the individual \"" + _fid[_keep[i]] + " " + _pid[_keep[i]] + "\" is missing.\nUse --update-sex option to update the sex information of the individuals.");
    }
}

void gcta::make_grm(bool grm_d_flag, bool grm_xchr_flag, bool inbred, bool output_bin, int grm_mtd, bool mlmassoc, bool diag_f3_flag, string subpopu_file)
{
    bool have_mis = false;

    if (!grm_d_flag && grm_xchr_flag) check_chrX();
    else check_autosome();

    unsigned long i = 0, j = 0, k = 0, n = _keep.size(), m = _include.size();
    if(grm_d_flag) have_mis = make_XMat_d(_geno);
    else  have_mis = make_XMat(_geno);

    eigenVector sd_SNP;
    if (grm_mtd == 0) {
        if (grm_d_flag) std_XMat_d(_geno, sd_SNP, false, true);
        else{
            if(subpopu_file.empty()) std_XMat(_geno, sd_SNP, grm_xchr_flag, false, true);
            else std_XMat_subpopu(subpopu_file, _geno, sd_SNP, grm_xchr_flag, false, true);
        }
    } 
    else {
        if (grm_d_flag) std_XMat_d(_geno, sd_SNP, false, false);
        else std_XMat(_geno, sd_SNP, grm_xchr_flag, false, false);
    }

    if (!mlmassoc) LOGGER << "\nCalculating the" << ((grm_d_flag) ? " dominance" : "") << " genetic relationship matrix (GRM)" << (grm_xchr_flag ? " for the X chromosome" : "") << (_dosage_flag ? " using imputed dosage data" : "") << " ... (Note: default speed-optimized mode, may use huge RAM)" << endl;
    else LOGGER << "\nCalculating the genetic relationship matrix (GRM) ... " << endl;

    // count the number of missing genotypes
    vector< vector<int> > miss_pos;
    const size_t words_m = (m + 63) / 64;
    vector<uint64_t> obs_bits;
    if(have_mis){
        miss_pos.resize(n);
        obs_bits.assign(n * words_m, 0ULL);
        #pragma omp parallel for schedule(static)
        for (long ii = 0; ii < (long)n; ii++) {
            uint64_t *row_obs = obs_bits.data() + (size_t)ii * words_m;
            for (unsigned long jj = 0; jj < m; jj++) {
                if (_geno(ii, jj) < 1e5f) {
                    row_obs[jj >> 6] |= (1ULL << (jj & 63));
                } else {
                    _geno(ii, jj) = 0.0f;
                    miss_pos[ii].push_back((int)jj);
                }
            }
        }
    }

    // Calculate A_N matrix
    if(have_mis){
        _grm_N.resize(n, n);
        const size_t sparse_cut = m / 32 + 64;
        #pragma omp parallel for schedule(dynamic, 16)
        for (long ii = 0; ii < (long)n; ii++) {
            const uint64_t *obs_i = obs_bits.data() + (size_t)ii * words_m;
            const size_t miss_i = miss_pos[ii].size();
            for (long jj = 0; jj <= ii; jj++) {
                const vector<int> &mp_j = miss_pos[jj];
                float n_pair = 0.0f;
                if (mp_j.size() <= sparse_cut) {
                    int miss_j = 0;
                    for (size_t kk = 0; kk < mp_j.size(); kk++) {
                        const uint32_t col = static_cast<uint32_t>(mp_j[kk]);
                        miss_j += static_cast<int>((obs_i[col >> 6] >> (col & 63)) & 1ULL);
                    }
                    n_pair = static_cast<float>(m - miss_i - static_cast<size_t>(miss_j));
                } else {
                    const uint64_t *obs_j = obs_bits.data() + (size_t)jj * words_m;
                    uint32_t common_obs = 0;
                    for (size_t w = 0; w < words_m; w++) {
                        common_obs += popcount_u64(obs_i[w] & obs_j[w]);
                    }
                    n_pair = static_cast<float>(common_obs);
                }
                _grm_N(ii, jj) = n_pair;
                _grm_N(jj, ii) = n_pair;
            }
        }
        obs_bits.clear();
        obs_bits.shrink_to_fit();
    }
    else _grm_N = MatrixXf::Constant(n, n, m);

    // Calculate WW' via symmetric rank-k update (lower triangle only -> 50% fewer FLOPs)
    if (grm_mtd == 1) _grm_N = _grm_N.array() * sd_SNP.mean();

    #ifdef SINGLE_PRECISION
    if (n > 0 && m > 0) {
        _grm.resize(n, n);
        parallelTiledSyrkFloatLower(static_cast<int>(n), static_cast<int>(m), _geno.data(), _grm.data());
    } else {
        _grm.setZero(n, n);
    }
    #pragma omp parallel for schedule(dynamic, 32)
    for (long ii = 0; ii < (long)n; ii++) {
        for (long jj = 0; jj <= ii; jj++) {
            if (_grm_N(ii, jj) > 0.0f) _grm(ii, jj) /= _grm_N(ii, jj);
            else _grm(ii, jj) = 0.0f;
            _grm(jj, ii) = _grm(ii, jj);
        }
    }
    #else
    {
        MatrixXf grm_f;
        if (n > 0 && m > 0) {
            grm_f.resize(n, n);
            parallelTiledSyrkFloatLower(static_cast<int>(n), static_cast<int>(m), _geno.data(), grm_f.data());
        } else {
            grm_f = MatrixXf::Zero(n, n);
        }
        _grm.resize(n, n);
        #pragma omp parallel for schedule(dynamic, 32)
        for (long ii = 0; ii < (long)n; ii++) {
            for (long jj = 0; jj <= ii; jj++) {
                const double denom = static_cast<double>(_grm_N(ii, jj));
                const double val = (denom > 0.0) ? (static_cast<double>(grm_f(ii, jj)) / denom) : 0.0;
                _grm(ii, jj) = val;
                _grm(jj, ii) = val;
            }
        }
    }
    #endif

    // GRM summary
    double diag_m = 0.0, diag_v = 0.0, off_m = 0.0, off_v = 0.0;
    calcu_grm_var(diag_m, diag_v, off_m, off_v);
    LOGGER<<"\nSummary of the GRM:" << endl;
    LOGGER<<"Mean of diagonals = "<<diag_m<<endl;
    LOGGER<<"Variance of diagonals = "<<diag_v<<endl;
    LOGGER<<"Mean of off-diagonals = " << off_m <<endl;
    LOGGER<<"Variance of off-diagonals = " << off_v <<endl;

    // re-calcuate the diagonals (Fhat3+1)
    if (diag_f3_flag) {
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            _grm(i,i) = 0.0;
            double non_missing = 0.0;
            for (j = 0; j < m; j++) {
                if (_geno(i,j) < 1e5){
                    _grm(i,i) += _geno(i,j)*(_geno(i,j)+(_mu[_include[j]] - 1.0) * sd_SNP[j]);
                    non_missing += 1.0;
                } 
            }
            _grm(i,i) /= non_missing; 
        }
    }

    if (inbred) {
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) _grm(i,j) *= 0.5;
        }
    }

    if (mlmassoc && grm_mtd == 0) {
        for (j = 0; j < m; j++) {
            if (fabs(sd_SNP[j]) < 1.0e-50) sd_SNP[j] = 0.0;
            else sd_SNP[j] = 1.0 / sd_SNP[j];
        }
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                if (_geno(i,j) < 1e5) _geno(i,j) *= sd_SNP[j];
                else _geno(i,j) = 0.0;
            }
        }
    } 
    else {
        // Output A_N and A
        string out_buf = _out;
        if (grm_d_flag) _out += ".d";
        output_grm(output_bin);
        _out = out_buf;
    }
}

void gcta::calcu_grm_var(double &diag_m, double &diag_v, double &off_m, double &off_v)
{
    const long n = static_cast<long>(_keep.size());
    diag_m = _grm.diagonal().mean();
    diag_v = (_grm.diagonal().array() - diag_m).square().sum() / ((double)n - 1.0);
    const double off_num = 0.5 * n * (n - 1.0);
    double sum_off = 0.0;
    #pragma omp parallel for reduction(+:sum_off) schedule(dynamic, 64)
    for (long c = 0; c < n - 1; c++) {
        double col_sum = 0.0;
        for (long r = c + 1; r < n; r++) {
            col_sum += _grm(r, c);
        }
        sum_off += col_sum;
    }
    off_m = sum_off / off_num;

    double sq_off = 0.0;
    #pragma omp parallel for reduction(+:sq_off) schedule(dynamic, 64)
    for (long c = 0; c < n - 1; c++) {
        double col_sq = 0.0;
        for (long r = c + 1; r < n; r++) {
            const double diff = _grm(r, c) - off_m;
            col_sq += diff * diff;
        }
        sq_off += col_sq;
    }
    off_v = sq_off / (off_num - 1.0);
}

void gcta::output_grm(bool output_grm_bin)
{
    const long n_keep = static_cast<long>(_keep.size());
    string grm_file;
    if (output_grm_bin) {
        // Save matrix A in binary file using buffered stripe writes
        grm_file = _out + ".grm.bin";
        FILE *A_Bin = fopen(grm_file.c_str(), "wb");
        if (!A_Bin) LOGGER.e(0, "cannot open the file [" + grm_file + "] to write.");
        setvbuf(A_Bin, NULL, _IOFBF, 4 * 1024 * 1024);

        const long kStripeRows = 256;
        vector<float> stripe_buf(static_cast<size_t>(kStripeRows) * static_cast<size_t>(std::max<long>(1, n_keep)));
        vector<size_t> row_off(kStripeRows + 1, 0);

        for (long r0 = 0; r0 < n_keep; r0 += kStripeRows) {
            const long r_cnt = std::min<long>(kStripeRows, n_keep - r0);
            row_off[0] = 0;
            for (long dr = 0; dr < r_cnt; dr++) {
                row_off[dr + 1] = row_off[dr] + static_cast<size_t>(r0 + dr + 1);
            }
            #pragma omp parallel for schedule(static)
            for (long dr = 0; dr < r_cnt; dr++) {
                const long r = r0 + dr;
                float *dst = stripe_buf.data() + row_off[dr];
                for (long c = 0; c <= r; c++) {
                    dst[c] = static_cast<float>(_grm(r, c));
                }
            }
            const size_t n_elems = row_off[r_cnt];
            if (n_elems > 0 && fwrite(stripe_buf.data(), sizeof(float), n_elems, A_Bin) != n_elems) {
                fclose(A_Bin);
                LOGGER.e(0, "Failed to write to [" + grm_file + "].");
            }
        }
        fclose(A_Bin);
        LOGGER << "GRM of " << n_keep << " individuals has been saved in the file [" + grm_file + "] (in binary format)." << endl;

        string grm_N_file = _out + ".grm.N.bin";
        FILE *N_Bin = fopen(grm_N_file.c_str(), "wb");
        if (!N_Bin) LOGGER.e(0, "cannot open the file [" + grm_N_file + "] to write.");
        setvbuf(N_Bin, NULL, _IOFBF, 4 * 1024 * 1024);

        for (long r0 = 0; r0 < n_keep; r0 += kStripeRows) {
            const long r_cnt = std::min<long>(kStripeRows, n_keep - r0);
            row_off[0] = 0;
            for (long dr = 0; dr < r_cnt; dr++) {
                row_off[dr + 1] = row_off[dr] + static_cast<size_t>(r0 + dr + 1);
            }
            #pragma omp parallel for schedule(static)
            for (long dr = 0; dr < r_cnt; dr++) {
                const long r = r0 + dr;
                float *dst = stripe_buf.data() + row_off[dr];
                for (long c = 0; c <= r; c++) {
                    dst[c] = static_cast<float>(_grm_N(r, c));
                }
            }
            const size_t n_elems = row_off[r_cnt];
            if (n_elems > 0 && fwrite(stripe_buf.data(), sizeof(float), n_elems, N_Bin) != n_elems) {
                fclose(N_Bin);
                LOGGER.e(0, "Failed to write to [" + grm_N_file + "].");
            }
        }
        fclose(N_Bin);
        LOGGER << "Number of SNPs to calculate the genetic relationship between each pair of individuals has been saved in the file [" + grm_N_file + "] (in binary format)." << endl;
    }
    else {
        // Save A matrix in RFC-1952 multi-member gzip compressed text format (parallel pigz-style blocks)
        grm_file = _out + ".grm.gz";
        FILE *gz_fp = fopen(grm_file.c_str(), "wb");
        if (!gz_fp) LOGGER.e(0, "cannot open the file [" + grm_file + "] to write.");
        setvbuf(gz_fp, NULL, _IOFBF, 4 * 1024 * 1024);
        LOGGER << "Saving the genetic relationship matrix to the file [" + grm_file + "] (in compressed text format)." << endl;

        auto compress_gzip_member = [](const char *src, size_t src_len, string &dst) -> bool {
            z_stream zs;
            std::memset(&zs, 0, sizeof(zs));
            if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
                return false;
            }
            const uLong bound = deflateBound(&zs, static_cast<uLong>(src_len)) + 32;
            dst.resize(bound);
            zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(src));
            zs.avail_in = static_cast<uInt>(src_len);
            zs.next_out = reinterpret_cast<Bytef *>(&dst[0]);
            zs.avail_out = static_cast<uInt>(dst.size());
            const int ret = deflate(&zs, Z_FINISH);
            const size_t out_bytes = zs.total_out;
            deflateEnd(&zs);
            if (ret != Z_STREAM_END) return false;
            dst.resize(out_bytes);
            return true;
        };

        const bool has_N = (_grm_N.rows() > 0);
        if (n_keep <= 0) {
            string empty_gz;
            if (!compress_gzip_member("", 0, empty_gz) ||
                fwrite(empty_gz.data(), 1, empty_gz.size(), gz_fp) != empty_gz.size()) {
                fclose(gz_fp);
                LOGGER.e(0, "Failed to write to [" + grm_file + "].");
            }
        } else {
            const long kBatchBlocks = std::max<long>(8, omp_get_max_threads() * 4);
            const long kRowsPerBlock = 16;
            vector<string> gz_blocks(kBatchBlocks);
            const long total_blocks = (n_keep + kRowsPerBlock - 1) / kRowsPerBlock;

            for (long b0 = 0; b0 < total_blocks; b0 += kBatchBlocks) {
                const long b_cnt = std::min<long>(kBatchBlocks, total_blocks - b0);
                bool ok = true;
                #pragma omp parallel for schedule(dynamic, 1) reduction(&:ok)
                for (long db = 0; db < b_cnt; db++) {
                    const long r_start = (b0 + db) * kRowsPerBlock;
                    const long r_end = std::min<long>(n_keep, r_start + kRowsPerBlock);
                    ostringstream oss;
                    oss.setf(ios::scientific);
                    oss.precision(6);
                    if (has_N) {
                        for (long r = r_start; r < r_end; r++) {
                            for (long c = 0; c <= r; c++) {
                                oss << (r + 1) << '\t' << (c + 1) << '\t' << _grm_N(r, c) << '\t' << _grm(r, c) << '\n';
                            }
                        }
                    } else {
                        for (long r = r_start; r < r_end; r++) {
                            for (long c = 0; c <= r; c++) {
                                oss << (r + 1) << '\t' << (c + 1) << "\t0\t" << _grm(r, c) << '\n';
                            }
                        }
                    }
                    const string raw_txt = oss.str();
                    if (!compress_gzip_member(raw_txt.data(), raw_txt.size(), gz_blocks[db])) {
                        ok = false;
                    }
                }
                if (!ok) {
                    fclose(gz_fp);
                    LOGGER.e(0, "Failed to compress [" + grm_file + "].");
                }
                for (long db = 0; db < b_cnt; db++) {
                    const string &blk = gz_blocks[db];
                    if (!blk.empty() && fwrite(blk.data(), 1, blk.size(), gz_fp) != blk.size()) {
                        fclose(gz_fp);
                        LOGGER.e(0, "Failed to write to [" + grm_file + "].");
                    }
                }
            }
        }
        fclose(gz_fp);
        LOGGER << "The genetic relationship matrix has been saved in the file [" + grm_file + "] (in compressed text format)." << endl;
    }

    string famfile = _out + ".grm.id";
    ofstream Fam(famfile.c_str());
    if (!Fam) LOGGER.e(0, "cannot open the file [" + famfile + "] to write.");
    for (long i = 0; i < n_keep; i++) Fam << _fid[_keep[i]] << '\t' << _pid[_keep[i]] << '\n';
    Fam.close();
    LOGGER << "IDs for the GRM file [" + grm_file + "] have been saved in the file [" + famfile + "]." << endl;
}

int gcta::read_grm_id(string grm_file, vector<string> &grm_id, bool out_id_log, bool read_id_only)
{
    // read GRM IDs
    string grm_id_file = grm_file + ".grm.id";
    if (out_id_log) LOGGER << "Reading IDs of the GRM from [" + grm_id_file + "]." << endl;
    ifstream i_grm_id(grm_id_file.c_str());
    if (!i_grm_id) LOGGER.e(0, "cannot open the file [" + grm_id_file + "] to read.");
    string str_buf, id_buf;
    vector<string> fid, pid;
    grm_id.clear();
    while (i_grm_id) {
        i_grm_id >> str_buf;
        if (i_grm_id.eof()) break;
        fid.push_back(str_buf);
        id_buf = str_buf + ":";
        i_grm_id >> str_buf;
        pid.push_back(str_buf);
        id_buf += str_buf;
        grm_id.push_back(id_buf);
        getline(i_grm_id, str_buf);
    }
    i_grm_id.close();
    int n = grm_id.size();
    if (out_id_log) LOGGER << n << " IDs are read from [" + grm_id_file + "]." << endl;

    if (_id_map.empty()) {
        _fid = fid;
        _pid = pid;
        _indi_num = _fid.size();
        _sex.resize(_fid.size());
        init_keep();
    }

    return (n);
}

void gcta::read_grm(string grm_file, vector<string> &grm_id, bool out_id_log, bool read_id_only, bool dont_read_N)
{
    if (_grm_bin_flag) read_grm_bin(grm_file, grm_id, out_id_log, read_id_only, dont_read_N);
    else read_grm_gz(grm_file, grm_id, out_id_log, read_id_only);
}

void gcta::read_grm_gz(string grm_file, vector<string> &grm_id, bool out_id_log, bool read_id_only) {
    long n = read_grm_id(grm_file, grm_id, out_id_log, read_id_only);

    if (read_id_only) return;

    string grm_gzfile = grm_file + ".grm.gz", str_buf;
    const int MAX_LINE_LENGTH = 1000;
    char buf[MAX_LINE_LENGTH];
    gzifstream zinf;
    zinf.open(grm_gzfile.c_str());
    if (!zinf.is_open()) LOGGER.e(0, "cannot open the file [" + grm_gzfile + "] to read.");

    long indx1 = 0, indx2 = 0, nline = 0;
    double grm_buf = 0.0, grm_N_buf;
    string errmsg = "failed to read [" + grm_gzfile + "]. The format of the GRM file has been changed?\nError occurs in line:\n";
    LOGGER << "Reading the GRM from [" + grm_gzfile + "]." << endl;
    _grm.resize(n, n);
    _grm_N.resize(n, n);
    while (1) {
        zinf.getline(buf, MAX_LINE_LENGTH, '\n');
        if (zinf.fail() || !zinf.good()) break;
        stringstream ss(buf);
        if (!(ss >> indx1)) LOGGER.e(0, errmsg + buf);
        if (!(ss >> indx2)) LOGGER.e(0, errmsg + buf);
        if (!(ss >> grm_N_buf)) LOGGER.e(0, errmsg + buf);
        if (!(ss >> grm_buf)) LOGGER.e(0, errmsg + buf);
        if (indx1 < indx2 || indx1 > n || indx2 > n) LOGGER.e(0, errmsg + buf);
        if (grm_N_buf == 0) LOGGER << "Warning: " << buf << endl;
        _grm_N(indx1 - 1, indx2 - 1) = _grm_N(indx2 - 1, indx1 - 1) = grm_N_buf;
        _grm(indx1 - 1, indx2 - 1) = _grm(indx2 - 1, indx1 - 1) = grm_buf;
        nline++;
        if (ss >> str_buf) LOGGER.e(0, errmsg + buf);
    }
    zinf.close();
    if (!_within_family && nline != (long) n * (n + 1)*0.5){
        stringstream errmsg_tmp;
        errmsg_tmp << "there are " << nline << " lines in the [" << grm_gzfile << "] file. The expected number of lines is " << (long) (n * (n + 1)*0.5) << "." << endl;
        LOGGER.e(0, errmsg_tmp.str());
        //LOGGER.e(0, "incorrect number of lines in the grm file. Are *.grm.gz file and *.grm.id file mismatched?");
    }
    LOGGER << "GRM for " << n << " individuals are included from [" + grm_gzfile + "]." << endl;
}

void gcta::read_grm_bin(string grm_file, vector<string> &grm_id, bool out_id_log, bool read_id_only, bool dont_read_N)
{
    const long n = static_cast<long>(read_grm_id(grm_file, grm_id, out_id_log, read_id_only));

    if (read_id_only) return;

    string grm_binfile = grm_file + ".grm.bin";
    FILE *A_bin = fopen(grm_binfile.c_str(), "rb");
    if (!A_bin) LOGGER.e(0, "cannot open the file [" + grm_binfile + "] to read.");
    setvbuf(A_bin, NULL, _IOFBF, 4 * 1024 * 1024);

    _grm.resize(n, n);
    LOGGER << "Reading the GRM from [" + grm_binfile + "]." << endl;
    const long kStripeRows = 256;
    vector<float> stripe_buf(static_cast<size_t>(kStripeRows) * static_cast<size_t>(std::max<long>(1, n)));
    vector<size_t> row_off(kStripeRows + 1, 0);

    for (long r0 = 0; r0 < n; r0 += kStripeRows) {
        const long r_cnt = std::min<long>(kStripeRows, n - r0);
        row_off[0] = 0;
        for (long dr = 0; dr < r_cnt; dr++) {
            row_off[dr + 1] = row_off[dr] + static_cast<size_t>(r0 + dr + 1);
        }
        const size_t n_elems = row_off[r_cnt];
        if (n_elems > 0 && fread(stripe_buf.data(), sizeof(float), n_elems, A_bin) != n_elems) {
            fclose(A_bin);
            LOGGER.e(0, "Is the size of the [" + grm_binfile + "] file incorrect?");
        }
        #pragma omp parallel for schedule(static)
        for (long dr = 0; dr < r_cnt; dr++) {
            const long r = r0 + dr;
            const float *src = stripe_buf.data() + row_off[dr];
            for (long c = 0; c <= r; c++) {
                const float val = src[c];
                _grm(r, c) = val;
                _grm(c, r) = val;
            }
        }
    }
    fclose(A_bin);

    if(!dont_read_N){
        string grm_Nfile = grm_file + ".grm.N.bin";
        FILE *N_bin = fopen(grm_Nfile.c_str(), "rb");
        if (!N_bin) LOGGER.e(0, "cannot open the file [" + grm_Nfile + "] to read.");
        setvbuf(N_bin, NULL, _IOFBF, 4 * 1024 * 1024);
        _grm_N.resize(n, n);
        LOGGER << "Reading the number of SNPs for the GRM from [" + grm_Nfile + "]." << endl;

        for (long r0 = 0; r0 < n; r0 += kStripeRows) {
            const long r_cnt = std::min<long>(kStripeRows, n - r0);
            row_off[0] = 0;
            for (long dr = 0; dr < r_cnt; dr++) {
                row_off[dr + 1] = row_off[dr] + static_cast<size_t>(r0 + dr + 1);
            }
            const size_t n_elems = row_off[r_cnt];
            if (n_elems > 0 && fread(stripe_buf.data(), sizeof(float), n_elems, N_bin) != n_elems) {
                fclose(N_bin);
                LOGGER.e(0, "Is the size of the [" + grm_Nfile + "] file incorrect?");
            }
            #pragma omp parallel for schedule(static)
            for (long dr = 0; dr < r_cnt; dr++) {
                const long r = r0 + dr;
                const float *src = stripe_buf.data() + row_off[dr];
                for (long c = 0; c <= r; c++) {
                    const float val = src[c];
                    _grm_N(r, c) = val;
                    _grm_N(c, r) = val;
                }
            }
        }
        fclose(N_bin);
    }

    LOGGER << "GRM for " << n << " individuals are included from [" + grm_binfile + "]." << endl;
}

void gcta::rm_cor_indi(double grm_cutoff) {
    LOGGER << "Pruning the GRM with a cutoff of " << grm_cutoff << " ..." << endl;

    const long n_keep = static_cast<long>(_keep.size());
    const int n_threads = std::max(1, omp_get_max_threads());
    vector<vector<pair<int, int>>> thread_pairs(n_threads);

    #pragma omp parallel num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        vector<pair<int, int>> &local_pairs = thread_pairs[tid];
        #pragma omp for schedule(dynamic, 32)
        for (long i = 0; i < n_keep; i++) {
            const int ki = _keep[i];
            for (long j = 0; j < i; j++) {
                const int kj = _keep[j];
                if (_grm(ki, kj) > grm_cutoff) {
                    local_pairs.push_back(std::make_pair(ki, kj));
                }
            }
        }
    }

    size_t total_pairs = 0;
    for (int t = 0; t < n_threads; t++) total_pairs += thread_pairs[t].size();
    vector<pair<int, int>> rel_pairs;
    rel_pairs.reserve(total_pairs);
    for (int t = 0; t < n_threads; t++) {
        rel_pairs.insert(rel_pairs.end(), thread_pairs[t].begin(), thread_pairs[t].end());
    }

    const size_t n_total_id = _fid.size();
    vector<int> deg(n_total_id, 0);
    for (size_t k = 0; k < rel_pairs.size(); k++) {
        deg[rel_pairs[k].first]++;
        deg[rel_pairs[k].second]++;
    }

    vector<uint8_t> removed_mask(n_total_id, 0);
    for (size_t k = 0; k < rel_pairs.size(); k++) {
        const int id1 = rel_pairs[k].first;
        const int id2 = rel_pairs[k].second;
        const int rm_id = (deg[id1] < deg[id2]) ? id2 : id1;
        removed_mask[rm_id] = 1;
    }

    vector<string> removed_ID;
    for (size_t id = 0; id < n_total_id; id++) {
        if (removed_mask[id]) {
            removed_ID.push_back(_fid[id] + ":" + _pid[id]);
        }
    }

    // update _keep and _id_map
    update_id_map_rm(removed_ID, _id_map, _keep);

    LOGGER << "After pruning the GRM, there are " << _keep.size() << " individuals (" << removed_ID.size() << " individuals removed)." << endl;
}

void gcta::adj_grm(double adj_grm_fac) {
    LOGGER << "Adjusting the GRM for sampling errors ..." << endl;
    const long n = static_cast<long>(_keep.size());
    double off_mean = 0.0, diag_mean = 0.0, off_var = 0.0, diag_var = 0.0;
    #pragma omp parallel for reduction(+:diag_mean, off_mean) schedule(dynamic, 32)
    for (long i = 0; i < n; i++) {
        const int ki = _keep[i];
        diag_mean += _grm(ki, ki);
        double row_off = 0.0;
        for (long j = 0; j < i; j++) row_off += _grm(ki, _keep[j]);
        off_mean += row_off;
    }
    diag_mean /= n;
    off_mean /= 0.5 * n * (n - 1.0);

    #pragma omp parallel for reduction(+:diag_var, off_var) schedule(dynamic, 32)
    for (long i = 0; i < n; i++) {
        const int ki = _keep[i];
        const double d_diag = _grm(ki, ki) - diag_mean;
        diag_var += d_diag * d_diag;
        double row_var = 0.0;
        for (long j = 0; j < i; j++) {
            const double d_off = _grm(ki, _keep[j]) - off_mean;
            row_var += d_off * d_off;
        }
        off_var += row_var;
    }
    diag_var /= n - 1.0;
    off_var /= 0.5 * n * (n - 1.0) - 1.0;

    #pragma omp parallel for schedule(dynamic, 32)
    for (long i = 0; i < n; i++) {
        const int ki = _keep[i];
        const double d_buf = 1.0 - (adj_grm_fac + 1.0 / _grm_N(ki, ki)) / diag_var;
        if (_grm(ki, ki) > 0) _grm(ki, ki) = 1.0 + d_buf * (_grm(ki, ki) - 1.0);
        for (long j = 0; j < i; j++) {
            const int kj = _keep[j];
            if (_grm_N(ki, kj) > 0) {
                _grm(ki, kj) *= 1.0 - (adj_grm_fac + 1.0 / _grm_N(ki, kj)) / off_var;
                _grm(kj, ki) = _grm(ki, kj);
            }
        }
    }
}

void gcta::dc(int dosage_compen) {
    LOGGER << "Parameterizing the GRM under the assumption of ";
    if (dosage_compen == 1) LOGGER << "full dosage compensation ..." << endl;
    else if (dosage_compen == 0) LOGGER << "no dosage compensation ..." << endl;

    int i = 0, j = 0, i_buf = 0;
    double c1 = 1.0, c2 = 1.0;
    if (dosage_compen == 1) {
        c1 = 2.0;
        c2 = sqrt(2.0);
    }// full dosage compensation
    else if (dosage_compen == 0) {
        c1 = 0.5;
        c2 = sqrt(0.5);
    } // on dosage compensation
    for (i = 0; i < _keep.size(); i++) {
        for (j = 0; j <= i; j++) {
            i_buf = _sex[_keep[i]] * _sex[_keep[j]];
            if (i_buf == 1) _grm(i, j) *= c1;
            else if (i_buf == 2) _grm(i, j) *= c2;
        }
    }
}

void gcta::manipulate_grm(string grm_file, string keep_indi_file, string remove_indi_file, string sex_file, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool merge_grm_flag, bool dont_read_N)
{
    int i = 0, j = 0;

    vector<string> grm_id;
    if (merge_grm_flag) merge_grm(grm_file);
    else read_grm(grm_file, grm_id, true, false, dont_read_N);

    if (!keep_indi_file.empty()) keep_indi(keep_indi_file);
    if (!remove_indi_file.empty()) remove_indi(remove_indi_file);
    if (grm_cutoff>-1.0) rm_cor_indi(grm_cutoff);
    if (!sex_file.empty()) update_sex(sex_file);
    if (adj_grm_fac>-1.0) adj_grm(adj_grm_fac);
    if (dosage_compen>-1) dc(dosage_compen);
    if (grm_cutoff>-1.0 || !keep_indi_file.empty() || !remove_indi_file.empty()) {
        const long nk = static_cast<long>(_keep.size());
        eigenMatrix grm_buf;
        grm_buf.swap(_grm);
        _grm.resize(nk, nk);
        #pragma omp parallel for schedule(dynamic, 32)
        for (long r = 0; r < nk; r++) {
            const int kr = _keep[r];
            for (long c = 0; c <= r; c++) {
                const int kc = _keep[c];
                const auto val = (kr >= kc) ? grm_buf(kr, kc) : grm_buf(kc, kr);
                _grm(r, c) = val;
                _grm(c, r) = val;
            }
        }
        grm_buf.resize(0, 0);
        if(!dont_read_N){
            MatrixXf grm_N_buf;
            grm_N_buf.swap(_grm_N);
            _grm_N.resize(nk, nk);
            #pragma omp parallel for schedule(dynamic, 32)
            for (long r = 0; r < nk; r++) {
                const int kr = _keep[r];
                for (long c = 0; c <= r; c++) {
                    const int kc = _keep[c];
                    const float val = (kr >= kc) ? grm_N_buf(kr, kc) : grm_N_buf(kc, kr);
                    _grm_N(r, c) = val;
                    _grm_N(c, r) = val;
                }
            }
        }
    }
}

void gcta::save_grm(string grm_file, string keep_indi_file, string remove_indi_file, string sex_file, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool merge_grm_flag, bool output_grm_bin) {
    if (dosage_compen>-1) check_sex();
    manipulate_grm(grm_file, keep_indi_file, remove_indi_file, sex_file, grm_cutoff, adj_grm_fac, dosage_compen, merge_grm_flag);
    output_grm(output_grm_bin);
}

void gcta::merge_grm(string merge_grm_file) {
    vector<string> grm_files, grm_id;
    read_grm_filenames(merge_grm_file, grm_files);

    int f = 0, i = 0;
    for (f = 0; f < (int)grm_files.size(); f++) {
        read_grm(grm_files[f], grm_id, false, true);
        update_id_map_kp(grm_id, _id_map, _keep);
    }
    vector<string> uni_id;
    uni_id.reserve(_keep.size());
    for (i = 0; i < (int)_keep.size(); i++) uni_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
    _n = uni_id.size();
    if (_n == 0) LOGGER.e(0, "no individual is in common among the GRM files.");
    else LOGGER << _n << " individuals in common in the GRM files." << endl;

    vector<int> kp;
    if (_grm_bin_flag) {
        _grm.setZero(_n, _n);
        eigenMatrix grm_N_acc = eigenMatrix::Zero(_n, _n);
        const long kStripeRows = 256;
        vector<float> buf_A;
        vector<float> buf_N;
        vector<size_t> row_off(kStripeRows + 1, 0);

        for (f = 0; f < (int)grm_files.size(); f++) {
            LOGGER << "Reading the GRM from the " << f + 1 << "th file ..." << endl;
            const long nf = static_cast<long>(read_grm_id(grm_files[f], grm_id, true, false));
            StrFunc::match(uni_id, grm_id, kp);

            bool is_identity = (nf == _n);
            bool injective = true;
            vector<int> inv_kp(nf, -1);
            for (long u = 0; u < _n; u++) {
                const int src_idx = kp[u];
                if (src_idx != u) is_identity = false;
                if (src_idx < 0 || src_idx >= nf || inv_kp[src_idx] != -1) {
                    injective = false;
                    is_identity = false;
                    break;
                }
                inv_kp[src_idx] = static_cast<int>(u);
            }

            if (injective) {
                string grm_binfile = grm_files[f] + ".grm.bin";
                string grm_Nfile = grm_files[f] + ".grm.N.bin";
                FILE *A_bin = fopen(grm_binfile.c_str(), "rb");
                if (!A_bin) LOGGER.e(0, "cannot open the file [" + grm_binfile + "] to read.");
                FILE *N_bin = fopen(grm_Nfile.c_str(), "rb");
                if (!N_bin) {
                    fclose(A_bin);
                    LOGGER.e(0, "cannot open the file [" + grm_Nfile + "] to read.");
                }
                setvbuf(A_bin, NULL, _IOFBF, 4 * 1024 * 1024);
                setvbuf(N_bin, NULL, _IOFBF, 4 * 1024 * 1024);
                LOGGER << "Reading the GRM from [" + grm_binfile + "]." << endl;
                LOGGER << "Reading the number of SNPs for the GRM from [" + grm_Nfile + "]." << endl;

                const size_t max_stripe = static_cast<size_t>(kStripeRows) * static_cast<size_t>(std::max<long>(1, nf));
                if (buf_A.size() < max_stripe) {
                    buf_A.resize(max_stripe);
                    buf_N.resize(max_stripe);
                }

                for (long r0 = 0; r0 < nf; r0 += kStripeRows) {
                    const long r_cnt = std::min<long>(kStripeRows, nf - r0);
                    row_off[0] = 0;
                    for (long dr = 0; dr < r_cnt; dr++) {
                        row_off[dr + 1] = row_off[dr] + static_cast<size_t>(r0 + dr + 1);
                    }
                    const size_t n_elems = row_off[r_cnt];
                    if (n_elems > 0 && fread(buf_A.data(), sizeof(float), n_elems, A_bin) != n_elems) {
                        fclose(A_bin);
                        fclose(N_bin);
                        LOGGER.e(0, "Is the size of the [" + grm_binfile + "] file incorrect?");
                    }
                    if (n_elems > 0 && fread(buf_N.data(), sizeof(float), n_elems, N_bin) != n_elems) {
                        fclose(A_bin);
                        fclose(N_bin);
                        LOGGER.e(0, "Is the size of the [" + grm_Nfile + "] file incorrect?");
                    }
                    if (is_identity) {
                        #pragma omp parallel for schedule(static)
                        for (long dr = 0; dr < r_cnt; dr++) {
                            const long r = r0 + dr;
                            const float *srcA = buf_A.data() + row_off[dr];
                            const float *srcN = buf_N.data() + row_off[dr];
                            for (long c = 0; c <= r; c++) {
                                const double vn = static_cast<double>(srcN[c]);
                                _grm(r, c) += static_cast<double>(srcA[c]) * vn;
                                grm_N_acc(r, c) += vn;
                            }
                        }
                    } else {
                        #pragma omp parallel for schedule(static)
                        for (long dr = 0; dr < r_cnt; dr++) {
                            const long src_r = r0 + dr;
                            const int dst_r = inv_kp[src_r];
                            if (dst_r < 0) continue;
                            const float *srcA = buf_A.data() + row_off[dr];
                            const float *srcN = buf_N.data() + row_off[dr];
                            for (long src_c = 0; src_c <= src_r; src_c++) {
                                const int dst_c = inv_kp[src_c];
                                if (dst_c < 0) continue;
                                const int hi = std::max(dst_r, dst_c);
                                const int lo = std::min(dst_r, dst_c);
                                const double vn = static_cast<double>(srcN[src_c]);
                                _grm(hi, lo) += static_cast<double>(srcA[src_c]) * vn;
                                grm_N_acc(hi, lo) += vn;
                            }
                        }
                    }
                }
                fclose(A_bin);
                fclose(N_bin);
                LOGGER << "GRM for " << nf << " individuals are included from [" + grm_binfile + "]." << endl;
            } else {
                eigenMatrix cur_acc;
                cur_acc.swap(_grm);
                read_grm(grm_files[f], grm_id);
                #pragma omp parallel for schedule(dynamic, 32)
                for (long c = 0; c < _n; c++) {
                    const int kc = kp[c];
                    for (long r = c; r < _n; r++) {
                        const int kr = kp[r];
                        const int hi = std::max(kr, kc);
                        const int lo = std::min(kr, kc);
                        const double vn = static_cast<double>(_grm_N(hi, lo));
                        cur_acc(r, c) += static_cast<double>(_grm(hi, lo)) * vn;
                        grm_N_acc(r, c) += vn;
                    }
                }
                _grm.swap(cur_acc);
            }
        }

        _grm_N.resize(_n, _n);
        #pragma omp parallel for schedule(dynamic, 32)
        for (long c = 0; c < _n; c++) {
            for (long r = c; r < _n; r++) {
                const double vn = grm_N_acc(r, c);
                const double val = (vn == 0.0) ? 0.0 : (_grm(r, c) / vn);
                _grm(r, c) = val;
                _grm(c, r) = val;
                const float vnf = static_cast<float>(vn);
                _grm_N(r, c) = vnf;
                _grm_N(c, r) = vnf;
            }
        }
        grm_N_acc.resize(0, 0);
    } else {
        eigenMatrix grm = eigenMatrix::Zero(_n, _n);
        eigenMatrix grm_N = eigenMatrix::Zero(_n, _n);
        for (f = 0; f < (int)grm_files.size(); f++) {
            LOGGER << "Reading the GRM from the " << f + 1 << "th file ..." << endl;
            read_grm(grm_files[f], grm_id);
            StrFunc::match(uni_id, grm_id, kp);
            #pragma omp parallel for schedule(dynamic, 32)
            for (long c = 0; c < _n; c++) {
                const int kc = kp[c];
                for (long r = c; r < _n; r++) {
                    const int kr = kp[r];
                    const int hi = std::max(kr, kc);
                    const int lo = std::min(kr, kc);
                    const double vn = static_cast<double>(_grm_N(hi, lo));
                    grm(r, c) += static_cast<double>(_grm(hi, lo)) * vn;
                    grm_N(r, c) += vn;
                }
            }
        }
        _grm.resize(_n, _n);
        _grm_N.resize(_n, _n);
        #pragma omp parallel for schedule(dynamic, 32)
        for (long c = 0; c < _n; c++) {
            for (long r = c; r < _n; r++) {
                const double vn = grm_N(r, c);
                const double val = (vn == 0.0) ? 0.0 : (grm(r, c) / vn);
                _grm(r, c) = val;
                _grm(c, r) = val;
                const float vnf = static_cast<float>(vn);
                _grm_N(r, c) = vnf;
                _grm_N(c, r) = vnf;
            }
        }
    }
    LOGGER << "\n" << grm_files.size() << " GRMs have been merged together." << endl;
}

void gcta::align_grm(string m_grm_file) {
    vector<string> grm_files, grm_id;
    read_grm_filenames(m_grm_file, grm_files);
    
    int f = 0, i = 0;
    for (f = 0; f < (int)grm_files.size(); f++) {
        read_grm(grm_files[f], grm_id, false, true);
        update_id_map_kp(grm_id, _id_map, _keep);
    }
    vector<string> uni_id;
    uni_id.reserve(_keep.size());
    for (i = 0; i < (int)_keep.size(); i++) uni_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
    _n = uni_id.size();
    if (_n == 0) LOGGER.e(0, "no individual is in common among the GRM files.");
    else LOGGER << _n << " individuals are in common among the GRM files." << endl;
    
    string _out_save = _out;
    
    vector<int> kp;
    eigenMatrix grm(_n, _n);
    MatrixXf grm_N(_n, _n);
    for (f = 0; f < (int)grm_files.size(); f++) {
        LOGGER << "Reading the GRM from the " << f + 1 << "th file ..." << endl;
        read_grm(grm_files[f], grm_id);
        StrFunc::match(uni_id, grm_id, kp);
        #pragma omp parallel for schedule(dynamic, 32)
        for (long c = 0; c < _n; c++) {
            const int kc = kp[c];
            for (long r = c; r < _n; r++) {
                const int kr = kp[r];
                const int hi = std::max(kr, kc);
                const int lo = std::min(kr, kc);
                const float vn = _grm_N(hi, lo);
                const double val = (vn == 0.0f) ? 0.0 : static_cast<double>(_grm(hi, lo));
                grm(r, c) = val;
                grm(c, r) = val;
                grm_N(r, c) = vn;
                grm_N(c, r) = vn;
            }
        }
        _grm.swap(grm);
        _grm_N.swap(grm_N);
        
        _out = grm_files[f] + ".aligned";
        output_grm(true);
    }
    
    _out = _out_save;
    
    grm.resize(0, 0);
    grm_N.resize(0, 0);
    LOGGER << "\n" << grm_files.size() << " GRMs have been aligned." << endl;
}


void gcta::read_grm_filenames(string merge_grm_file, vector<string> &grm_files, bool out_log) {
    ifstream merge_grm(merge_grm_file.c_str());
    if (!merge_grm) LOGGER.e(0, "cannot open the file [" + merge_grm_file + "] to read.");
    string str_buf;
    grm_files.clear();
    vector<string> vs_buf;
    while (getline(merge_grm, str_buf)) {
        if (!str_buf.empty()) {
            if (StrFunc::split_string(str_buf, vs_buf) == 1) grm_files.push_back(vs_buf[0]);
        }
    }
    if (out_log) LOGGER << "There are " << grm_files.size() << " GRM file names specified in [" + merge_grm_file + "]." << endl;
    if (grm_files.size() > 1000) LOGGER.e(0, "too many GRM file names specified in [" + merge_grm_file + "]. The maximum number is 1000.");
    if (grm_files.size() < 1) LOGGER.e(0, "no GRM file name is found in [" + merge_grm_file + "].");
}

void gcta::grm_bK(string grm_file, string keep_indi_file, string remove_indi_file, double threshold, bool grm_out_bin_flag)
{
    int i = 0, j = 0;
    vector<string> grm_id;
    read_grm(grm_file, grm_id);
    if (!keep_indi_file.empty()) keep_indi(keep_indi_file);
    if (!remove_indi_file.empty()) remove_indi(remove_indi_file);
    if (!keep_indi_file.empty() || !remove_indi_file.empty()) {
        eigenMatrix grm_buf(_grm);
        _grm.resize(_keep.size(), _keep.size());
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) _grm(i, j) = grm_buf(_keep[i], _keep[j]);
        }
        grm_buf.resize(0,0);
        MatrixXf grm_N_buf = _grm_N;
        _grm_N.resize(_keep.size(), _keep.size());
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) _grm_N(i, j) = grm_N_buf(_keep[i], _keep[j]);
        }
    }

    LOGGER << "\nThe off-diagonals that are < " << threshold << " are set to zero.\n" << endl;
    for (i = 0; i < _keep.size(); i++) {
        for (j = 0; j < i; j++){
            if(_grm(i, j) < threshold) _grm(i, j) = 0.0;
        }
    }

    output_grm(grm_out_bin_flag);
}

void gcta::pca(string grm_file, string keep_indi_file, string remove_indi_file, double grm_cutoff, bool merge_grm_flag, int out_pc_num)
{
    manipulate_grm(grm_file, keep_indi_file, remove_indi_file, "", grm_cutoff, -2.0, -2, merge_grm_flag, true);
    _grm_N.resize(0, 0);
    int i = 0, j = 0, n = _keep.size();
    LOGGER << "\nPerforming principal component analysis ..." << endl;

    SelfAdjointEigenSolver<MatrixXd> eigensolver(_grm.cast<double>());
    MatrixXd evec = (eigensolver.eigenvectors());
    VectorXd eval = eigensolver.eigenvalues();

    string eval_file = _out + ".eigenval";
    ofstream o_eval(eval_file.c_str());
    if (!o_eval) LOGGER.e(0, "cannot open the file [" + eval_file + "] to read.");
    for (i = n - 1; i >= 0; i--) o_eval << eval(i) << endl;
    o_eval.close();
    LOGGER << "Eigenvalues of " << n << " individuals have been saved in [" + eval_file + "]." << endl;
    string evec_file = _out + ".eigenvec";
    ofstream o_evec(evec_file.c_str());
    if (!o_evec) LOGGER.e(0, "cannot open the file [" + evec_file + "] to read.");
    if (out_pc_num > n) out_pc_num = n;
    for (i = 0; i < n; i++) {
        o_evec << _fid[_keep[i]] << " " << _pid[_keep[i]];
        for (j = n - 1; j >= (n - out_pc_num); j--) o_evec << " " << evec(i, j);
        o_evec << endl;
    }
    o_evec.close();
    LOGGER << "The first " << out_pc_num << " eigenvectors of " << n << " individuals have been saved in [" + evec_file + "]." << endl;
}

void gcta::snp_pc_loading(string pc_file)
{
    // read eigenvectors and eigenvalues
    string eigenval_file = pc_file + ".eigenval";
    ifstream in_eigenval(eigenval_file.c_str());
    if (!in_eigenval) LOGGER.e(0, "cannot open the file [" + eigenval_file + "] to read.");
    string eigenvec_file = pc_file + ".eigenvec";
    ifstream in_eigenvec(eigenvec_file.c_str());
    if (!in_eigenvec) LOGGER.e(0, "cannot open the file [" + eigenvec_file + "] to read.");
  
    LOGGER << "Reading eigenvectors from [" + eigenvec_file + "]." << endl;
    vector<string> eigenvec_ID;
    vector< vector<string> > eigenvec_str;
    int eigenvec_num = read_fac(in_eigenvec, eigenvec_ID, eigenvec_str);
    LOGGER << eigenvec_num << " eigenvectors of " << eigenvec_ID.size() << " individuals are included from [" + eigenvec_file + "]." << endl;
    update_id_map_kp(eigenvec_ID, _id_map, _keep);

    LOGGER << "\nReading eigenvalues from [" + eigenval_file + "]." << endl;
    vector<double> eigenval_buf;
    double d_buf = 0.0;
    int eigenval_num = 0;
    while(in_eigenval && eigenval_num < eigenvec_num){
        in_eigenval >> d_buf;
        if(d_buf > 1e10 || d_buf < 1e-10) LOGGER.e(0, "invalid eigenvalue in the file [" + eigenval_file + "].");
        eigenval_buf.push_back(d_buf);
        eigenval_num++;
    }
    if(eigenvec_num != eigenval_num) LOGGER.e(0, "inconsistent numbers of eigenvalues and eigenvectors in the files [" + eigenval_file + "] and [" + eigenvec_file + "]");
    LOGGER << eigenval_num << " eigenvalues are read from [" + eigenval_file + "]" << endl;  

    int i = 0, j = 0;
    vector<string> uni_id;
    map<string, int> uni_id_map;
    map<string, int>::iterator iter;
    for(i=0; i<_keep.size(); i++){
        uni_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
        uni_id_map.insert(pair<string,int>(_fid[_keep[i]]+":"+_pid[_keep[i]], i));
    }
    _n = _keep.size();
    int m = _include.size();
    if(_n < 1) LOGGER.e(0, "no individual is in common among the input files.");
    LOGGER << _n << " individuals in common between the input files are included in the analysis."<<endl;
    
    eigenMatrix eigenvec(eigenvec_num, _n);
    for(i = 0; i < eigenvec_ID.size(); i++){
        iter = uni_id_map.find(eigenvec_ID[i]);
        if(iter == uni_id_map.end()) continue;
        for(j = 0; j < eigenvec_num; j++) eigenvec(j, iter->second) = atof(eigenvec_str[i][j].c_str());
    }

    eigenVector inv_eigenval(eigenval_num);
    for(i = 0; i < eigenval_num; i++)  inv_eigenval(i) = 1.0 / (eigenval_buf[i] * m);

    // calculating SNP loading
    if (_mu.empty()) calcu_mu();
    LOGGER << "\nCalculating PC loadings of SNPs..." << endl;
    eigenMatrix snp_loading(m, eigenvec_num);
    eigenVector x(_n);
    for(j = 0; j < m ; j++) {
        makex_eigenVector(j, x, false, true);
        x = x.array() / sqrt(_mu[_include[j]]*(1.0 - 0.5*_mu[_include[j]]));
        snp_loading.row(j) = (eigenvec * x).array() * inv_eigenval.array();
    }

    string filename = _out + ".pcl";
    LOGGER << "\nSaving the PC loadings of " << m << " SNPs to [" + filename + "] ..." << endl;
    ofstream ofile(filename.c_str());
    if(!ofile) LOGGER.e(0, "cannot open the file [" + filename + "] to write.");
    ofile << "SNP\tA1\tA2\tmu";
    for(i = 0; i < eigenval_num; i++) ofile << "\tpc" << i+1 << "_loading";
    ofile << endl;
    for(i = 0; i < m; i++){
        ofile << _snp_name[_include[i]] << "\t" << _ref_A[_include[i]] << "\t" << _other_A[_include[i]] << "\t" <<  _mu[_include[i]];
        for(j = 0; j < eigenvec_num; j++) ofile << "\t" << snp_loading(i, j);
        ofile << "\n";
    }
    ofile.close();
}

//This function changes the original _geno, _include, never use genotype variable after it!!
void gcta::project_loading(string pc_load, int N){

    #ifdef SINGLE_PRECISION
    typedef float t_val;
    #else
    typedef double t_val;
    #endif

    string f_pc_load = pc_load + ".pcl";
    ifstream h_pc_load(f_pc_load.c_str());
    if (!h_pc_load) LOGGER.e(0, "cannot open the loading file [" + f_pc_load + "] to read.");
    
    // output eigenvec. Moving this up to save the time for user when running in an unwritable directory.
    string out_filename = _out + ".proj.eigenvec";
    //LOGGER << "\nOpen the project file [" << out_filename << "] to write."<< endl;
    ofstream ofile(out_filename.c_str());
    if(!ofile) LOGGER.e(0, "failed to open the file [" + out_filename + "] to write.");

    LOGGER << "Reading PC loadings of SNPs from [" + f_pc_load + "]." << endl;
    string buf;
    vector<string> vsec_buf;
    string header;
    getline(h_pc_load,header);
    size_t N_loading_file = count(header.begin(),header.end(),'p');
    LOGGER << "Number of PC loading vectors: " << N_loading_file << endl;
    while(h_pc_load >> buf){
        vsec_buf.push_back(buf);
    }
    h_pc_load.close();
    buf.clear();
    header.clear();

    int len_col = N_loading_file + 4;
    int num_snp = vsec_buf.size() / len_col;
    LOGGER << "Number of SNPs in loading array: " << num_snp << "." << endl; 
    if(vsec_buf.size() % len_col != 0){
        LOGGER.e(0, "invalid number of columns in the PC loading file. Please check.");
    }
    if(N > N_loading_file){
        LOGGER.e(0, "only " + to_string(N_loading_file) + " vectors of PC loadings available, thus not able to project onto " + to_string(N) + " PCs.");
    }

    vector<string> snps(num_snp);
    vector<string> A1(num_snp);
    vector<string> A2(num_snp);
    vector<t_val> mu(num_snp);
    vector<t_val> snp_loading (num_snp * N);
    for(int read_snp_index=0; read_snp_index < num_snp; read_snp_index++){
       int base_index = read_snp_index * len_col;
       snps[read_snp_index] = vsec_buf[base_index];
       A1[read_snp_index] = vsec_buf[base_index + 1];
       A2[read_snp_index] = vsec_buf[base_index + 2];
       #ifdef SINGLE_PRECISION
       mu[read_snp_index] = atof(vsec_buf[base_index + 3].c_str());
       #else
       mu[read_snp_index] = stod(vsec_buf[base_index + 3]);
       #endif
       for(int N_index=0; N_index<N; N_index++){
           // atof .c_str()
            #ifdef SINGLE_PRECISION
            snp_loading[read_snp_index*N + N_index] = atof(vsec_buf[base_index+4+N_index].c_str());
            #else
            snp_loading[read_snp_index*N + N_index] = stod(vsec_buf[base_index+4+N_index]);
            #endif
        }
    }

    vsec_buf.clear();
    vsec_buf.shrink_to_fit();
    
    LOGGER << "Matching alleles..." << endl;
    vector<int> snp_index_include(snps.size());
    StrFunc::match(snps,_snp_name,snp_index_include);
    // _include should be fixed here, it cause maf caculation go vain; 
    unordered_set<int> ori_SNPs(_include.begin(),_include.end());
    _include.clear();

    vector<t_val> filter_snp_loading;
    filter_snp_loading.reserve(num_snp * N);

    LOGGER << "Adjusting A1" << endl;
    bool remove_flag = true;
    vector<string> missnp_list;
    vector<t_val> mu_adj;
    for(int snp_index=0; snp_index < snps.size(); snp_index++){
        int cur_snp_index = snp_index_include[snp_index];
        if(cur_snp_index >= 0 && ori_SNPs.find(cur_snp_index) != ori_SNPs.end() ){
            if((StrFunc::i_compare(A1[snp_index],_allele1[cur_snp_index]) && 
                StrFunc::i_compare(A2[snp_index],_allele2[cur_snp_index])) 
               || (StrFunc::i_compare(A1[snp_index],_allele2[cur_snp_index]) && 
                StrFunc::i_compare(A2[snp_index],_allele1[cur_snp_index]))){
                     _include.push_back(cur_snp_index);
                     mu_adj.push_back(mu[snp_index]);
                     _ref_A[cur_snp_index] = A1[snp_index];
                     filter_snp_loading.insert(filter_snp_loading.end(), snp_loading.begin() + N*snp_index, snp_loading.begin() + N*snp_index + N );
                     continue;
            }
        }
        missnp_list.push_back(snps[snp_index]);
    }

    snp_loading.clear();
    snp_loading.shrink_to_fit();

    //Map the vector to Matrix, share the same memory, thus to save the memory.
    eigenMatrix m_snp_loading = Map< Matrix<t_val,Dynamic,Dynamic,RowMajor> > (filter_snp_loading.data(), _include.size(), N);
    LOGGER << " " << m_snp_loading.rows() << " SNPs are included for loading" << endl;

    if(missnp_list.size() > 0){
        LOGGER.w(0, to_string(missnp_list.size()) + " SNPs are not found or alleles mismatch in the target genotype"); 
        string miss_file = _out + ".proj.missnp";
        LOGGER << " See [" << miss_file << "] for details. If there are many missing SNPs, the projection might be biased." << endl;
        ofstream h_miss(miss_file);
        ostream_iterator<string> output_iterator(h_miss,"\n");
        copy(missnp_list.begin(), missnp_list.end(), output_iterator);
    }
    missnp_list.clear();
    missnp_list.shrink_to_fit();

    //if(_mu.empty()) calcu_mu();
    LOGGER << "Standardizing genotypes and projecting PCs..." << endl;
    LOGGER << "Total number of subjects: " << _keep.size() << "\n" << endl;
    //ofstream demo(_out + ".proj.matrix");
    LOGGER << "Processing subject number: " << endl;
    eigenMatrix PCs(_keep.size(),N);
    #pragma omp parallel for ordered schedule(dynamic)
    for(int ind_index=0; ind_index < _keep.size(); ind_index++){
        LOGGER <<  to_string(ind_index+1) + "\r" << flush;
        Matrix<t_val,1,Dynamic> geno(_include.size());
        for(int snp_index=0; snp_index < _include.size(); snp_index++){
            if (!_snp_1[_include[snp_index]][_keep[ind_index]] || _snp_2[_include[snp_index]][_keep[ind_index]]) {
                geno(snp_index) = _snp_1[_include[snp_index]][_keep[ind_index]] + _snp_2[_include[snp_index]][_keep[ind_index]];
                if (_allele1[_include[snp_index]] != _ref_A[_include[snp_index]]) geno(snp_index) = 2.0 - geno(snp_index);
                geno(snp_index) = (geno(snp_index) - mu_adj[snp_index]) / sqrt(mu_adj[snp_index]*(1.0 - 0.5*mu_adj[snp_index]));
            }else{
                geno(snp_index) = 0.0;
            }
        }
        PCs.row(ind_index) = geno * m_snp_loading;
        /*
        if(ind_index==0){
            demo << "geno:" << endl;
            demo << geno << endl;
            demo << "m_snp_loading" << endl;
            demo << m_snp_loading << endl;
            demo << "PC" << endl;
            demo << PCs << endl;
            demo.close();
        }
        */
        geno.resize(0);
    }
    
    // Output the values
    for(int ind_index=0; ind_index < _keep.size(); ind_index++){
        ofile << _fid[_keep[ind_index]] << "\t" << _pid[_keep[ind_index]] << "\t";
        for(int pc_index=0; pc_index<N; pc_index++){
            ofile << PCs(ind_index,pc_index) << "\t"; 
        }
        ofile << "\n";
    }
    ofile.close();
    
    LOGGER << "\nFinished, and the PCs have all been saved to " << out_filename << endl;
}

