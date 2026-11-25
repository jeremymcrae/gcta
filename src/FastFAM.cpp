/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   FastFAM regression

   Depends on the class of genotype

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/
#include "FastFAM.h"
#include "OptionIO.h"
#include "StatLib.h"
#include <cmath>
#include <algorithm>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#if GCTA_CPU_x86
#include <Eigen/PardisoSupport>
#endif

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <sstream>
#include <iterator>
#include "utils.hpp"
#include "Logger.h"
#include "ThreadPool.h"
#include "omp.h"
#include "Matrix.hpp"
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
#include "Covar.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <boost/math/distributions/beta.hpp>

struct InvItem{
    int32_t row;
    int32_t col;
    double val;
};

struct ScopedSingleThreadBLAS {
#if defined(OPENBLAS_CONST)
    int prev_threads;
    ScopedSingleThreadBLAS() : prev_threads(openblas_get_num_threads()) {
        openblas_set_num_threads(1);
    }
    ~ScopedSingleThreadBLAS() {
        openblas_set_num_threads(prev_threads);
    }
#else
    ScopedSingleThreadBLAS() {}
    ~ScopedSingleThreadBLAS() {}
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

static MatrixXd parallelAtB(const Ref<const MatrixXd> &A, const Ref<const MatrixXd> &B) {
    const Eigen::Index n_rows = A.rows();
    const Eigen::Index ca = A.cols();
    const Eigen::Index cb = B.cols();
    if (n_rows < 8192) {
        return A.transpose() * B;
    }
    const Eigen::Index kTile = 2048;
    const Eigen::Index n_tiles = (n_rows + kTile - 1) / kTile;
    vector<MatrixXd> partials(static_cast<size_t>(n_tiles));
    ScopedSingleThreadBLAS blas_guard;
    #pragma omp parallel
    {
        ScopedSingleThreadBLAS::ThreadLocal tl_guard;
        #pragma omp for schedule(static)
        for (Eigen::Index t = 0; t < n_tiles; ++t) {
            const Eigen::Index r0 = t * kTile;
            const Eigen::Index rows = std::min<Eigen::Index>(kTile, n_rows - r0);
            partials[static_cast<size_t>(t)].noalias() =
                A.middleRows(r0, rows).transpose() * B.middleRows(r0, rows);
        }
    }
    MatrixXd out = MatrixXd::Zero(ca, cb);
    for (Eigen::Index t = 0; t < n_tiles; ++t) {
        out.noalias() += partials[static_cast<size_t>(t)];
    }
    return out;
}

static VectorXd parallelAtVec(const Ref<const MatrixXd> &A, const Ref<const VectorXd> &v) {
    const Eigen::Index n_rows = A.rows();
    const Eigen::Index ca = A.cols();
    if (n_rows < 8192) {
        return A.transpose() * v;
    }
    const Eigen::Index kTile = 2048;
    const Eigen::Index n_tiles = (n_rows + kTile - 1) / kTile;
    vector<VectorXd> partials(static_cast<size_t>(n_tiles));
    ScopedSingleThreadBLAS blas_guard;
    #pragma omp parallel
    {
        ScopedSingleThreadBLAS::ThreadLocal tl_guard;
        #pragma omp for schedule(static)
        for (Eigen::Index t = 0; t < n_tiles; ++t) {
            const Eigen::Index r0 = t * kTile;
            const Eigen::Index rows = std::min<Eigen::Index>(kTile, n_rows - r0);
            partials[static_cast<size_t>(t)].noalias() =
                A.middleRows(r0, rows).transpose() * v.segment(r0, rows);
        }
    }
    VectorXd out = VectorXd::Zero(ca);
    for (Eigen::Index t = 0; t < n_tiles; ++t) {
        out.noalias() += partials[static_cast<size_t>(t)];
    }
    return out;
}

static void updateLogisticWorkingFromEta(const Ref<const VectorXd> &Xa_b,
                                         const Ref<const VectorXd> &phenoVec,
                                         VectorXd &mu,
                                         VectorXd &var_mu_i,
                                         VectorXd &Y) {
    const Eigen::Index n = Xa_b.size();
    mu.resize(n);
    var_mu_i.resize(n);
    Y.resize(n);
    const double * __restrict eta_p = Xa_b.data();
    const double * __restrict y_obs = phenoVec.data();
    double * __restrict mu_p = mu.data();
    double * __restrict v_p = var_mu_i.data();
    double * __restrict y_wrk = Y.data();

    #pragma omp parallel for schedule(static) if(n >= 4096)
    for (Eigen::Index i = 0; i < n; ++i) {
        const double eta = eta_p[i];
        const double m = 1.0 / (std::exp(-eta) + 1.0);
        const double v = 1.0 / (m * (1.0 - m));
        mu_p[i] = m;
        v_p[i] = v;
        y_wrk[i] = eta + v * (y_obs[i] - m);
    }
}

struct FastFAM::BinSparseLDLT {
    typedef SparseMatrix<double, Eigen::ColMajor, int32_t> SpMat32;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;

    static constexpr int32_t kGramTileRows = 256;

    struct Workspace {
        vector<double> diag_s;
        vector<double> inv_diag_s;
        vector<double> Lx;
        vector<double> diag;
        vector<double> inv_diag;
        vector<double> y;
        RowMatrixXd U_r;
        RowMatrixXd scaled_tile;
        MatrixXd G;
        MatrixXd XtVX;
        MatrixXd solve_buf;
        vector<double> solve_vec;
        bool is_diag_only = false;

        void init(int32_t n_s, int32_t n_r, int32_t nnz_L, int32_t n_cols_aug = 0) {
            diag_s.assign(n_s, 0.0);
            inv_diag_s.assign(n_s, 0.0);
            Lx.assign(nnz_L, 0.0);
            diag.assign(n_r, 0.0);
            inv_diag.assign(n_r, 0.0);
            y.assign(n_r, 0.0);
            solve_vec.assign(n_r, 0.0);
            if (n_cols_aug > 0) {
                U_r.resize(n_r, n_cols_aug);
                scaled_tile.resize(kGramTileRows, n_cols_aug);
                G.resize(n_cols_aug, n_cols_aug);
                XtVX.resize(n_cols_aug - 1, n_cols_aug - 1);
            }
        }

        // Lightweight worker workspace for evalLogL (omits unused singleton buffers)
        void initEval(int32_t n_r, int32_t nnz_L, int32_t n_cols_aug) {
            Lx.assign(nnz_L, 0.0);
            diag.assign(n_r, 0.0);
            inv_diag.assign(n_r, 0.0);
            y.assign(n_r, 0.0);
            U_r.resize(n_r, n_cols_aug);
            scaled_tile.resize(kGramTileRows, n_cols_aug);
            G.resize(n_cols_aug, n_cols_aug);
            XtVX.resize(n_cols_aug - 1, n_cols_aug - 1);
        }
    };

    int32_t n_total = 0;
    int32_t n_s = 0;
    int32_t n = 0; // number of related individuals (n_r)
    int32_t nnz_L = 0;
    int32_t n_covar = 0;

    vector<int32_t> s_orig;
    vector<double> k_diag_s;
    double min_k_diag_s = 0.0;
    double max_k_diag_s = 0.0;
    double min_w_s = 0.0;
    vector<int32_t> r_perm_orig;

    // Pre-gathered design + working response [X, Y] in row-major layout for singletons and permuted relateds
    RowMatrixXd B_s;
    RowMatrixXd B_r_perm;

    vector<int32_t> ap_col_ptr;
    vector<int32_t> ap_row_idx;
    vector<double> ap_val;
    vector<double> ap_diag;

    vector<int32_t> Lp;
    vector<int32_t> Li;
    vector<int32_t> row_ptr;
    vector<int32_t> row_col;
    vector<int32_t> row_lpos;

    Workspace main_ws;
    vector<double> inv_d_diag_orig;
    MatrixXd XtVX_inv;

    void init(const SpMat &fam) {
        if (fam.rows() != fam.cols()) {
            LOGGER.e(0, "the sparse GRM matrix must be square.");
        }
        if (fam.rows() >= std::numeric_limits<int32_t>::max() ||
            fam.nonZeros() + fam.rows() >= std::numeric_limits<int32_t>::max()) {
            LOGGER.e(0, "the sparse GRM exceeds 32-bit index capacity.");
        }
        n_total = static_cast<int32_t>(fam.rows());
        if (n_total == 0) {
            n_s = 0;
            n = 0;
            nnz_L = 0;
            return;
        }

        // Partition samples into singletons (zero off-diagonal entries in lower triangle)
        // and related individuals (at least one non-zero off-diagonal entry).
        vector<uint8_t> is_rel(n_total, 0);
        vector<double> diag_orig(n_total, 0.0);
        for (int32_t j = 0; j < n_total; ++j) {
            for (SpMat::InnerIterator it(fam, j); it; ++it) {
                int32_t i = static_cast<int32_t>(it.index());
                if (i == j) {
                    diag_orig[j] += it.value();
                } else if (i > j && it.value() != 0.0) {
                    is_rel[j] = 1;
                    is_rel[i] = 1;
                }
            }
        }

        s_orig.clear();
        k_diag_s.clear();
        vector<int32_t> r_orig;
        vector<int32_t> orig_to_r(n_total, -1);
        for (int32_t i = 0; i < n_total; ++i) {
            if (is_rel[i]) {
                orig_to_r[i] = static_cast<int32_t>(r_orig.size());
                r_orig.push_back(i);
            } else {
                s_orig.push_back(i);
                k_diag_s.push_back(diag_orig[i]);
            }
        }
        n_s = static_cast<int32_t>(s_orig.size());
        n = static_cast<int32_t>(r_orig.size());
        if (n_s > 0) {
            auto mm = std::minmax_element(k_diag_s.begin(), k_diag_s.end());
            min_k_diag_s = *mm.first;
            max_k_diag_s = *mm.second;
        }

        if (n == 0) {
            nnz_L = 0;
            r_perm_orig.clear();
            ap_col_ptr.clear();
            ap_row_idx.clear();
            ap_val.clear();
            ap_diag.clear();
            Lp.clear();
            Li.clear();
            row_ptr.clear();
            row_col.clear();
            row_lpos.clear();
            main_ws.init(n_s, 0, 0);
            return;
        }

        // Build 32-bit sparse submatrix on the n x n related subset (lower triangle + diagonal)
        SpMat32 fam32(n, n);
        vector<int32_t> col_counts(n, 1); // 1 reserved for diagonal
        for (int32_t rj = 0; rj < n; ++rj) {
            int32_t orig_j = r_orig[rj];
            for (SpMat::InnerIterator it(fam, orig_j); it; ++it) {
                int32_t orig_i = static_cast<int32_t>(it.index());
                if (orig_i > orig_j && it.value() != 0.0) {
                    col_counts[rj]++;
                }
            }
        }
        fam32.reserve(col_counts);
        for (int32_t rj = 0; rj < n; ++rj) {
            int32_t orig_j = r_orig[rj];
            fam32.insertBackUncompressed(rj, rj) = diag_orig[orig_j];
            for (SpMat::InnerIterator it(fam, orig_j); it; ++it) {
                int32_t orig_i = static_cast<int32_t>(it.index());
                if (orig_i > orig_j && it.value() != 0.0) {
                    int32_t ri = orig_to_r[orig_i];
                    fam32.insertBackUncompressed(ri, rj) = it.value();
                }
            }
        }
        fam32.finalize();
        fam32.makeCompressed();

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t> Pinv, P;
        {
            SpMat32 C;
            C = fam32.selfadjointView<Eigen::Lower>();
            Eigen::AMDOrdering<int32_t> ordering;
            ordering(C, Pinv);
        }

        SpMat32 ap(n, n);
        r_perm_orig.resize(n);
        if (Pinv.size() > 0) {
            P = Pinv.inverse();
            for (int32_t k = 0; k < n; ++k) {
                r_perm_orig[k] = r_orig[Pinv.indices()[k]];
            }
            ap.selfadjointView<Eigen::Upper>() = fam32.selfadjointView<Eigen::Lower>().twistedBy(P);
        } else {
            for (int32_t k = 0; k < n; ++k) {
                r_perm_orig[k] = r_orig[k];
            }
            ap.selfadjointView<Eigen::Upper>() = fam32.selfadjointView<Eigen::Lower>();
        }
        ap.makeCompressed();

        ap_col_ptr.assign(n + 1, 0);
        ap_diag.assign(n, 0.0);
        ap_row_idx.clear();
        ap_val.clear();
        ap_row_idx.reserve(ap.nonZeros());
        ap_val.reserve(ap.nonZeros());

        for (int32_t k = 0; k < n; ++k) {
            ap_col_ptr[k] = static_cast<int32_t>(ap_row_idx.size());
            for (SpMat32::InnerIterator it(ap, k); it; ++it) {
                int32_t i = it.index();
                if (i < k) {
                    ap_row_idx.push_back(i);
                    ap_val.push_back(it.value());
                } else if (i == k) {
                    ap_diag[k] += it.value();
                }
            }
        }
        ap_col_ptr[n] = static_cast<int32_t>(ap_row_idx.size());

        // Symbolic elimination tree and column counts of L on the related submatrix
        vector<int32_t> parent(n, -1);
        vector<int32_t> tags(n, -1);
        vector<int32_t> nz_per_col(n, 0);
        for (int32_t k = 0; k < n; ++k) {
            parent[k] = -1;
            tags[k] = k;
            nz_per_col[k] = 0;
            for (int32_t idx = ap_col_ptr[k]; idx < ap_col_ptr[k + 1]; ++idx) {
                int32_t i = ap_row_idx[idx];
                for (; tags[i] != k; i = parent[i]) {
                    if (parent[i] == -1) parent[i] = k;
                    nz_per_col[i]++;
                    tags[i] = k;
                }
            }
        }

        Lp.resize(n + 1, 0);
        int64_t total_nnz = 0;
        for (int32_t k = 0; k < n; ++k) {
            Lp[k] = static_cast<int32_t>(total_nnz);
            total_nnz += nz_per_col[k];
            if (total_nnz >= std::numeric_limits<int32_t>::max()) {
                LOGGER.e(0, "Cholesky factor nonzeros exceed 32-bit index capacity.");
            }
        }
        nnz_L = static_cast<int32_t>(total_nnz);
        Lp[n] = nnz_L;

        // Precompute immutable Li and row-wise topological elimination schedule
        Li.resize(nnz_L);
        row_ptr.resize(n + 1, 0);
        row_col.resize(nnz_L);
        row_lpos.resize(nnz_L);

        vector<int32_t> pattern(n, 0);
        std::fill(tags.begin(), tags.end(), -1);
        std::fill(nz_per_col.begin(), nz_per_col.end(), 0);
        int32_t r_cursor = 0;
        for (int32_t k = 0; k < n; ++k) {
            row_ptr[k] = r_cursor;
            int32_t top = n;
            tags[k] = k;
            for (int32_t idx = ap_col_ptr[k]; idx < ap_col_ptr[k + 1]; ++idx) {
                int32_t i = ap_row_idx[idx];
                int32_t len = 0;
                for (; tags[i] != k; i = parent[i]) {
                    pattern[len++] = i;
                    tags[i] = k;
                }
                while (len > 0) {
                    pattern[--top] = pattern[--len];
                }
            }
            for (int32_t t = top; t < n; ++t) {
                int32_t i = pattern[t];
                int32_t p2 = Lp[i] + nz_per_col[i];
                Li[p2] = k;
                nz_per_col[i]++;
                row_col[r_cursor] = i;
                row_lpos[r_cursor] = p2;
                r_cursor++;
            }
        }
        row_ptr[n] = r_cursor;

        main_ws.init(n_s, n, nnz_L);
    }

    void prepareDesign(const Ref<const MatrixXd> &X) {
        n_covar = static_cast<int32_t>(X.cols());
        const int32_t m = n_covar + 1;
        B_s.resize(n_s, m);
        B_r_perm.resize(n, m);
        main_ws.U_r.resize(n, m);
        main_ws.scaled_tile.resize(kGramTileRows, m);
        main_ws.G.resize(m, m);
        main_ws.XtVX.resize(n_covar, n_covar);

        const int32_t * __restrict so = s_orig.data();
        #pragma omp parallel for schedule(static) if(n_s >= 4096)
        for (int32_t s = 0; s < n_s; ++s) {
            const int32_t idx = so[s];
            for (int32_t j = 0; j < n_covar; ++j) {
                B_s(s, j) = X(idx, j);
            }
            B_s(s, n_covar) = 0.0;
        }

        const int32_t * __restrict rpo = r_perm_orig.data();
        #pragma omp parallel for schedule(static) if(n >= 4096)
        for (int32_t k = 0; k < n; ++k) {
            const int32_t idx = rpo[k];
            for (int32_t j = 0; j < n_covar; ++j) {
                B_r_perm(k, j) = X(idx, j);
            }
            B_r_perm(k, n_covar) = 0.0;
        }
    }

    void updateWorkingState(const Ref<const VectorXd> &var_mu_i, const Ref<const VectorXd> &Y,
                            vector<double> &v_s, vector<double> &v_r_perm) {
        const int32_t c = n_covar;
        const double * __restrict w_src = var_mu_i.data();
        const double * __restrict y_src = Y.data();

        v_s.resize(n_s);
        double * __restrict vs_dst = v_s.data();
        const int32_t * __restrict so = s_orig.data();
        double cur_min_w = std::numeric_limits<double>::infinity();
        #pragma omp parallel for schedule(static) reduction(min:cur_min_w) if(n_s >= 4096)
        for (int32_t s = 0; s < n_s; ++s) {
            const int32_t idx = so[s];
            const double wv = w_src[idx];
            vs_dst[s] = wv;
            if (wv < cur_min_w) cur_min_w = wv;
            B_s(s, c) = y_src[idx];
        }
        min_w_s = (n_s > 0) ? cur_min_w : 0.0;

        v_r_perm.resize(n);
        double * __restrict vr_dst = v_r_perm.data();
        const int32_t * __restrict rpo = r_perm_orig.data();
        #pragma omp parallel for schedule(static) if(n >= 4096)
        for (int32_t k = 0; k < n; ++k) {
            const int32_t idx = rpo[k];
            vr_dst[k] = w_src[idx];
            B_r_perm(k, c) = y_src[idx];
        }
    }

    static void accumulateWeightedGram(const RowMatrixXd &B, const double * __restrict inv_d,
                                       int32_t n_rows, int32_t m,
                                       RowMatrixXd &scaled_tile, MatrixXd &G) {
        bool use_inner_omp = false;
#ifdef _OPENMP
        use_inner_omp = (omp_in_parallel() == 0 && omp_get_max_threads() > 1);
#endif
        if (use_inner_omp && n_rows >= 4096) {
            const int32_t n_tiles = (n_rows + kGramTileRows - 1) / kGramTileRows;
            vector<MatrixXd> g_part(static_cast<size_t>(n_tiles));
            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel
            {
                ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                RowMatrixXd S_tile(kGramTileRows, m);
                #pragma omp for schedule(static)
                for (int32_t t = 0; t < n_tiles; ++t) {
                    const int32_t r0 = t * kGramTileRows;
                    const int32_t rows = std::min<int32_t>(kGramTileRows, n_rows - r0);
                    auto B_blk = B.middleRows(r0, rows);
                    auto S_blk = S_tile.topRows(rows);
                    for (int32_t r = 0; r < rows; ++r) {
                        S_blk.row(r) = inv_d[r0 + r] * B_blk.row(r);
                    }
                    g_part[static_cast<size_t>(t)].noalias() = B_blk.transpose() * S_blk;
                }
            }
            for (int32_t t = 0; t < n_tiles; ++t) {
                G.noalias() += g_part[static_cast<size_t>(t)];
            }
            return;
        }
        for (int32_t r0 = 0; r0 < n_rows; r0 += kGramTileRows) {
            const int32_t rows = std::min<int32_t>(kGramTileRows, n_rows - r0);
            auto B_blk = B.middleRows(r0, rows);
            auto S_blk = scaled_tile.topRows(rows);
            for (int32_t r = 0; r < rows; ++r) {
                S_blk.row(r) = inv_d[r0 + r] * B_blk.row(r);
            }
            G.noalias() += B_blk.transpose() * S_blk;
        }
    }

    bool evalGramAndLogdet(const double * __restrict w_s, const double * __restrict w_r_perm,
                           double cur_tao, Workspace &ws, double &out_logdet_v) const {
        ws.is_diag_only = (cur_tao == 0.0);
        const int32_t c = n_covar;
        const int32_t m = c + 1;

        double logdet_r = 0.0;
        if (n > 0) {
            if (!factorizeRelatedOnly(w_r_perm, cur_tao, ws, logdet_r)) {
                return false;
            }
        }

        ws.G.setZero(m, m);

        bool use_inner_omp = false;
#ifdef _OPENMP
        use_inner_omp = (omp_in_parallel() == 0 && omp_get_max_threads() > 1);
#endif

        double logdet_s = 0.0;
        if (n_s > 0) {
            const double * __restrict kd_s = k_diag_s.data();
            const bool guaranteed_pos = (ws.is_diag_only && min_w_s > 0.0) ||
                (cur_tao > 0.0 && min_w_s + cur_tao * min_k_diag_s > 0.0) ||
                (cur_tao < 0.0 && min_w_s + cur_tao * max_k_diag_s > 0.0);

            if (use_inner_omp && n_s >= 4096) {
                const int32_t n_tiles = (n_s + kGramTileRows - 1) / kGramTileRows;
                vector<MatrixXd> g_part(static_cast<size_t>(n_tiles));
                vector<double> ld_part(static_cast<size_t>(n_tiles), 0.0);
                int any_bad = 0;
                ScopedSingleThreadBLAS blas_guard;
                #pragma omp parallel
                {
                    ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                    RowMatrixXd S_tile(kGramTileRows, m);
                    double d_tile[kGramTileRows];
                    #pragma omp for schedule(static) reduction(|:any_bad)
                    for (int32_t t = 0; t < n_tiles; ++t) {
                        const int32_t r0 = t * kGramTileRows;
                        const int32_t rows = std::min<int32_t>(kGramTileRows, n_s - r0);
                        auto B_blk = B_s.middleRows(r0, rows);
                        auto S_blk = S_tile.topRows(rows);
                        int loc_bad = 0;
                        if (ws.is_diag_only) {
                            if (!guaranteed_pos) {
                                for (int32_t r = 0; r < rows; ++r) {
                                    const double d = w_s[r0 + r];
                                    if (!(d > 0.0)) { loc_bad = 1; break; }
                                    d_tile[r] = d;
                                }
                            } else {
                                std::memcpy(d_tile, w_s + r0, static_cast<size_t>(rows) * sizeof(double));
                            }
                        } else if (guaranteed_pos) {
                            for (int32_t r = 0; r < rows; ++r) {
                                d_tile[r] = w_s[r0 + r] + cur_tao * kd_s[r0 + r];
                            }
                        } else {
                            for (int32_t r = 0; r < rows; ++r) {
                                const double d = w_s[r0 + r] + cur_tao * kd_s[r0 + r];
                                if (!(d > 0.0)) { loc_bad = 1; break; }
                                d_tile[r] = d;
                            }
                        }
                        if (loc_bad) {
                            any_bad |= 1;
                        } else {
                            Map<const Eigen::ArrayXd> d_arr(d_tile, rows);
                            ld_part[static_cast<size_t>(t)] = d_arr.log().sum();
                            for (int32_t r = 0; r < rows; ++r) {
                                S_blk.row(r) = (1.0 / d_tile[r]) * B_blk.row(r);
                            }
                            g_part[static_cast<size_t>(t)].noalias() = B_blk.transpose() * S_blk;
                        }
                    }
                }
                if (any_bad) return false;
                for (int32_t t = 0; t < n_tiles; ++t) {
                    logdet_s += ld_part[static_cast<size_t>(t)];
                    ws.G.noalias() += g_part[static_cast<size_t>(t)];
                }
            } else {
                double d_tile[kGramTileRows];
                for (int32_t r0 = 0; r0 < n_s; r0 += kGramTileRows) {
                    const int32_t rows = std::min<int32_t>(kGramTileRows, n_s - r0);
                    auto B_blk = B_s.middleRows(r0, rows);
                    auto S_blk = ws.scaled_tile.topRows(rows);
                    if (ws.is_diag_only) {
                        if (!guaranteed_pos) {
                            for (int32_t r = 0; r < rows; ++r) {
                                const double d = w_s[r0 + r];
                                if (!(d > 0.0)) return false;
                                d_tile[r] = d;
                            }
                        } else {
                            std::memcpy(d_tile, w_s + r0, static_cast<size_t>(rows) * sizeof(double));
                        }
                    } else if (guaranteed_pos) {
                        for (int32_t r = 0; r < rows; ++r) {
                            d_tile[r] = w_s[r0 + r] + cur_tao * kd_s[r0 + r];
                        }
                    } else {
                        for (int32_t r = 0; r < rows; ++r) {
                            const double d = w_s[r0 + r] + cur_tao * kd_s[r0 + r];
                            if (!(d > 0.0)) return false;
                            d_tile[r] = d;
                        }
                    }
                    Map<const Eigen::ArrayXd> d_arr(d_tile, rows);
                    logdet_s += d_arr.log().sum();
                    for (int32_t r = 0; r < rows; ++r) {
                        S_blk.row(r) = (1.0 / d_tile[r]) * B_blk.row(r);
                    }
                    ws.G.noalias() += B_blk.transpose() * S_blk;
                }
            }
            if (!std::isfinite(logdet_s)) {
                return false;
            }
        }

        if (n > 0) {
            if (ws.is_diag_only || nnz_L == 0) {
                accumulateWeightedGram(B_r_perm, ws.inv_diag.data(), n, m, ws.scaled_tile, ws.G);
            } else {
                const int32_t * __restrict lp = Lp.data();
                const int32_t * __restrict li = Li.data();
                const double * __restrict Lx = ws.Lx.data();
                if (use_inner_omp && n >= 4096 && m >= 2) {
                    double * __restrict u_out = ws.U_r.data();
                    const double * __restrict b_in = B_r_perm.data();
                    #pragma omp parallel
                    {
                        vector<double> u_col(static_cast<size_t>(n));
                        double * __restrict uc = u_col.data();
                        #pragma omp for schedule(static)
                        for (int32_t col = 0; col < m; ++col) {
                            for (int32_t i = 0; i < n; ++i) {
                                uc[i] = b_in[static_cast<size_t>(i) * m + col];
                            }
                            for (int32_t i = 0; i < n; ++i) {
                                const int32_t p_start = lp[i];
                                const int32_t p_end = lp[i + 1];
                                if (p_start == p_end) continue;
                                const double ui = uc[i];
                                for (int32_t p = p_start; p < p_end; ++p) {
                                    uc[li[p]] -= Lx[p] * ui;
                                }
                            }
                            for (int32_t i = 0; i < n; ++i) {
                                u_out[static_cast<size_t>(i) * m + col] = uc[i];
                            }
                        }
                    }
                } else {
                    ws.U_r = B_r_perm;
                    double * __restrict u_data = ws.U_r.data();
                    for (int32_t i = 0; i < n; ++i) {
                        const int32_t p_start = lp[i];
                        const int32_t p_end = lp[i + 1];
                        if (p_start == p_end) continue;
                        const double * __restrict u_i = u_data + static_cast<size_t>(i) * m;
                        for (int32_t p = p_start; p < p_end; ++p) {
                            const double l_val = Lx[p];
                            double * __restrict u_dst = u_data + static_cast<size_t>(li[p]) * m;
                            for (int32_t col = 0; col < m; ++col) {
                                u_dst[col] -= l_val * u_i[col];
                            }
                        }
                    }
                }
                accumulateWeightedGram(ws.U_r, ws.inv_diag.data(), n, m, ws.scaled_tile, ws.G);
            }
        }

        out_logdet_v = logdet_s + logdet_r;
        return std::isfinite(out_logdet_v);
    }

    double evalLogL(const double * __restrict w_s, const double * __restrict w_r_perm,
                    double cur_tao, Workspace &ws) const {
        double logdet_v = 0.0;
        if (!evalGramAndLogdet(w_s, w_r_perm, cur_tao, ws, logdet_v)) {
            return -std::numeric_limits<double>::infinity();
        }
        const int32_t c = n_covar;
        ws.XtVX = ws.G.topLeftCorner(c, c);
        Eigen::ColPivHouseholderQR<MatrixXd> XtVX_solver(ws.XtVX);
        if (!XtVX_solver.isInvertible()) {
            return -std::numeric_limits<double>::infinity();
        }
        const double logdet_XtVX = XtVX_solver.logAbsDeterminant();
        const VectorXd XtVY = ws.G.block(0, c, c, 1);
        const VectorXd beta_hat = XtVX_solver.solve(XtVY);
        const double yPy = ws.G(c, c) - XtVY.dot(beta_hat);

        return -0.5 * (logdet_v + logdet_XtVX + yPy);
    }

    // Zero-ViX BLUP step: solves for fixed effects out_a via L^{-1}[X, Y] Gram matrix
    // and computes out_eta = Y - w .* V^{-1}(Y - X * out_a) using ONE single-vector sparse solve.
    bool solveBlupStep(const Ref<const MatrixXd> &X,
                       const Ref<const VectorXd> &in_var_mu,
                       const Ref<const VectorXd> &in_Y,
                       double cur_tao,
                       vector<double> &v_s,
                       vector<double> &v_r_perm,
                       VectorXd &out_a,
                       VectorXd &out_eta) {
        updateWorkingState(in_var_mu, in_Y, v_s, v_r_perm);
        double logdet_v = 0.0;
        if (!evalGramAndLogdet(v_s.data(), v_r_perm.data(), cur_tao, main_ws, logdet_v)) {
            return false;
        }
        const int32_t c = n_covar;
        main_ws.XtVX = main_ws.G.topLeftCorner(c, c);
        Eigen::ColPivHouseholderQR<MatrixXd> XtVX_solver(main_ws.XtVX);
        if (!XtVX_solver.isInvertible()) {
            return false;
        }
        out_a = XtVX_solver.solve(main_ws.G.block(0, c, c, 1));
        if (!out_a.array().isFinite().all()) {
            return false;
        }

        out_eta.resize(n_total);
        double * __restrict eta_dst = out_eta.data();
        const double * __restrict y_src = in_Y.data();
        const double * __restrict w_src = in_var_mu.data();

        // Singletons: eta_i = Y_i - w_i * (Y_i - x_i' a) / (w_i + tao * k_ii)
        if (n_s > 0) {
            const int32_t * __restrict so = s_orig.data();
            const double * __restrict ws_arr = v_s.data();
            const double * __restrict kd_s = k_diag_s.data();
            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel if(n_s >= 4096)
            {
                ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                VectorXd r_tile(static_cast<Eigen::Index>(kGramTileRows));
                #pragma omp for schedule(static)
                for (int32_t r0 = 0; r0 < n_s; r0 += kGramTileRows) {
                    const int32_t rows = std::min<int32_t>(kGramTileRows, n_s - r0);
                    auto rt = r_tile.head(rows);
                    rt.noalias() = B_s.block(r0, c, rows, 1) - B_s.block(r0, 0, rows, c) * out_a;
                    if (cur_tao == 0.0) {
                        for (int32_t r = 0; r < rows; ++r) {
                            const int32_t idx = so[r0 + r];
                            eta_dst[idx] = y_src[idx] - rt[r];
                        }
                    } else {
                        for (int32_t r = 0; r < rows; ++r) {
                            const int32_t s = r0 + r;
                            const int32_t idx = so[s];
                            const double d = ws_arr[s] + cur_tao * kd_s[s];
                            eta_dst[idx] = y_src[idx] - ws_arr[s] * (rt[r] / d);
                        }
                    }
                }
            }
        }

        // Relateds: single-vector forward + diagonal + backward solve on r_perm = Y_r - X_r * out_a
        if (n > 0) {
            double * __restrict dst = main_ws.solve_vec.data();
            {
                ScopedSingleThreadBLAS blas_guard;
                #pragma omp parallel if(n >= 4096)
                {
                    ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                    VectorXd r_tile(static_cast<Eigen::Index>(kGramTileRows));
                    #pragma omp for schedule(static)
                    for (int32_t r0 = 0; r0 < n; r0 += kGramTileRows) {
                        const int32_t rows = std::min<int32_t>(kGramTileRows, n - r0);
                        auto rt = r_tile.head(rows);
                        rt.noalias() = B_r_perm.block(r0, c, rows, 1) - B_r_perm.block(r0, 0, rows, c) * out_a;
                        std::memcpy(dst + r0, rt.data(), static_cast<size_t>(rows) * sizeof(double));
                    }
                }
            }
            const int32_t * __restrict rpo = r_perm_orig.data();
            const double * __restrict inv_diag = main_ws.inv_diag.data();
            if (main_ws.is_diag_only || nnz_L == 0) {
                #pragma omp parallel for schedule(static) if(n >= 4096)
                for (int32_t k = 0; k < n; ++k) {
                    const int32_t idx = rpo[k];
                    eta_dst[idx] = y_src[idx] - w_src[idx] * (dst[k] * inv_diag[k]);
                }
            } else {
                const int32_t * __restrict lp = Lp.data();
                const int32_t * __restrict li = Li.data();
                const double * __restrict Lx = main_ws.Lx.data();
                for (int32_t i = 0; i < n; ++i) {
                    const double tmp = dst[i];
                    if (tmp != 0.0) {
                        for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                            dst[li[p]] -= tmp * Lx[p];
                        }
                    }
                }
                for (int32_t i = 0; i < n; ++i) {
                    dst[i] *= inv_diag[i];
                }
                for (int32_t i = n - 1; i >= 0; --i) {
                    double tmp = dst[i];
                    for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                        tmp -= Lx[p] * dst[li[p]];
                    }
                    dst[i] = tmp;
                }
                #pragma omp parallel for schedule(static) if(n >= 4096)
                for (int32_t k = 0; k < n; ++k) {
                    const int32_t idx = rpo[k];
                    eta_dst[idx] = y_src[idx] - w_src[idx] * dst[k];
                }
            }
        }

        (void)X;
        return out_eta.array().isFinite().all();
    }

    // Exact REML logL(0), right-hand score d(logL)/d(tao)|_{tao=0}, and Fisher information I(0)
    // using purely diagonal V(0) = diag(w) (zero Cholesky factorizations).
    bool evalZeroTaoScoreAndInfo(const double * __restrict w_s, const double * __restrict w_r_perm,
                                 double &out_logL0, double &out_score0, double &out_fisher0) const {
        const int32_t c = n_covar;
        const int32_t m = c + 1;

        MatrixXd G0 = MatrixXd::Zero(m, m);
        double logdet_v0 = 0.0;

        // 1. Parallel Gram & logdet at tao = 0 across singletons and relateds
        if (n_s > 0) {
            const int32_t n_tiles = (n_s + kGramTileRows - 1) / kGramTileRows;
            vector<MatrixXd> g_part(static_cast<size_t>(n_tiles));
            vector<double> ld_part(static_cast<size_t>(n_tiles), 0.0);
            int any_bad = 0;
            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel if(n_s >= 4096)
            {
                ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                RowMatrixXd S_tile(kGramTileRows, m);
                double d_tile[kGramTileRows];
                #pragma omp for schedule(static) reduction(|:any_bad)
                for (int32_t t = 0; t < n_tiles; ++t) {
                    const int32_t r0 = t * kGramTileRows;
                    const int32_t rows = std::min<int32_t>(kGramTileRows, n_s - r0);
                    auto B_blk = B_s.middleRows(r0, rows);
                    auto S_blk = S_tile.topRows(rows);
                    int local_bad = 0;
                    for (int32_t r = 0; r < rows; ++r) {
                        const double d = w_s[r0 + r];
                        if (!(d > 0.0)) { local_bad = 1; break; }
                        d_tile[r] = d;
                        S_blk.row(r) = (1.0 / d) * B_blk.row(r);
                    }
                    if (local_bad) {
                        any_bad |= 1;
                    } else {
                        Map<const Eigen::ArrayXd> da(d_tile, rows);
                        ld_part[static_cast<size_t>(t)] = da.log().sum();
                        g_part[static_cast<size_t>(t)].noalias() = B_blk.transpose() * S_blk;
                    }
                }
            }
            if (any_bad) return false;
            for (int32_t t = 0; t < n_tiles; ++t) {
                logdet_v0 += ld_part[static_cast<size_t>(t)];
                G0.noalias() += g_part[static_cast<size_t>(t)];
            }
        }

        if (n > 0) {
            const int32_t n_tiles = (n + kGramTileRows - 1) / kGramTileRows;
            vector<MatrixXd> g_part(static_cast<size_t>(n_tiles));
            vector<double> ld_part(static_cast<size_t>(n_tiles), 0.0);
            int any_bad = 0;
            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel if(n >= 4096)
            {
                ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                RowMatrixXd S_tile(kGramTileRows, m);
                double d_tile[kGramTileRows];
                #pragma omp for schedule(static) reduction(|:any_bad)
                for (int32_t t = 0; t < n_tiles; ++t) {
                    const int32_t r0 = t * kGramTileRows;
                    const int32_t rows = std::min<int32_t>(kGramTileRows, n - r0);
                    auto B_blk = B_r_perm.middleRows(r0, rows);
                    auto S_blk = S_tile.topRows(rows);
                    int local_bad = 0;
                    for (int32_t r = 0; r < rows; ++r) {
                        const double d = w_r_perm[r0 + r];
                        if (!(d > 0.0)) { local_bad = 1; break; }
                        d_tile[r] = d;
                        S_blk.row(r) = (1.0 / d) * B_blk.row(r);
                    }
                    if (local_bad) {
                        any_bad |= 1;
                    } else {
                        Map<const Eigen::ArrayXd> da(d_tile, rows);
                        ld_part[static_cast<size_t>(t)] = da.log().sum();
                        g_part[static_cast<size_t>(t)].noalias() = B_blk.transpose() * S_blk;
                    }
                }
            }
            if (any_bad) return false;
            for (int32_t t = 0; t < n_tiles; ++t) {
                logdet_v0 += ld_part[static_cast<size_t>(t)];
                G0.noalias() += g_part[static_cast<size_t>(t)];
            }
        }

        MatrixXd XtDinvX = G0.topLeftCorner(c, c);
        Eigen::ColPivHouseholderQR<MatrixXd> x_solver(XtDinvX);
        if (!x_solver.isInvertible()) return false;
        const double logdet_x = x_solver.logAbsDeterminant();
        const VectorXd XtDinvY = G0.block(0, c, c, 1);
        const VectorXd beta0 = x_solver.solve(XtDinvY);
        const MatrixXd C0 = x_solver.solve(MatrixXd::Identity(c, c));
        const double yPy0 = G0(c, c) - XtDinvY.dot(beta0);
        out_logL0 = -0.5 * (logdet_v0 + logdet_x + yPy0);
        if (!std::isfinite(out_logL0)) return false;

        // 2. Exact score l'(0) and Fisher information I(0)
        double rKr = 0.0;
        double tr_DinvK = 0.0;
        double tr_DinvK2 = 0.0;
        MatrixXd MtKM = MatrixXd::Zero(c, c);
        MatrixXd QtDinvQ = MatrixXd::Zero(c, c);

        if (n_s > 0) {
            const double * __restrict kd_s = k_diag_s.data();
            const int32_t n_tiles = (n_s + kGramTileRows - 1) / kGramTileRows;
            vector<double> rkr_p(static_cast<size_t>(n_tiles), 0.0);
            vector<double> tr1_p(static_cast<size_t>(n_tiles), 0.0);
            vector<double> tr2_p(static_cast<size_t>(n_tiles), 0.0);
            vector<MatrixXd> mtkm_p(static_cast<size_t>(n_tiles));
            vector<MatrixXd> qtdq_p(static_cast<size_t>(n_tiles));

            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel if(n_s >= 4096)
            {
                ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                RowMatrixXd S1_tile(kGramTileRows, c);
                RowMatrixXd S2_tile(kGramTileRows, c);
                VectorXd r_tile(static_cast<Eigen::Index>(kGramTileRows));
                #pragma omp for schedule(static)
                for (int32_t t = 0; t < n_tiles; ++t) {
                    const int32_t r0 = t * kGramTileRows;
                    const int32_t rows = std::min<int32_t>(kGramTileRows, n_s - r0);
                    auto X_blk = B_s.block(r0, 0, rows, c);
                    auto y_blk = B_s.block(r0, c, rows, 1);
                    auto rt = r_tile.head(rows);
                    rt.noalias() = y_blk - X_blk * beta0;
                    auto S1 = S1_tile.topRows(rows);
                    auto S2 = S2_tile.topRows(rows);

                    double loc_rkr = 0.0, loc_tr1 = 0.0, loc_tr2 = 0.0;
                    for (int32_t r = 0; r < rows; ++r) {
                        const double inv_d = 1.0 / w_s[r0 + r];
                        const double ks = kd_s[r0 + r];
                        const double kd_ratio = ks * inv_d;
                        const double r0_i = inv_d * rt[r];
                        loc_rkr += ks * r0_i * r0_i;
                        loc_tr1 += kd_ratio;
                        loc_tr2 += kd_ratio * kd_ratio;
                        const double w1 = ks * inv_d * inv_d;
                        const double w2 = kd_ratio * kd_ratio * inv_d;
                        S1.row(r) = w1 * X_blk.row(r);
                        S2.row(r) = w2 * X_blk.row(r);
                    }
                    rkr_p[static_cast<size_t>(t)] = loc_rkr;
                    tr1_p[static_cast<size_t>(t)] = loc_tr1;
                    tr2_p[static_cast<size_t>(t)] = loc_tr2;
                    mtkm_p[static_cast<size_t>(t)].noalias() = X_blk.transpose() * S1;
                    qtdq_p[static_cast<size_t>(t)].noalias() = X_blk.transpose() * S2;
                }
            }
            for (int32_t t = 0; t < n_tiles; ++t) {
                rKr += rkr_p[static_cast<size_t>(t)];
                tr_DinvK += tr1_p[static_cast<size_t>(t)];
                tr_DinvK2 += tr2_p[static_cast<size_t>(t)];
                MtKM.noalias() += mtkm_p[static_cast<size_t>(t)];
                QtDinvQ.noalias() += qtdq_p[static_cast<size_t>(t)];
            }
        }

        if (n > 0) {
            VectorXd inv_dr(n);
            VectorXd r_r(n);
            RowMatrixXd M_r(n, c);
            auto X_r = B_r_perm.leftCols(c);
            auto y_r = B_r_perm.col(c);
            {
                ScopedSingleThreadBLAS blas_guard;
                #pragma omp parallel if(n >= 4096)
                {
                    ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                    VectorXd r_tile(static_cast<Eigen::Index>(kGramTileRows));
                    #pragma omp for schedule(static)
                    for (int32_t r0 = 0; r0 < n; r0 += kGramTileRows) {
                        const int32_t rows = std::min<int32_t>(kGramTileRows, n - r0);
                        auto rt = r_tile.head(rows);
                        rt.noalias() = y_r.segment(r0, rows) - X_r.middleRows(r0, rows) * beta0;
                        for (int32_t r = 0; r < rows; ++r) {
                            const int32_t k = r0 + r;
                            const double inv_d = 1.0 / w_r_perm[k];
                            inv_dr[k] = inv_d;
                            r_r[k] = inv_d * rt[r];
                            M_r.row(k) = inv_d * X_r.row(k);
                        }
                    }
                }
            }

            VectorXd q_r = VectorXd::Zero(n);
            RowMatrixXd Q_r = RowMatrixXd::Zero(n, c);
            const int32_t * __restrict ap_p = ap_col_ptr.data();
            const int32_t * __restrict ap_i = ap_row_idx.data();
            const double * __restrict ap_v = ap_val.data();
            const double * __restrict ap_d = ap_diag.data();
            const double * __restrict idr = inv_dr.data();

            double loc_tr1 = 0.0;
            double loc_tr2 = 0.0;
            for (int32_t k = 0; k < n; ++k) {
                const double dk = ap_d[k];
                const double ik = idr[k];
                const double kd = dk * ik;
                loc_tr1 += kd;
                loc_tr2 += kd * kd;
                q_r[k] += dk * r_r[k];
                Q_r.row(k) += dk * M_r.row(k);

                for (int32_t idx = ap_p[k]; idx < ap_p[k + 1]; ++idx) {
                    const int32_t i = ap_i[idx];
                    const double v = ap_v[idx];
                    loc_tr2 += 2.0 * (v * v) * idr[i] * ik;
                    q_r[i] += v * r_r[k];
                    q_r[k] += v * r_r[i];
                    Q_r.row(i) += v * M_r.row(k);
                    Q_r.row(k) += v * M_r.row(i);
                }
            }
            rKr += r_r.dot(q_r);
            tr_DinvK += loc_tr1;
            tr_DinvK2 += loc_tr2;

            const int32_t n_rtiles = (n + kGramTileRows - 1) / kGramTileRows;
            vector<MatrixXd> mtkm_r(static_cast<size_t>(n_rtiles));
            vector<MatrixXd> qtdq_r(static_cast<size_t>(n_rtiles));
            {
                ScopedSingleThreadBLAS blas_guard;
                #pragma omp parallel if(n >= 4096)
                {
                    ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                    RowMatrixXd S_tile(kGramTileRows, c);
                    #pragma omp for schedule(static)
                    for (int32_t t = 0; t < n_rtiles; ++t) {
                        const int32_t r0 = t * kGramTileRows;
                        const int32_t rows = std::min<int32_t>(kGramTileRows, n - r0);
                        auto M_blk = M_r.middleRows(r0, rows);
                        auto Q_blk = Q_r.middleRows(r0, rows);
                        auto S_blk = S_tile.topRows(rows);
                        mtkm_r[static_cast<size_t>(t)].noalias() = M_blk.transpose() * Q_blk;
                        for (int32_t r = 0; r < rows; ++r) {
                            S_blk.row(r) = idr[r0 + r] * Q_blk.row(r);
                        }
                        qtdq_r[static_cast<size_t>(t)].noalias() = Q_blk.transpose() * S_blk;
                    }
                }
            }
            for (int32_t t = 0; t < n_rtiles; ++t) {
                MtKM.noalias() += mtkm_r[static_cast<size_t>(t)];
                QtDinvQ.noalias() += qtdq_r[static_cast<size_t>(t)];
            }
        }

        const MatrixXd C0_MtKM = C0 * MtKM;
        const double tr_P0K = tr_DinvK - C0_MtKM.trace();
        out_score0 = 0.5 * (rKr - tr_P0K);

        const double tr_P0KP0K = tr_DinvK2 - 2.0 * (C0 * QtDinvQ).trace() + (C0_MtKM * C0_MtKM).trace();
        out_fisher0 = 0.5 * std::max(0.0, tr_P0KP0K);
        return std::isfinite(out_score0) && std::isfinite(out_fisher0);
    }

    void permuteWeights(const Ref<const VectorXd> &v_orig, vector<double> &v_s, vector<double> &v_r_perm) const {
        v_s.resize(n_s);
        double * __restrict vs_dst = v_s.data();
        const double * __restrict src = v_orig.data();
        const int32_t * __restrict so = s_orig.data();
        #pragma omp parallel for schedule(static) if(n_s >= 4096)
        for (int32_t s = 0; s < n_s; ++s) {
            vs_dst[s] = src[so[s]];
        }
        v_r_perm.resize(n);
        double * __restrict vr_dst = v_r_perm.data();
        const int32_t * __restrict rpo = r_perm_orig.data();
        #pragma omp parallel for schedule(static) if(n >= 4096)
        for (int32_t k = 0; k < n; ++k) {
            vr_dst[k] = src[rpo[k]];
        }
    }

    bool factorizeRelatedOnly(const double * __restrict w_r_perm, double cur_tao, Workspace &ws, double &out_logdet_r) const {
        if (n == 0) {
            out_logdet_r = 0.0;
            return true;
        }
        double * __restrict diag = ws.diag.data();
        double * __restrict inv_diag = ws.inv_diag.data();
        if (ws.is_diag_only) {
            for (int32_t k = 0; k < n; ++k) {
                const double d = w_r_perm[k];
                if (!(d > 0.0)) {
                    return false;
                }
                diag[k] = d;
                inv_diag[k] = 1.0 / d;
            }
        } else {
            double * __restrict Lx = ws.Lx.data();
            double * __restrict y = ws.y.data();
            const int32_t * __restrict lp = Lp.data();
            const int32_t * __restrict li = Li.data();
            const int32_t * __restrict ap_p = ap_col_ptr.data();
            const int32_t * __restrict ap_i = ap_row_idx.data();
            const double * __restrict ap_v = ap_val.data();
            const double * __restrict ap_d = ap_diag.data();
            const int32_t * __restrict rp = row_ptr.data();
            const int32_t * __restrict rc = row_col.data();
            const int32_t * __restrict rl = row_lpos.data();

            for (int32_t k = 0; k < n; ++k) {
                for (int32_t idx = ap_p[k]; idx < ap_p[k + 1]; ++idx) {
                    y[ap_i[idx]] = cur_tao * ap_v[idx];
                }
                double d = w_r_perm[k] + cur_tao * ap_d[k];
                const int32_t r_end = rp[k + 1];
                for (int32_t r = rp[k]; r < r_end; ++r) {
                    const int32_t i = rc[r];
                    const int32_t p2 = rl[r];
                    const double yi = y[i];
                    y[i] = 0.0;
                    const double l_ki = yi / diag[i];
                    for (int32_t p = lp[i]; p < p2; ++p) {
                        y[li[p]] -= Lx[p] * yi;
                    }
                    d -= l_ki * yi;
                    Lx[p2] = l_ki;
                }
                if (!(d > 0.0)) {
                    return false;
                }
                diag[k] = d;
                inv_diag[k] = 1.0 / d;
            }
        }

        Map<const VectorXd> dr_vec(diag, n);
        out_logdet_r = dr_vec.array().log().sum();
        return std::isfinite(out_logdet_r);
    }

    bool factorize(const double * __restrict w_s, const double * __restrict w_r_perm, double cur_tao, Workspace &ws, double &out_logdet) const {
        ws.is_diag_only = (cur_tao == 0.0);
        if (n_total == 0) {
            out_logdet = 0.0;
            return true;
        }
        double logdet_r = 0.0;
        if (!factorizeRelatedOnly(w_r_perm, cur_tao, ws, logdet_r)) {
            return false;
        }
        double logdet_s = 0.0;
        if (n_s > 0) {
            double * __restrict d_s = ws.diag_s.data();
            double * __restrict inv_d_s = ws.inv_diag_s.data();
            const double * __restrict kd_s = k_diag_s.data();
            const int32_t kTile = 1024;
            const int32_t n_tiles = (n_s + kTile - 1) / kTile;
            vector<double> tile_logs(static_cast<size_t>(n_tiles), 0.0);
            int any_bad = 0;
            const bool is_diag = ws.is_diag_only;

            #pragma omp parallel for schedule(static) reduction(|:any_bad) if(n_s >= 4096)
            for (int32_t t = 0; t < n_tiles; ++t) {
                const int32_t s0 = t * kTile;
                const int32_t rows = std::min<int32_t>(kTile, n_s - s0);
                int local_bad = 0;
                if (is_diag) {
                    for (int32_t r = 0; r < rows; ++r) {
                        const double d = w_s[s0 + r];
                        if (!(d > 0.0)) {
                            local_bad = 1;
                            break;
                        }
                        d_s[s0 + r] = d;
                        inv_d_s[s0 + r] = 1.0 / d;
                    }
                } else {
                    for (int32_t r = 0; r < rows; ++r) {
                        const double d = w_s[s0 + r] + cur_tao * kd_s[s0 + r];
                        if (!(d > 0.0)) {
                            local_bad = 1;
                            break;
                        }
                        d_s[s0 + r] = d;
                        inv_d_s[s0 + r] = 1.0 / d;
                    }
                }
                if (local_bad) {
                    any_bad |= 1;
                } else {
                    Map<const Eigen::ArrayXd> ds_blk(d_s + s0, rows);
                    tile_logs[static_cast<size_t>(t)] = ds_blk.log().sum();
                }
            }
            if (any_bad) {
                return false;
            }
            for (int32_t t = 0; t < n_tiles; ++t) {
                logdet_s += tile_logs[static_cast<size_t>(t)];
            }
            if (!std::isfinite(logdet_s)) {
                return false;
            }
        }

        out_logdet = logdet_s + logdet_r;
        return std::isfinite(out_logdet);
    }

    void solveMatrix(const Ref<const MatrixXd> &rhs, MatrixXd &out, Workspace &ws) const {
        const Eigen::Index cols = rhs.cols();
        out.resize(n_total, cols);
        if (n_total == 0) return;

        const int32_t * __restrict so = s_orig.data();
        const double * __restrict inv_d_s = ws.inv_diag_s.data();
        const int32_t * __restrict rpo = r_perm_orig.data();
        const int32_t * __restrict lp = Lp.data();
        const int32_t * __restrict li = Li.data();
        const double * __restrict Lx = ws.Lx.data();
        const double * __restrict inv_diag = ws.inv_diag.data();

        #pragma omp parallel if(cols > 1 && n_total >= 4096)
        {
            vector<double> local_dst((n > 0 && !ws.is_diag_only && nnz_L > 0) ? n : 0);
            double * __restrict dst = local_dst.data();

            #pragma omp for schedule(static)
            for (Eigen::Index col = 0; col < cols; ++col) {
                const double * __restrict src = rhs.col(col).data();
                double * __restrict out_col = out.col(col).data();

                for (int32_t s = 0; s < n_s; ++s) {
                    const int32_t idx = so[s];
                    out_col[idx] = src[idx] * inv_d_s[s];
                }

                if (n > 0) {
                    if (ws.is_diag_only || nnz_L == 0) {
                        for (int32_t k = 0; k < n; ++k) {
                            const int32_t idx = rpo[k];
                            out_col[idx] = src[idx] * inv_diag[k];
                        }
                    } else {
                        for (int32_t k = 0; k < n; ++k) {
                            dst[k] = src[rpo[k]];
                        }
                        for (int32_t i = 0; i < n; ++i) {
                            const double tmp = dst[i];
                            if (tmp != 0.0) {
                                for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                                    dst[li[p]] -= tmp * Lx[p];
                                }
                            }
                        }
                        for (int32_t i = 0; i < n; ++i) {
                            dst[i] *= inv_diag[i];
                        }
                        for (int32_t i = n - 1; i >= 0; --i) {
                            double tmp = dst[i];
                            for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                                tmp -= Lx[p] * dst[li[p]];
                            }
                            dst[i] = tmp;
                        }
                        for (int32_t k = 0; k < n; ++k) {
                            out_col[rpo[k]] = dst[k];
                        }
                    }
                }
            }
        }
    }

    void solveVector(const Ref<const VectorXd> &rhs, VectorXd &out, Workspace &ws) const {
        out.resize(n_total);
        if (n_total == 0) return;

        const double * __restrict src = rhs.data();
        double * __restrict out_ptr = out.data();
        const int32_t * __restrict so = s_orig.data();
        const double * __restrict inv_d_s = ws.inv_diag_s.data();
        #pragma omp parallel for schedule(static) if(n_s >= 4096)
        for (int32_t s = 0; s < n_s; ++s) {
            const int32_t idx = so[s];
            out_ptr[idx] = src[idx] * inv_d_s[s];
        }

        if (n > 0) {
            const int32_t * __restrict rpo = r_perm_orig.data();
            const double * __restrict inv_diag = ws.inv_diag.data();
            if (ws.is_diag_only || nnz_L == 0) {
                for (int32_t k = 0; k < n; ++k) {
                    const int32_t idx = rpo[k];
                    out_ptr[idx] = src[idx] * inv_diag[k];
                }
            } else {
                ws.solve_buf.resize(n, 1);
                double * __restrict dst = ws.solve_buf.data();
                const int32_t * __restrict lp = Lp.data();
                const int32_t * __restrict li = Li.data();
                const double * __restrict Lx = ws.Lx.data();

                for (int32_t k = 0; k < n; ++k) {
                    dst[k] = src[rpo[k]];
                }
                for (int32_t i = 0; i < n; ++i) {
                    const double tmp = dst[i];
                    if (tmp != 0.0) {
                        for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                            dst[li[p]] -= tmp * Lx[p];
                        }
                    }
                }
                for (int32_t i = 0; i < n; ++i) {
                    dst[i] *= inv_diag[i];
                }
                for (int32_t i = n - 1; i >= 0; --i) {
                    double tmp = dst[i];
                    for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                        tmp -= Lx[p] * dst[li[p]];
                    }
                    dst[i] = tmp;
                }
                for (int32_t k = 0; k < n; ++k) {
                    out_ptr[rpo[k]] = dst[k];
                }
            }
        }
    }

    void prepareGrammarTuning(const Ref<const MatrixXd> &covar_mat, MatrixXd &ViX_out) {
        solveMatrix(covar_mat, ViX_out, main_ws);
        MatrixXd XtVX = parallelAtB(covar_mat, ViX_out);
        auto XtVX_solver = XtVX.colPivHouseholderQr();
        if (!XtVX_solver.isInvertible()) {
            LOGGER.e(0, "XtViX is not invertible.");
        }
        const Eigen::Index c = covar_mat.cols();
        XtVX_inv = XtVX_solver.solve(MatrixXd::Identity(c, c));

        inv_d_diag_orig.assign(n_total, 0.0);
        double * __restrict inv_d_orig = inv_d_diag_orig.data();
        const int32_t * __restrict so = s_orig.data();
        const double * __restrict inv_ds = main_ws.inv_diag_s.data();
        #pragma omp parallel for schedule(static) if(n_s >= 4096)
        for (int32_t s = 0; s < n_s; ++s) {
            inv_d_orig[so[s]] = inv_ds[s];
        }
        if (main_ws.is_diag_only || nnz_L == 0) {
            const int32_t * __restrict rpo = r_perm_orig.data();
            const double * __restrict inv_dr = main_ws.inv_diag.data();
            #pragma omp parallel for schedule(static) if(n >= 4096)
            for (int32_t k = 0; k < n; ++k) {
                inv_d_orig[rpo[k]] = inv_dr[k];
            }
        }
    }

    double evalGrammarGamma(const Ref<const VectorXd> &curGeno,
                            const Ref<const VectorXd> &dWp_vec,
                            const Ref<const MatrixXd> &ViX_mat,
                            VectorXd &u_r_buf,
                            VectorXd &w_buf) const {
        const double * __restrict g = curGeno.data();
        const double * __restrict w_p = dWp_vec.data();
        const double * __restrict inv_ds = inv_d_diag_orig.data();

        double quad_v = 0.0;
        double denom = 0.0;
        for (int32_t idx = 0; idx < n_total; ++idx) {
            const double gi = g[idx];
            const double gi2 = gi * gi;
            quad_v += inv_ds[idx] * gi2;
            denom += w_p[idx] * gi2;
        }

        if (n > 0 && !main_ws.is_diag_only && nnz_L > 0) {
            u_r_buf.resize(n);
            double * __restrict dst = u_r_buf.data();
            const int32_t * __restrict rpo = r_perm_orig.data();
            const int32_t * __restrict lp = Lp.data();
            const int32_t * __restrict li = Li.data();
            const double * __restrict Lx = main_ws.Lx.data();
            const double * __restrict inv_diag = main_ws.inv_diag.data();

            for (int32_t k = 0; k < n; ++k) {
                dst[k] = g[rpo[k]];
            }
            for (int32_t i = 0; i < n; ++i) {
                const double tmp = dst[i];
                if (tmp != 0.0) {
                    for (int32_t p = lp[i]; p < lp[i + 1]; ++p) {
                        dst[li[p]] -= tmp * Lx[p];
                    }
                }
            }
            double quad_r = 0.0;
            for (int32_t i = 0; i < n; ++i) {
                quad_r += inv_diag[i] * dst[i] * dst[i];
            }
            quad_v += quad_r;
        }

        w_buf.noalias() = ViX_mat.transpose() * curGeno;
        const double proj = w_buf.dot(XtVX_inv * w_buf);
        return (quad_v - proj) / denom;
    }
};

constexpr int32_t FastFAM::BinSparseLDLT::kGramTileRows;

struct MDLHeader{
    char magic[5];
    uint32_t num_indi;
    bool fam_flag;
    bool isGrammar;
    uint64_t phenoVec_start;
    bool covarFlag;
    uint64_t covarVec_start;
    uint32_t covarVec_cols;
    uint64_t V_inverse_start;
    uint32_t V_inverse_items;
    double c_inf;
    bool has_envir;
    double VR_copy;
    uint64_t envirVec_start;
};
    


using std::to_string;
using Eigen::Matrix;
using Eigen::ArrayXd;

map<string, string> FastFAM::options;
map<string, double> FastFAM::options_d;
vector<string> FastFAM::processFunctions;


struct K1Res{
    double root;
    int nIter;
    bool bConverge;
};

void printVector(const vector<double> &v, string label){
    LOGGER << label << ": ";
    for(int i = 0; i < v.size(); i++){
        LOGGER << v[i] << " ";
    }
    LOGGER << std::endl;
}

class SPA{
private:
    double q, qinv;
    double gPos, gNeg;
    static ArrayXd mu;
    static int nSample;
    ArrayXd mu1muG;
    ArrayXd geno;
    bool bFast;
    ArrayXd muNZ;
    ArrayXd mu1muNZG;
    ArrayXd gNZ;
    double NAmu;
    double NAsigma;

    double K1adj(double t1, double q){
        //ArrayXd temp1 = mu1 * (-geno * t1).exp() + mu;
        //ArrayXd temp2 = mu * geno;
        if(bFast){
            double temp2dtemp1 = ((muNZ * gNZ)/((1.0 - muNZ) * (-gNZ * t1).exp() + muNZ)).sum();
            return temp2dtemp1 + NAmu + NAsigma * t1 - q;
        }else{
            return ((mu * geno)/((1.0 - mu) * (-geno * t1).exp() + mu)).sum() - q;
        }
    }

    double K2(double t1){
        if(bFast){
            ArrayXd genot1exp = (-gNZ * t1).exp();
            ArrayXd div21 = (mu1muNZG * genot1exp) / ((1.0 - muNZ) * genot1exp + muNZ).square();
            return ((!div21.isNaN()).select(div21, 0)).sum() + NAsigma;
        }else{
            ArrayXd genot1exp = (-geno * t1).exp();
            ArrayXd div21 = (mu1muG * genot1exp) / ((1.0 - mu) * genot1exp + mu).square();
            return ((!div21.isNaN()).select(div21, 0)).sum();
        }
    }

    double Korg(double t1){
        return (1.0 - mu + mu * (geno * t1).exp()).log().sum();
    }

    struct KS{ double k1; double k2;};

    inline void KorgK2(double t1, KS *ks){
        if(bFast){
            ArrayXd genot1exp = (-gNZ * t1).exp();
            ArrayXd div21 = (mu1muNZG * genot1exp) / ((1.0 - muNZ) * genot1exp + muNZ).square();
            ks->k2 = ((!div21.isNaN()).select(div21, 0)).sum() + NAsigma;
            ks->k1 = (1.0 - muNZ + muNZ / genot1exp).log().sum() + NAmu * t1 + 0.5 * NAsigma * t1 * t1;
        }else{
            ArrayXd genot1exp = (-geno * t1).exp();
            ArrayXd div21 = (mu1muG * genot1exp) / ((1.0 - mu) * genot1exp + mu).square();
            ks->k2 = ((!div21.isNaN()).select(div21, 0)).sum();
            ks->k1 = (1.0 - mu + mu / genot1exp).log().sum();
        }
    }

    double getSaddleProb(double zeta, double q){
        KS ks;
        KorgK2(zeta, &ks);
        //double k1 = Korg(zeta);
        //double k2 = K2(zeta);
        double k1 = ks.k1;
        double k2 = ks.k2;
        double pval = 0;
        if(std::isfinite(k1) && std::isfinite(k2)){
            double temp1 = zeta * q - k1;
            double w = sgn(zeta) * std::sqrt(2.0) * std::sqrt(temp1);
            double wi = 1.0 / w;
            double v = zeta * std::sqrt(k2);
            double z_test = w + wi * std::log(v * wi); 
            if(z_test > 0){
                pval = StatLib::pnorm(z_test, false);
            }else{
                pval = StatLib::pnorm(z_test, true);
            }
        }
        return(pval);
    }

    K1Res getRootK1(double init, double q, double thresh=0.0001220703, int maxIter=1000){
        K1Res res;
        if(q >= gPos || q <= gNeg){
            res.root = std::numeric_limits<double>::infinity();
            res.nIter = 0;
            res.bConverge = true;
            return(res);
        }else{
            double t1 = init;
            double K1eval = K1adj(t1, q);
            double prevJump = std::numeric_limits<double>::infinity();
            int nIter = 1;
            while(true){
                double K2eval = K2(t1);
                double tnew = t1 - K1eval / K2eval;
                if(!std::isfinite(tnew)){
                    res.bConverge = false;
                    break;
                }
                if(std::abs(tnew - t1) <= thresh){
                    res.bConverge = true;
                    break;
                }
                if(nIter == maxIter){
                    res.bConverge = false;
                    break;
                }
                double newK1 = K1adj(tnew, q);
                if(sgn(K1eval) != sgn(newK1)){
                    double absTnewT1 = std::abs(tnew - t1);
                    if(absTnewT1 > prevJump - thresh){
                        tnew = t1 + sgn(newK1 - K1eval) * prevJump * 0.5;
                        newK1 = K1adj(tnew, q);
                        prevJump = prevJump * 0.5;
                    }else{
                        prevJump = absTnewT1;
                    }
                }
                nIter++;
                t1 = tnew;
                K1eval = newK1;
            }
            res.root = t1;
            res.nIter = nIter;
            /*
               if(!res.bConverge){
               printVector(K1evals, "K1eval");
               printVector(prevJumps, "prevJump");
               printVector(K2evals, "K2eval");
               printVector(newK1s, "newK1");
               printVector(tnews, "tnew");
            //output the tempory
            }
            */
            return(res);
        }
    }

public:
    static void setMu(VectorXd mu){
        SPA::mu = mu.array();
        //SPA::mu1 = 1.0 - SPA::mu;
        //SPA::mu1mu = SPA::mu1 * SPA::mu;
        SPA::nSample = mu.size();
    }

    SPA(double q, double qinv, Ref<VectorXd> rgen, const vector<uint32_t> &index){
        this->q = q;
        this->qinv = qinv;
        //new (&geno) Map<VectorXd>(xp, nSample);
        this->geno = rgen.array();
        //LOGGER << "geno address: " << static_cast<void *>(geno.data()) << std::endl;
        //LOGGER << "orig address: " << static_cast<void *>(rgen.data()) << std::endl;
        //LOGGER.e(0, "debug");
        //double gPos = (geno.array()>0).select(geno, 0).sum();
        //double gNeg = (geno.array()<0).select(geno, 0).sum();
        gPos = 0;
        gNeg = 0;
        for(int i = 0; i < nSample; i++){
            double tgeno = geno[i];
            if(tgeno > 0){
                gPos+=tgeno;
            }else{
                gNeg+=tgeno;
            }
        }

        mu1muG = mu * (1.0 - mu) * geno.square();

        int nNZ = index.size();
        bFast = false;
        if((double)nNZ / nSample < 0.5){
            bFast = true;
            muNZ.resize(nNZ);
            mu1muNZG.resize(nNZ);
            gNZ.resize(nNZ);
            int curIndex = 0;
            for(int i=0; i < nNZ; i++){
                int tempIndex = index[i];
                muNZ[curIndex] = mu[tempIndex];
                mu1muNZG[curIndex] = mu1muG[tempIndex];
                gNZ[curIndex] = geno[tempIndex];
                curIndex++;
            }

            NAmu = (qinv + q) * 0.5 - (gNZ * muNZ).sum();
            NAsigma = mu1muG.sum() - mu1muNZG.sum();

        }
    }

    void saddleProb(SPARes *res){
        K1Res k1Res = getRootK1(0, q);
        K1Res k2Res = getRootK1(0, qinv);
        if(k1Res.bConverge && k2Res.bConverge){
            double p1 = getSaddleProb(k1Res.root, q);
            double p2 = getSaddleProb(k2Res.root, qinv);
            res->p_adj = std::abs(p1) + std::abs(p2);
            res->bConverge = true;
        }else{
            res->p_adj = std::numeric_limits<double>::quiet_NaN();
            res->bConverge = false;
        }

        if(res->p_adj != 0 && res->p / res->p_adj > 1000){
            res->p_adj = res->p;
            res->bConverge = false;
        }
    }

};
ArrayXd SPA::mu;
int SPA::nSample = 0;

void FastFAM::loadBinModel(){
        LOGGER << "Loading saved GLM model file prefixed with [" << options["model_file"] << "]..." << std::endl;
        LOGGER << "Note: phenotype, covariates, sparse GRM and association test methods are included in the model file, thus these flags will be ignored." << std::endl;
        string model_file = options["model_file"] + ".mdl";

        string id_file = model_file + ".id";
        vector<string> modelIDs = Pheno::read_sublist(id_file);
        num_indi = modelIDs.size();

        uint32_t sample_keep_geno = pheno->count_keep();
        vector<string> sampleIDs = pheno->get_id(0, sample_keep_geno - 1, "\t");
        vector<uint32_t> id1, id2;
        vector_commonIndex_sorted1(sampleIDs, modelIDs, id1, id2);
        if(id2.size() != num_indi){
            LOGGER.e(0, "some sample IDs in the saved model file do not exist in the genotype file.");
        }

        pheno->filter_keep_index(id1);
        LOGGER << "  " << num_indi << " valid individuals to be included." << std::endl;

        fam_flag = true;
        covarFlag = true;
        if(fam_flag){
            options["grmsparse_file"] = "demo";
        }

        string bin_file = model_file + ".bin2";
        FILE* ibin = fopen(bin_file.c_str(), "rb");
        if(!ibin){
            LOGGER.e(0, "can't open file [" + bin_file + "] to read.");
        }
        char magic[5];
        if(fread(magic, sizeof(char), 5, ibin) != 5){
            LOGGER.e(0, "can't read magic number.");
        }
        if(strcmp(magic, "fGLM") != 0){
            LOGGER.e(0, "wrong header in [" + bin_file + "]. This file can only be generated from GCTA.");
        }

        uint32_t num_indi2;
        if(fread(&num_indi2, sizeof(uint32_t), 1, ibin) != 1){
            LOGGER.e(0, "can't read sample size.");
        }
        if(num_indi != num_indi2){
            LOGGER.e(0, "incorrect number of individuals in [" + bin_file + "]. This file can only be generated from GCTA.");
        }

        uint32_t covar_col;
        if(fread(&covar_col, sizeof(uint32_t), 1, ibin) != 1){
            LOGGER.e(0, "can't read number of covariates in [" + bin_file + "]. This file can only be generated from GCTA.");
        }

        num_covar = covar_col;

        if(fread(&bPreciseCovar, sizeof(bool), 1, ibin) != 1){
            LOGGER.e(0, "can't read covariate flag");
        }

        if(fread(&taoVal, sizeof(double), 1, ibin) != 1){
            LOGGER.e(0, "can't find tao value in [" + bin_file + "]. This file can only be generated from GCTA.");
        }

        if(fread(&c_inf, sizeof(double), 1, ibin) != 1){
            LOGGER.e(0, "can't find c_inf in [" + bin_file + "]. This file can only be generated from GCTA.");
        }
        mu.resize(num_indi);
        phenoVec.resize(num_indi);
        covar.resize(num_indi, covar_col);
        H.resize(covar_col, num_indi);

        double *tempVec = new double[num_indi];
        if(fread(tempVec, sizeof(double), num_indi, ibin) != num_indi){
            LOGGER.e(0, "failed to read mu.");
        }
        //switch order;
        for(int i = 0; i < num_indi; i++){
            mu[i] = tempVec[id2[i]];
        }

        if(fread(tempVec, sizeof(double), num_indi, ibin) != num_indi){
            LOGGER.e(0, "failed to read phenotype.");
        }
        //switch order;
        for(int i = 0; i < num_indi; i++){
            phenoVec[i] = tempVec[id2[i]];
        }
        delete[] tempVec;
        uint64_t num_covar_elements = (uint64_t)num_indi * covar_col;
        double *tempCovar = new double[num_covar_elements];
        //covariate
        if(fread(tempCovar, sizeof(double), num_covar_elements, ibin) != num_covar_elements){
            LOGGER.e(0, "failed to read covariates from the saved model.");
        }
        for(uint64_t i = 0; i < covar_col; i++){
            uint64_t cur_base = i * num_indi;
            for(uint64_t j = 0; j < num_indi; j++){
                covar(j, i) = tempCovar[cur_base + id2[j]];
            }
        }
        // H
        if(fread(tempCovar, sizeof(double), num_covar_elements, ibin) != num_covar_elements){
            LOGGER.e(0, "failed to read H from the saved model.");
        }
        for(uint64_t i = 0; i < num_indi; i++){
            uint64_t cur_base = id2[i] * covar_col;
            for(uint64_t j = 0; j < covar_col; j++){
                H(j, i) = tempCovar[cur_base + j];
            }
        }
        delete[] tempCovar;

        initVar();

        fclose(ibin);
        LOGGER << "  loaded successfully." << std::endl;
}

void FastFAM::loadModel(){
        bBinary = false;
        LOGGER << "Loading saved model file prefixed with [" << options["model_file"] << "]..." << std::endl;
        LOGGER << "Note: phenotype, covariates, sparse GRM and association test methods are included in the model file, thus these flags will be ignored." << std::endl;
        string model_file = options["model_file"] + ".mdl";

        string id_file = model_file + ".id";
        vector<string> modelIDs = Pheno::read_sublist(id_file);
        num_indi = modelIDs.size();

        uint32_t sample_keep_geno = pheno->count_keep();
        vector<string> sampleIDs = pheno->get_id(0, sample_keep_geno - 1, "\t");
        vector<uint32_t> id1, id2;
        vector_commonIndex_sorted1(sampleIDs, modelIDs, id1, id2);
        if(id2.size() != num_indi){
            LOGGER.e(0, "Some sample IDs in the saved model file do not exist in genotype file!");
        }

        pheno->filter_keep_index(id1);
        LOGGER << "  " << num_indi << " valid individuals to be included." << std::endl;

        MDLHeader head;
        string bin_file = model_file + ".bin";
        FILE* ibin = fopen(bin_file.c_str(), "rb");
        if(!ibin){
            LOGGER.e(0, "can't open file [" + bin_file + "] to read.");
        }
        if(fread(&head, sizeof(MDLHeader), 1, ibin) != 1){
            LOGGER.e(0, "can't read header in [" + bin_file + "].");
        }
        if(strcmp(head.magic, "fGWA") != 0){
            LOGGER.e(0, "incorrect header in [" + bin_file + "]. This file can only be generated from GCTA model.");
        }
        if(head.num_indi != modelIDs.size()){
            LOGGER.e(0, "the sample IDs in the saved model is different from that in genotype data file!");
        }
        fam_flag = head.fam_flag;
        if(fam_flag){
            options["grmsparse_file"] = "demo";
        }

        bGrammar = head.isGrammar;
        if(fam_flag && bGrammar){
            options["grammar"] = "yes";
        }

        covarFlag = head.covarFlag;
        c_inf = head.c_inf;

        has_envir = head.has_envir;
        if (has_envir){
            options["envir"] = "yes";
        }

        VR_copy = head.VR_copy;

        // read the phenotype
        LOGGER << "  loading phenotypes..." << std::endl;
        double *phenoP;
        double *tempP = new double[num_indi];
        if (has_envir){
            Vi_y.resize(num_indi);
            phenoP = Vi_y.data();
        }else if((fam_flag && (!bGrammar)) || (!fam_flag)){
            phenoVec.resize(num_indi);
            phenoP = phenoVec.data();       
        }else{
            Vi_y_cinf.resize(num_indi);
            phenoP = Vi_y_cinf.data();
        }
        fseek(ibin, head.phenoVec_start, SEEK_SET);
        if(fread(tempP, sizeof(double), num_indi, ibin) != num_indi){
            LOGGER.e(0, "failed to read phenotype from the saved model.");
        }
        // switch order;
        for(int i = 0; i < num_indi; i++){
            phenoP[i] = tempP[id2[i]];
        }
        delete[] tempP;

        // read covariates
        LOGGER << "  loading covariates..." << std::endl;
        uint64_t num_covar_elements = (uint64_t)num_indi * head.covarVec_cols;
        fseek(ibin, head.covarVec_start, SEEK_SET);
        this->covar.resize(num_indi, head.covarVec_cols);
        double *tempCovar = new double[num_covar_elements];
        if(fread(tempCovar, sizeof(double), num_covar_elements, ibin) != num_covar_elements){
            LOGGER.e(0, "failed to read covariates from the saved model.");
        }
        for(uint64_t i = 0; i < head.covarVec_cols; i++){
            uint64_t cur_base = i * num_indi;
            for(uint64_t j = 0; j < num_indi; j++){
                covar(j, i) = tempCovar[cur_base + id2[j]];
            }
        }

        if(covarFlag){
            LOGGER << "  loading " << head.covarVec_cols << " covariates..." << std::endl;
            //TODO  H seems incorrect;
            H.resize(num_indi, head.covarVec_cols);
            if(fread(tempCovar, sizeof(double), num_covar_elements, ibin) != num_covar_elements){
                LOGGER.e(0, "failed to read covariates from the saved model.");
            }
            for(uint64_t i = 0; i < head.covarVec_cols; i++){
                uint64_t cur_base = i * num_indi;
                for(uint64_t j = 0; j < num_indi; j++){
                    H(j, i) = tempCovar[cur_base + id2[j]];
                }
            }
        }
        delete[] tempCovar;

        //read the environment variable
        if (has_envir){
            LOGGER << "  loading the environment variable..." << std::endl;
            double *envirE;
            double *tempE = new double[num_indi];
            envirVec_scaled.resize(num_indi);
            envirE = envirVec_scaled.data();
            fseek(ibin, head.envirVec_start, SEEK_SET);
            if(fread(tempE, sizeof(double), num_indi, ibin) != num_indi){
                LOGGER.e(0, "failed to read the environment variable from the saved model.");
            }
            // switch order;
            for(int i = 0; i < num_indi; i++){
                envirE[i] = tempE[id2[i]];
            }
            delete[] tempE;
        }

        if(fam_flag && (!bGrammar)){
            LOGGER << "  loading V matrix..." << std::endl;
            vector<size_t> lookup = sort_indexes(id2);

            vector<uint32_t> vi, vj;
            vector<double> val;
            vi.resize(head.V_inverse_items);
            vj.resize(head.V_inverse_items);
            val.resize(head.V_inverse_items);

            V_inverse.resize(num_indi, num_indi);

            uint32_t cur_item = 0;
            InvItem item;
            while(cur_item < head.V_inverse_items){
                if(fread(&item, sizeof(item), 1, ibin) == 1){
                    //V_inverse.insert(item.row, item.col) = item.val;
                    vi[cur_item] = lookup[item.row];
                    vj[cur_item] = lookup[item.col];
                    val[cur_item] = item.val; 

                    cur_item++;
                }else{
                    LOGGER.e(0, "failed to read the V inverse from the saved model.");
                }
            }
            //reorder
            vector<size_t> ordIndex = sort_indexes(vj, vi);
            for(int i = 0; i < head.V_inverse_items; i++){
                uint32_t idx = ordIndex[i]; 
                V_inverse.insert(vi[idx], vj[idx]) = val[idx];
            }
        }

        V_inverse.finalize();
        V_inverse.makeCompressed();
        fclose(ibin);
        LOGGER << "  loaded successfully." << std::endl;
}

FastFAM::FastFAM(){
    //Eigen::setNbThreads(THREADS.getThreadCount() + 1);
    //Eigen::setNbThreads(1);
    
    pheno = new Pheno();
    marker = new Marker();
    this->geno = new Geno(pheno, marker);
    hasInfo = geno->getGenoHasInfo();

    num_indi = pheno->count_keep();

    if(options.find("binary") != options.end()){
        bBinary = true;
    }
 
    if(options["model_file"] != ""){
        if(bBinary){
            loadBinModel();
        }else{
            loadModel();
        }
        return;
    }

    if(options.find("grammar") !=options.end()){
        bGrammar = true;
    }


    if(options_d["seed"] == 0){
        seed = pheno->getSeed();
    }else{
        seed = options_d["seed"];
        LOGGER << "Using random seed: " << seed << std::endl;
    }

    double VG;
    double VR;
    bool flag_est_GE = true;
    if(options.find("G") != options.end()){
        VG = std::stod(options["G"]);
        VR = std::stod(options["E"]);
        flag_est_GE = false;
    }

    vector<string> ids;
    pheno->get_pheno(ids, phenos);
    if(ids.size() != num_indi){
        LOGGER.e(0, "Did you forget to specify --pheno?");
    }

    // read covar
    vector<uint32_t> remain_index, remain_index_covar;
    bool has_covar = false;
    Covar covar;
    if(covar.getCommonSampleIndex(ids, remain_index, remain_index_covar)){
        has_covar = true;
        LOGGER.i(0, to_string(remain_index.size()) + " overlapping individuals with non-missing data to be included from the covariate file(s).");
        if (covar.hasEnvir()){ 
            has_envir = true;
        }else{
            has_envir = false;
        }
    }else{
        // have covariates, no overlapped
        if(covar.hasCovar()){
            LOGGER.e(0, "no overlapping individual with non-missing data to be included from the covariate file(s).");
        }else{
            // no covariates
            remain_index.resize(ids.size());
            std::iota(remain_index.begin(), remain_index.end(), 0);
        }
    }

    vector<string> remain_ids(remain_index.size());
    std::transform(remain_index.begin(), remain_index.end(), remain_ids.begin(), [&ids](size_t pos){return ids[pos];});

    // read fam
    string ffam_file = "";
    fam_flag = true;
    if(options.find("grmsparse_file") != options.end()){
        ffam_file = options["grmsparse_file"];
    }else{
        fam_flag = false;
    }

    // index in ids when merge to spare fam
    vector<uint32_t> remain_index_fam;
    SpMat fam;

    if(fam_flag){
        readFAM(ffam_file, fam, remain_ids, remain_index_fam);
    }else{
        remain_index_fam.resize(remain_ids.size());
        std::iota(remain_index_fam.begin(), remain_index_fam.end(), 0);
    }

    int n_remain_index_fam = remain_index_fam.size();

    //reorder phenotype, covar
    vector<double> remain_phenos(n_remain_index_fam);
    vector<string> remain_ids_fam(n_remain_index_fam);
    vector<uint32_t> total_remain_index(n_remain_index_fam);
    for(int i = 0; i != n_remain_index_fam; i++){
        uint32_t temp_index = remain_index[remain_index_fam[i]];
        remain_phenos[i] = phenos[temp_index];
        remain_ids_fam[i] = ids[temp_index];
        total_remain_index[i] = temp_index;
    }
    //fix pheno keep
    pheno->filter_keep_index(total_remain_index);
    //geno->init_keep();
    num_indi = pheno->count_keep();
    LOGGER.i(0, "After matching all the files, " + to_string(remain_phenos.size()) + " individuals to be included in the analysis.");


    // standerdize the phenotype, and condition the covar
    phenoVec = Map<VectorXd> (remain_phenos.data(), remain_phenos.size());
    rawPhenoVec = phenoVec;

    // condition the covar
    if(has_covar){
        vector<double> remain_covar;
        vector<uint32_t> remain_inds_index;
        covar.getCovarX(remain_ids_fam, remain_covar, remain_inds_index);

        // get rid of duplicates
        int num_remain = remain_phenos.size();
        int num_remain_col = remain_covar.size() / num_remain;
        vector<int> remain_cols;
        for(int i = 0; i < num_remain_col; i++){
            int baseIndex = i * num_remain;
            double temp = remain_covar[baseIndex]; 
            int num_ident = 0;
            for(int j = 0; j < num_remain; j++){
                if(std::abs(temp - remain_covar[j + baseIndex]) < 1e-8){
                    num_ident++;
                }
            }
            if(num_ident != num_remain){
                remain_cols.push_back(i);
            }
        }

        int num_remain_col2 = remain_cols.size();
        if(num_remain_col2 != num_remain_col){
            vector<double> remain_covar2;
            remain_covar2.reserve(num_remain_col2 * num_remain);
            LOGGER.w(0, "" + to_string(num_remain_col - num_remain_col2) + " duplicated covariates are removed.");
            for(int i = 0; i < num_remain_col2; i++){
                int index2 = remain_cols[i] * num_remain;
                for(int j = 0; j < num_remain; j++){
                    remain_covar2.push_back(remain_covar[index2 + j]);
                }
            }
            remain_covar = remain_covar2;
        }

        remain_covar.resize(remain_covar.size() + num_remain);
        std::fill(remain_covar.end() - num_remain, remain_covar.end(), 1.0);

        MatrixXd concovar = Map<Matrix<double, Dynamic, Dynamic, Eigen::ColMajor>>(remain_covar.data(), num_remain, remain_covar.size() / num_remain);

        /*
        std::ofstream covar_w(options["out"] + "_aln_covar.txt"), pheno_w(options["out"] + "_aln_phen.txt"), pheno_w2(options["out"] + "_adj_phen.txt");
        pheno_w << phenoVec << std::endl;
        */
        /*
        std::ofstream covar_w(options["out"] + "_aln_covar.txt");
        for(int i = 0; i < phenoVec.size(); i++){
            covar_w << phenoVec(i);
            for(int j = 0; j < concovar.cols(); j++){
                covar_w << "\t" << concovar(i, j);
            }
            covar_w << std::endl;
        }
        covar_w.close();
        */
        /*
        FILE * out = fopen((options["out"] + ".covar.bin").c_str(), "wb");
        fwrite(concovar.data(), sizeof(double), remain_covar.size(), out);
        fclose(out);

        FILE * pout = fopen((options["out"] + ".phenob").c_str(), "wb");
        fwrite(phenoVec.data(), sizeof(double), phenoVec.size(), pout);
        fclose(pout);
        */
        covarFlag = true;
        makeIH(concovar);
        if(!bBinary)conditionCovarReg(phenoVec);
        if(options.find("save_pheno") != options.end()){
            std::ofstream pheno_w((options["out"] + ".cphen").c_str());
            if(!pheno_w) LOGGER.e(0, "failed to write " + options["out"]+".cphen");
            for(int k = 0; k < remain_ids_fam.size(); k++){
                pheno_w << remain_ids_fam[k] << "\t" << phenoVec[k] << std::endl;
            }
            pheno_w.close();
        }
        /*
        FILE * p1out = fopen((options["out"] + ".phenoa1").c_str(), "wb");
        fwrite(phenoVec.data(), sizeof(double), phenoVec.size(), p1out);
        fclose(p1out);
        */


        /*
        pheno_w2 << phenoVec << std::endl;
        covar_w.close();
        pheno_w.close();
        pheno_w2.close();
        */
    }else{
        this->covar.resize(phenoVec.size(), 1);
        this->covar.setZero();
        this->covar = this->covar.array() + 1;
    }

    //scale the envir to mean 0 and variance 1
    if(has_envir){
        vector<double> remain_envir;
        vector<uint32_t> remain_inds_index_2;
        covar.getEnvirX(remain_ids_fam,remain_envir,remain_inds_index_2);
        envirVec = Map<VectorXd> (remain_envir.data(), remain_envir.size());
        VectorXd envirVec_centered = envirVec - VectorXd::Ones(envirVec.size()) * (envirVec.mean());
        double envirVec_sd= sqrt( envirVec_centered.dot(envirVec_centered)/(envirVec_centered.size()-1) );
        envirVec_scaled = envirVec_centered / envirVec_sd;
    }

   // joint covar
    if(options.find("adj_covar") != options.end() && covarFlag){
        bPreciseCovar = true;
        LOGGER.i(0, "Fitting covariates jointly in the association analysis.");
        covarFlag = true;
    }else if(bBinary){
        covarFlag = true;
    }else{
        covarFlag = false;
        // not usefull now but useful to REML
        this->covar.resize(phenoVec.size(), 1);
        this->covar.setZero();
        this->covar = this->covar.array() + 1;
    }

    if(bBinary){
        initBinary(fam);
        return;
        //goto saveRes;
    }

    // Center
    double phenoVec_mean = phenoVec.mean();
    phenoVec -= VectorXd::Ones(phenoVec.size()) * phenoVec_mean;

    double raw_mean = rawPhenoVec.mean();
    rawPhenoVec = rawPhenoVec.array() - raw_mean;

    double Vpheno = phenoVec.array().square().sum() / (phenoVec.size() - 1);
    if(Vpheno < 1e-5){
        LOGGER.e(0, "the Vp is below 1e-5. Please check: 1. Is there a scaling issue with the phenotype? 2. Can the covariates explain all the Vp (e.g., phenotype is included as a covariate by accident)? 3. If it is a binary trait, is the prevalence very low?");
    }


    //LOGGER << "DEBUG: samples size: " << geno->pheno->count_keep() << std::endl;
    /*
    FILE * p1out = fopen((options["out"] + ".phenoa2").c_str(), "wb");
    fwrite(phenoVec.data(), sizeof(double), phenoVec.size(), p1out);
    fclose(p1out);
    */

    std::map<string, string> mtdString;
    mtdString["REML"] = "fastGWA-REML (grid search)";
    mtdString["HE"] = "Haseman-Elston regression";

    if(fam_flag){

        if(options.find("inv_file") == options.end()){
            // no saved inv file
            vector<double> Aij;
            vector<double> Zij;

            if(flag_est_GE){

                double optimal_rho; //optimal hyperparameter
                
                if(has_envir){
                    
                    double grid_size = options_d["grid_size"];
                    
                    LOGGER.i(0, "Estimating the genetic variance (Vg) by grid search and " + mtdString[options["VgEstMethod"]] + "...");
                    LOGGER.ts("HE");
                    
                    LOGGER.i(0, "Hyperparameter for grid search ranges from 0 to 1." );
                    
                    SpMat envir_scaled_diag(envirVec_scaled.size(), envirVec_scaled.size());
                    envir_scaled_diag=envirVec_scaled.asDiagonal();
                    SpMat fam_temp;
                    
                    vector<double> rho(grid_size); // Hyperparameter for grid search
                    vector<double> VG_temp(grid_size);
                    bool fam_flag_temp;
                    vector<double> fam_flag_vec(grid_size);
                    vector<double> pvalue_temp(grid_size);
                    vector<double> logL_temp(grid_size);
                    
                    
                    int optimal_iter;
                    
                    
                    if(options["rel_only"] == "yes"){ // to check the validity of using this method in G+G*E 2df test
                        
                        
                        for (int iteration = 0; iteration < grid_size; iteration++) {
                            rho[iteration] = iteration / (grid_size - 1.0) ;
                            
                            std::cout << std::endl;
                            LOGGER.i(0, "When hyperparameter = " + to_string(rho[iteration]) + ",");
                            
                            fam_flag_temp = fam_flag;
                            fam_temp = rho[iteration] * fam + (1 - rho[iteration]) * envir_scaled_diag * fam * envir_scaled_diag;
                            
                            LOGGER.i(0, "Use related pairs only.");
                            for(int k = 0; k < fam_temp.outerSize(); ++k){
                                for(SpMat::InnerIterator it(fam_temp, k); it; ++it){
                                    if(it.row() < it.col()){
                                        Aij.push_back(it.value());
                                        Zij.push_back(phenoVec[it.row()] * phenoVec[it.col()]);
                                    }
                                }
                            }
                            
                            VG_temp[iteration] = HEreg(Zij, Aij, fam_flag_temp, pvalue_temp[iteration]);
                            fam_flag_vec[iteration] = fam_flag_temp;
                            
                            Aij.clear();
                            Zij.clear();
                        }
                        
                        std::cout << std::endl;
                        for (int iteration = 0; iteration < grid_size; iteration++){
                            LOGGER.i(0, "Hyperparameter = " + to_string(rho[iteration]) + ", " + "Vg = " + to_string(VG_temp[iteration]) + ", " + "P = " + to_string(pvalue_temp[iteration]) );
                        }
                        std::cout << std::endl;
                        
                        optimal_iter = std::min_element(pvalue_temp.begin(),pvalue_temp.end()) - pvalue_temp.begin();
                        optimal_rho = rho[optimal_iter];
                        LOGGER.i(0, "The hyperparameter with the smallest p value is the optimal hyperparameter.");
                        
                    }else{
                        
                        for (int iteration = 0; iteration < grid_size; iteration++) {
                            rho[iteration] = iteration / (grid_size - 1.0) ;
                            
                            std::cout << std::endl;
                            LOGGER.i(0, "When hyperparameter = " + to_string(rho[iteration]) + ",");
                            
                            fam_flag_temp = fam_flag;
                            fam_temp = rho[iteration] * fam + (1 - rho[iteration]) * envir_scaled_diag * fam * envir_scaled_diag;
                            
                            string vgEstMethod = options["VgEstMethod"];
                            if(vgEstMethod == "HE"){
                                VG_temp[iteration] = HEreg(fam_temp, phenoVec, fam_flag_temp, pvalue_temp[iteration]);
                                fam_flag_vec[iteration] = fam_flag_temp;
                                
                            }else if(vgEstMethod == "MCREML"){
                                //VG_temp[iteration] = MCREML(fam_temp, phenoVec, fam_flag_temp);
                                //fam_flag_vec[iteration] = fam_flag_temp;
                                //for now, the MCREML will not be adopted for estimating variance components in the G+G*E 2df test
                                LOGGER.e(0, "Unknown method to estimate the Vg");
                            }else if (vgEstMethod == "REML"){
                                VG_temp[iteration] = spREML_2df(fam_temp, phenoVec, fam_flag_temp, rho[iteration], logL_temp[iteration]);
                                fam_flag_vec[iteration] = fam_flag_temp;
                            }else{
                                LOGGER.e(0, "Unknown method to estimate the Vg");
                            }
                        }
                        
                        string vgEstMethod = options["VgEstMethod"];
                        if(vgEstMethod == "HE"){
                            std::cout << std::endl;
                            for (int iteration = 0; iteration < grid_size; iteration++){
                                LOGGER.i(0, "Hyperparameter = " + to_string(rho[iteration]) + ", " + "Vg = " + to_string(VG_temp[iteration]) + ", " + "P = " + to_string(pvalue_temp[iteration]) );
                            }
                            std::cout << std::endl;
                            
                            optimal_iter = std::min_element(pvalue_temp.begin(),pvalue_temp.end()) - pvalue_temp.begin();
                            optimal_rho = rho[optimal_iter];
                            LOGGER.i(0, "The hyperparameter with the smallest p value is the optimal hyperparameter.");
                        }else if (vgEstMethod == "REML"){
                            std::cout << std::endl;
                            for (int iteration = 0; iteration < grid_size; iteration++){
                                LOGGER.i(0, "Hyperparameter = " + to_string(rho[iteration]) + ", " + "Vg = " + to_string(VG_temp[iteration]) + ", " + "logL = " + to_string(logL_temp[iteration]) );
                            }
                            std::cout << std::endl;
                            
                            optimal_iter = std::max_element(logL_temp.begin(),logL_temp.end()) - logL_temp.begin();
                            optimal_rho = rho[optimal_iter];
                            LOGGER.i(0, "The hyperparameter with the largest log likelihood is the optimal hyperparameter.");
                        }
                    }
                    
                    //update fam_flag , fam , VG
                    LOGGER.i(0, "The optimal hyperparameter is " + to_string(optimal_rho));
                    fam_flag = fam_flag_vec[optimal_iter];
                    fam = optimal_rho * fam + (1 - optimal_rho) * envir_scaled_diag * fam * envir_scaled_diag;
                    VG = VG_temp[optimal_iter];
                    LOGGER.i(0, "The optimal Vg is " + to_string(VG));
                    
                    std::ofstream o_optimal_rho;
                    if(options.find("optimal-rho") != options.end()){ 
                        o_optimal_rho.open(options["out"] + ".optimalrho");  
                        o_optimal_rho << optimal_rho << std::endl;
                        o_optimal_rho.close();
                    }
                }else{
                    LOGGER.i(0, "Estimating the genetic variance (Vg) by " + mtdString[options["VgEstMethod"]] + "...");
                    LOGGER.ts("HE");
                    if(options["rel_only"] == "yes"){
                        LOGGER.i(0, "Using related pairs only.");
                        for(int k = 0; k < fam.outerSize(); ++k){
                            for(SpMat::InnerIterator it(fam, k); it; ++it){
                                if(it.row() < it.col()){
                                    Aij.push_back(it.value());
                                    Zij.push_back(phenoVec[it.row()] * phenoVec[it.col()]);
                                }
                            }
                        }

                        VG = HEreg(Zij, Aij, fam_flag);
                    }else{
                        string vgEstMethod = options["VgEstMethod"];
                        if(vgEstMethod == "HE"){
                            VG = HEreg(fam, phenoVec, fam_flag);
                        }else if(vgEstMethod == "MCREML"){
                            VG = MCREML(fam, phenoVec, fam_flag);
                        }else if (vgEstMethod == "REML"){
                            VG = spREML(fam, phenoVec, fam_flag);
                        }else{
                            LOGGER.e(0, "Unknown method to estimate the Vg");
                        }
                    }
                }
                if(options.find("force_gwa") != options.end()){
                    if(!fam_flag){
                        LOGGER.w(0, " Forcing the program to run fastGWA MLM");
                    }
                    fam_flag = true;
                }

                if(fam_flag){
                    VR = Vpheno - VG;
                    if(VG < 0 && options.find("force_gwa") == options.end()){
                        LOGGER.w(0, "Constraining Vg to 0.");
                        fam_flag = false;
                    }else if(VG > Vpheno && fam_flag){
                        if(options.find("no_constrain") != options.end()){
                            LOGGER.w(0, "Vg is larger than Vp");
                        }else{
                            VG = 0.99 * Vpheno;
                            LOGGER.w(0, "Constraining Vg to 0.99 * Vp: " + to_string(VG) + ".");
                        }
                    }
                }
                if(options["VgEstMethod"]=="REML"){
                    LOGGER << "fastGWA-REML runtime: ";
                }else if(options["VgEstMethod"]=="HE"){
                    LOGGER << "HE regression runtime: ";
                }
                LOGGER << LOGGER.tp("HE") << " sec." << std::endl;

            }
            if(!fam_flag){
                LOGGER.w(0, "the estimate of Vg is not statistically significant (i.e., p > 0.05). "
                        "This is likely because the number of closely related individuals in the sample is not large enough. "
                        "\nIn this case, the program will use linear regression for association test.");
 
                fam.resize(0, 0);
                V_inverse.resize(0, 0);
                if(options.find("save_inv") != options.end()){
                    std::ofstream inv_id((options["out"]+".grm.id").c_str());
                    if(!inv_id) LOGGER.e(0, "failed to write to " + options["out"]+".grm.id");
                    inv_id << "--fastGWA" << std::endl;
                    inv_id.close();
                }
                goto saveRes;
            }

            if(!bGrammar){
                // ordinary inverse way.
                inverseFAM(fam, VG, VR);
                if(options.find("save_inv") != options.end()){
                    LOGGER.i(0, "Saving the inverse of V matrix for further analysis, to be used by --load-inv for further analysis");
                    std::ofstream inv_id((options["out"]+".grm.id").c_str());
                    if(!inv_id) LOGGER.e(0, "failed to write to " + options["out"]+".grm.id");
                    for(int k = 0; k < remain_ids_fam.size(); k++){
                        inv_id << remain_ids_fam[k] << std::endl;
                    }

                    FILE * inv_out = fopen((options["out"]+".grm.inv").c_str(), "wb");
                    InvItem item;
                    for(int k = 0; k < V_inverse.outerSize(); ++k){
                        for(SpMat::InnerIterator it(V_inverse, k); it; ++it){
                            item.row = it.row();
                            item.col = it.col();
                            item.val = it.value();
                            if(fwrite(&item, sizeof(item), 1, inv_out) != 1){
                                LOGGER.e(0, "failed to write to [" + options["out"] + ".grm.inv]");
                            }
                        }
                    }
                    fclose(inv_out);

                    LOGGER.i(0, "The inverse of V matrix has been saved to [" + options["out"] + ".grm.inv]");
                }
            }else{
                if(has_envir){
                    SpMat eye(fam.rows(), fam.cols());
                    eye.setIdentity();
                    fam *= VG;
                    fam += eye * VR;
                    LOGGER.ts("tuning");
                    solver.compute(fam);
                    if(solver.info() != Eigen::Success){
                        LOGGER.e(0, "the V matrix is not invertible.");
                    }
                    Vi_y = solver.solve(phenoVec);
                }else{
                    //grammar approx.
                    //LOGGER.i(0, "Performing GRAMMAR-Gamma approximation...");
                    grammar(fam, VG, VR);
                }
           }
        }else{
            // use saved inv 
            string id_file = options["inv_file"] + ".grm.id";
            std::ifstream h_id(id_file.c_str());
            if(!h_id){
                LOGGER.e(0, "can't read file [" + id_file + "].");
            }
            string line;
            uint32_t cur_index = 0;
            getline(h_id, line);
            if(line == "--fastGWA") {
                fam_flag = false;
                LOGGER.w(0, "The estimate of Vg is not statistically significant (i.e., p > 0.05). "
                        "This is likely because the number of relatives is not large enough. "
                        "\nPerforming simple regression via removing --grm-sparse instead...");
 
                fam.resize(0, 0);
                V_inverse.resize(0, 0);
                return;
            }
            // go to the original position
            h_id.clear(); 
            h_id.seekg(0);
            while(getline(h_id, line)){
                if(line != remain_ids_fam[cur_index]){
                    LOGGER.e(0, "sample IDs are not same from line " + to_string(cur_index + 1) + " in [" + id_file + "].");
                }
                cur_index++;
            }
            h_id.close();
            if(cur_index == remain_ids_fam.size()){
                LOGGER.i(0, to_string(cur_index) + " samples are confirmed to be identical in V inverse [" + id_file + "].");
            }else{
                LOGGER.e(0, "the file is empty or lines are not consistent in inverse V [" + id_file + "].");
            }

            V_inverse.resize(phenoVec.size(), phenoVec.size());
            string in_name = options["inv_file"] + ".grm.inv";
            LOGGER.i(0, "Loading the inverse of V martrix from " + in_name + "...");
            LOGGER.ts("LOAD_INV");
            FILE * in_file = fopen(in_name.c_str(), "rb");
            if(!in_file){
                LOGGER.e(0, "can't open the file.");
            }
            fseek(in_file, 0L, SEEK_END);
            size_t file_size = ftell(in_file);
            rewind(in_file);

            InvItem item;
            size_t cur_pos = 0;
            while(cur_pos < file_size){
                if(fread(&item, sizeof(item), 1, in_file) == 1){
                    V_inverse.insert(item.row, item.col) = item.val;
                    cur_pos += sizeof(item);
                }else{
                    LOGGER.e(0, "can't read file in pos: " + to_string(cur_pos));
                }
            }
            fclose(in_file);

            V_inverse.finalize();
            V_inverse.makeCompressed();
            LOGGER.i(0, "Inverse of V matrix loaded in " + to_string(LOGGER.tp("LOAD_INV")) + " seconds.");
        }
    }

    VR_copy = VR; 

saveRes:
    if(options.find("model_only") != options.end()){
        LOGGER << "Saving fastGWA model information..." << std::endl;
        string id_file = options["out"] + ".mdl.id";
        std::ofstream inv_id(id_file.c_str());
        if(!inv_id) LOGGER.e(0, "failed to write " + options["out"]+".grm.id");
        vector<string> ids = pheno->get_id(0, num_indi - 1, "\t");
        for(auto & t_id : ids){
            inv_id << t_id << "\n";
        }
        inv_id.close();
        LOGGER << "Sample information has been saved to [" << id_file << "]." << std::endl;

        MDLHeader header;
        header.magic[0] = 'f';
        header.magic[1] = 'G';
        header.magic[2] = 'W';
        header.magic[3] = 'A';
        header.magic[4] = '\0';
        header.num_indi = num_indi;
        header.fam_flag = fam_flag;
        header.isGrammar = bGrammar;
        header.phenoVec_start = 512;
        header.covarFlag = covarFlag;
        header.covarVec_start = header.phenoVec_start + header.num_indi * 8;
        header.covarVec_cols = this->covar.cols();
        uint64_t covar_size = header.covarVec_cols * header.num_indi * 8;
        header.has_envir = has_envir;
        header.envirVec_start = header.covarVec_start + covar_size + (header.covarFlag ? covar_size : 0);
        header.V_inverse_start = header.envirVec_start + (header.has_envir ? header.num_indi * 8 : 0);
        if(header.fam_flag & (!header.isGrammar)) header.V_inverse_items = V_inverse.nonZeros();
        header.c_inf = c_inf;
        header.VR_copy = VR_copy;

        string bin_file = options["out"] + ".mdl.bin";
        FILE* obin = fopen(bin_file.c_str(), "wb");
        if(!obin){
            LOGGER.e(0, "can't open " + bin_file + " to write.");
        }

        if(fwrite(&header, sizeof(header), 1, obin) != 1){
            LOGGER.e(0, "can't write headers to " + bin_file + ".");
        }

        fseek(obin, header.phenoVec_start, SEEK_SET);
        if (header.has_envir){
            if(fwrite(Vi_y.data(), sizeof(double), header.num_indi, obin) != header.num_indi){
                LOGGER.e(0, "can't write phenotype data to " + bin_file + ".");
            }
        }else if(fam_flag && header.isGrammar){
            if(fwrite(Vi_y_cinf.data(), sizeof(double), header.num_indi, obin) != header.num_indi){
                LOGGER.e(0, "can't write phenotype data to " + bin_file + ".");
            }
        }else{
            if(fwrite(this->phenoVec.data(), sizeof(double), header.num_indi, obin) != header.num_indi){
                LOGGER.e(0, "can't write phenotype data to " + bin_file + ".");
            }
        }

        uint64_t num_covar_elements = (uint64_t)header.num_indi * header.covarVec_cols;
        fseek(obin, header.covarVec_start, SEEK_SET);
        if(fwrite(this->covar.data(), sizeof(double), num_covar_elements, obin)
                != num_covar_elements){
            LOGGER.e(0, "can't write covariates to " + bin_file + ".");
        }

        if(header.covarFlag){
            if(fwrite(this->H.data(), sizeof(double), num_covar_elements, obin)
                    != num_covar_elements){
                LOGGER.e(0, "can't write covariates to " + bin_file + ".");
            }
        }

        fseek(obin, header.envirVec_start, SEEK_SET);
        if(header.has_envir){
            if(fwrite(envirVec_scaled.data(), sizeof(double), header.num_indi, obin) != header.num_indi){
                LOGGER.e(0, "can't write phenotype data to " + bin_file + ".");
            }
        }

        fseek(obin, header.V_inverse_start, SEEK_SET);
        if(header.fam_flag && (!header.isGrammar)){
            InvItem item;
            for(int k = 0; k < V_inverse.outerSize(); ++k){
                for(SpMat::InnerIterator it(V_inverse, k); it; ++it){
                    item.row = it.row();
                    item.col = it.col();
                    item.val = it.value();
                    if(fwrite(&item, sizeof(item), 1, obin) != 1){
                        LOGGER.e(0, "can't write V inverse to " + bin_file + ".");
                    }
                }
            }
        }
        fclose(obin);
        LOGGER << "Model has been saved to [" << bin_file << "]." << std::endl;
    }

 }
/*
void FastFAM::initMarkerVars(){
    num_marker = marker->count_extract();
    if(beta)delete[] beta;
    if(se) delete[] se;
    if(p) delete[] p;
    beta = new float[num_marker];
    se = new float[num_marker];
    p = new double[num_marker];
}
*/

FastFAM::~FastFAM(){
    delete pheno;
    delete marker;
    delete geno;
}

void FastFAM::makeIH(MatrixXd &X){
    MatrixXd ctc = X.transpose() * X;
    Eigen::FullPivLU<MatrixXd> lu(ctc);
    if(lu.rank() < ctc.rows()){
        LOGGER.w(0, "Highly correlated columns are found in the covariate matrix. The results are still reliable in most cases.");
    }
    covar = X;
    
    H = lu.solve(X.transpose()); 
}


void FastFAM::conditionCovarReg(VectorXd &y, VectorXd &condPheno){
    if(covarFlag){
        condPheno.noalias() = H * y;
    }else{
        condPheno = y;
    }
}

void FastFAM::conditionCovarBinReg(Eigen::Ref<VectorXd> y){
    double Hy_stack[64];
    vector<double> Hy_heap;
    double *Hy = Hy_stack;
    if (num_covar > 64) {
        Hy_heap.resize(num_covar);
        Hy = Hy_heap.data();
    }
    const char nT = 'N';
    const double a1 = 1.0;
    const double a2 = -1.0;
    const double b1 = 0;
    const double b2 = 1.0;
    const int incr = 1;
#if GCTA_CPU_x86
    dgemv(&nT, &num_covar, &numi_indi, &a1, H.data(), &num_covar, y.data(), &incr, &b1, Hy, &incr);
    dgemv(&nT, &numi_indi, &num_covar, &a2, covar.data(), &numi_indi, Hy, &incr, &b2, y.data(), &incr);
#else 
    dgemv_(&nT, &num_covar, &numi_indi, &a1, H.data(), &num_covar, y.data(), &incr, &b1, Hy, &incr);
    dgemv_(&nT, &numi_indi, &num_covar, &a2, covar.data(), &numi_indi, Hy, &incr, &b2, y.data(), &incr);
#endif
}

void FastFAM::conditionCovarReg(Eigen::Ref<VectorXd> y){
 
   if(covarFlag){
        y = y - covar * (H * y);
    }
 }

void generateRandom(Ref<MatrixXd> mat){
    uint64_t row = mat.rows();
    uint64_t col = mat.cols();
    
    boost::mt19937 rng( std::chrono::system_clock::now().time_since_epoch().count() + 77 );
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
        randn(rng, boost::normal_distribution<>(0.0, 1.0));
    for(int i = 0; i < col; i++){
        for(int j = 0; j < row; j++){
            mat(j, i) = randn();
        }
    }
}

void FastFAM::genRandY(uint64_t *buf, int num_marker){
    MatrixXd X(num_indi, num_marker);
    double * ptr = X.data();
    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < num_marker; i++){
        geno->makeMarkerX(buf, i, ptr + i * num_indi, true, true);
    }

    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < mcTrails; i++){
        rand_y.col(i) += X * rand_beta.block(finished_rand_marker, i, num_marker, 1);
    }

    finished_rand_marker += num_marker;
}

void FastFAM::estBeta(uint64_t *buf, int num_marker){
    MatrixXd X(num_indi, num_marker);
    double * ptr = X.data();
    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < num_marker; i++){
        geno->makeMarkerX(buf, i, ptr + i * num_indi, true, true);
    }

    phenoEstBeta.block(finished_rand_marker, 0, num_marker, 1) = X.transpose() * VinvY;

    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < mcTrails; i++){
        randEstBeta.block(finished_rand_marker, i, num_marker, 1) = X.transpose() * randVinvY.col(i);
    }

    finished_rand_marker += num_marker;
    
}
//MCREML
double FastFAM::fitREML(double logdet, const SpMat &fam, const VectorXd &pheno){
    SpMat eye(fam.rows(), fam.cols());
    eye.setIdentity();

    uint32_t n = fam.cols();
    double det = exp(logdet);
    double sqrt_det = sqrt(det);

    //solve the V
    SpMat V = fam + det * eye;
    Eigen::SimplicialLDLT<SpMat> solver;
    solver.compute(V);
    if(solver.info() != Eigen::Success){
        LOGGER.e(0, "the V matrix is not invertible.");
    }
    
    vector<uint32_t> marker_index = marker->get_extract_index();
    uint32_t m = marker_index.size();

    finished_rand_marker = 0;
    vector<function<void (uint64_t *, int)>> callBacks;
    if(!geno->filterMAF()){
        callBacks.push_back(std::bind(&Geno::freq64, geno, _1, _2));
    }
    callBacks.push_back(std::bind(&FastFAM::genRandY, this, _1, _2));
    geno->loop_64block(marker_index, callBacks, true);
    //geno->resetLoop();

    VectorXd sum_e2_rand(mcTrails); // sum e rand ^ 2
    double det2 = det * det;
    #pragma omp parallel for
    for(int i = 0; i < mcTrails; i++){
        rand_y.col(i) = rand_y.col(i) + sqrt_det * rand_e.col(i);
        randVinvY.col(i) = solver.solve(rand_y.col(i));
        sum_e2_rand(i) = det2 * (randVinvY.col(i).squaredNorm());
    }

    LOGGER << "solve VinvY" << std::endl;
    VinvY = solver.solve(pheno);
    double sum_e2 = det2 * (VinvY.squaredNorm()); 

    finished_rand_marker = 0;
    vector<function<void (uint64_t *, int)>> callBacks2;
    if(!geno->filterMAF()){
        callBacks2.push_back(std::bind(&Geno::freq64, geno, _1, _2));
    }
    callBacks2.push_back(std::bind(&FastFAM::estBeta, this, _1, _2));
    geno->loop_64block(marker_index, callBacks2, true);
    //geno->resetLoop();

    LOGGER << "sum of betas" << std::endl;

    VectorXd sum_beta2_rand(mcTrails);
    double invm2 = 1.0 / m / m;
    #pragma omp parallel for
    for(int i = 0; i < mcTrails; i++){
        sum_beta2_rand(i) = invm2 * randEstBeta.col(i).squaredNorm();
    }

    //sum beta 2
    LOGGER << "Get the sum of betas" << std::endl;
    double sum_beta2 = invm2 * phenoEstBeta.squaredNorm();

    double f = log( (sum_beta2 / sum_e2) / (sum_beta2_rand.sum() / sum_e2_rand.sum()));
    return f;
}


double FastFAM::MCREML(const Ref<const SpMat> fam, const Ref<const VectorXd> pheno, bool &isSig){
    uint64_t n = pheno.size();
    uint64_t m = marker->get_extract_index().size();
    //get random vectors
    mcTrails = 4e9 / n;
    mcTrails = mcTrails > 15 ? 15 : mcTrails;
    mcTrails = mcTrails < 3 ? 3 : mcTrails;

    rand_y.resize(n, mcTrails);
    rand_y.setZero();

    rand_beta.resize(m, mcTrails);
    rand_e.resize(n, mcTrails); 
    generateRandom(rand_beta);
    rand_beta = rand_beta.array() * sqrt(1.0/m);

    generateRandom(rand_e);

    randVinvY.resize(n, mcTrails);
    VinvY.resize(n);

    phenoEstBeta.resize(m, mcTrails);
    randEstBeta.resize(m, mcTrails);

    double hsq1 = 0.25;
    double logdet1 = log((1-hsq1)/hsq1);
    LOGGER << "Fitting MCREML, logdet: " << logdet1 << " h2: " << hsq1 << std::endl;
    double f1 =  fitREML(logdet1, fam, pheno);

    double hsq2;
    if(f1 < 0){
        hsq2 = 0.125;
    }else{
        hsq2 = 0.50;
    }
    double logdet2 = log((1-hsq2)/hsq2);
    LOGGER << "Fitting MCREML, logdet: " << logdet2 << " h2: " << hsq2 << std::endl;
    double f2 = fitREML(logdet2, fam, pheno);

    double prev2_logdet = logdet1, prev_logdet = logdet2;
    double prev2_f = f1, prev_f = f2;
    double f, logdet;
    for(int s = 0; s < 5; s++){
        logdet = (prev2_logdet * prev_f - prev_logdet * prev2_f) / (prev_f - prev2_f);
        if(std::abs(logdet - prev_logdet) < 0.01){
            break;
        }
        LOGGER << "Fitting MCREML, logdet: " << logdet << std::endl;
        f = fitREML(logdet, fam, pheno);
        prev2_logdet = prev_logdet;
        prev_logdet = logdet;
        prev2_f = prev_f;
        prev_f = f;
    }

    double h2 = pheno.dot(VinvY) / n;
    LOGGER << "  Vg = " << h2 << std::endl;
    isSig = true;
    return h2;
}

double FastFAM::HEreg(const Ref<const SpMat> fam, const Ref<const VectorXd> pheno, bool &isSig){
    double p_val;
    return HEreg(fam, pheno, isSig, p_val);
}

double FastFAM::HEreg(const Ref<const SpMat> fam, const Ref<const VectorXd> pheno, bool &isSig, double &pvalue){
    int num_covar = 1;
    int num_component = 1;
    int col_X = num_covar + num_component;
    MatrixXd XtX = MatrixXd::Zero(col_X, col_X);
    VectorXd XtY = VectorXd::Zero(col_X);
    double SSy = 0;

    uint64_t size = fam.cols() * fam.rows();
    XtX(0, 0) = size;

    int n = fam.cols();
    VectorXd SSys(n);
    VectorXd XtY0s(n);
    VectorXd XtY1s(n);
    VectorXd XtX01s(n);
    VectorXd XtX11s(n);
    #pragma omp parallel for
    for(int i = 1; i < n; i++){
        double temp_pheno = pheno[i];
        auto fam_block = fam.block(0, i, i, 1);
        auto pheno_block = pheno.head(i) * temp_pheno;
        SSys[i] = pheno_block.dot(pheno_block);
        XtY0s[i] = pheno_block.sum();
        XtY1s[i] = (pheno_block.transpose() * fam_block)[0];
        XtX01s[i] = fam_block.sum();
        XtX11s[i] = (fam_block.cwiseProduct(fam_block)).sum();
    }
    SSy = SSys.sum();
    XtY[0] = XtY0s.sum();
    XtY[1] = XtY1s.sum();
    XtX(0, 1) = XtX01s.sum();
    XtX(1, 1) = XtX11s.sum();

    //MatrixXd XtXi = XtX.selfadjointView<Eigen::Upper>().inverse();
    XtX(1,0) = XtX(0,1);
    LOGGER << "XtX:" << endl;
    LOGGER << XtX << endl;

    Eigen::FullPivLU<MatrixXd> lu(XtX);
    if(lu.rank() < XtX.rows()){
        LOGGER.w(0, "the XtX matrix is not invertible.");
        isSig = false;
        return std::numeric_limits<double>::quiet_NaN();
    }

    MatrixXd XtXi = lu.inverse();

    VectorXd betas = XtXi * XtY;
    LOGGER << "beta:" << endl;
    LOGGER << betas.transpose() << endl;

    double sse = (SSy - betas.dot(XtY)) / (size - col_X);
    LOGGER << "SSE: " << sse << endl;

    VectorXd SDs = sse * XtXi.diagonal();
    LOGGER << "SD: " << SDs.transpose() << endl;

    double hsq = betas[betas.size() - 1];
    double SD = SDs[SDs.size() - 1];
    double Zsq = hsq * hsq / SD;
    double p = StatLib::pchisqd1(Zsq);

    pvalue = p + 0.0;

    double Vpheno = pheno.array().square().sum() / (pheno.size() - 1);
    //LOGGER.i(2, "Vg = " + to_string(hsq) + ", se = " + to_string(sqrt(SD)) +  ", P = " + to_string(p));
    LOGGER << "\nSource\tVariance\tSE" << std::endl;
    LOGGER << "Vg\t" << hsq << "\t" << sqrt(SD) << std::endl;
    LOGGER << "Ve\t" << Vpheno - hsq << std::endl;
    LOGGER << "Vp\t" << Vpheno << std::endl;
    
    LOGGER << "\nHeritability = " << hsq / Vpheno << " (Pval = " << p << ")" << std::endl;

    if(p > 0.05){
        isSig = false;
    }else{
        isSig = true;
    }
    return hsq;
}

void FastFAM::logLREML(const Ref<const VectorXd> pheno, vector<double> &varcomp, double &logL, double *Hinv){
    int n_comp = varcomp.size();
    int n = A[0].cols();

    SpMat V(n, n);
    for(int j = 0; j < n_comp; j++){
        V += varcomp[j] * A[j];
    }
    //LOGGER << "Non zeros V: " << V.nonZeros() << std::endl;

    Eigen::SimplicialLDLT<SpMat> solverV;
    solverV.compute(V);
    if(solverV.info() != Eigen::Success){
        logL = -std::numeric_limits<double>::infinity();
        return;
    }

    VectorXd d = solverV.vectorD();
    if(d.minCoeff() <= 0.0){
        logL = -std::numeric_limits<double>::infinity();
        return;
    }
    double logdet_V = d.array().log().sum();

    MatrixXd ViX = solverV.solve(covar); // n*c

    MatrixXd XtViX = covar.transpose() * ViX; // c*c
    INVmethod method = INV_FQR;
    double logdet_XtViX;
    int rank;
    if(!SquareMatrixInverse(XtViX, logdet_XtViX, rank, method)){
        // Grid-search drivers (spREML, spREML_2df) tolerate -inf by skipping
        // this grid point. See binLogL for the PSD discussion.
        logL = -std::numeric_limits<double>::infinity();
        return;
    }

    MatrixXd b_proj_t = ViX * XtViX; //n*c
    VectorXd b2 = b_proj_t.transpose() * pheno; // c*1
    VectorXd resi = pheno - covar * b2; // n*1
    VectorXd Py = solverV.solve(resi);

    logL = -0.5 * (logdet_V + logdet_XtViX + pheno.dot(Py));

    if(Hinv){
        MatrixXd APy(n, n_comp);
        for(int i = 0; i < n_comp; i++){
            APy.col(i) = A[i] * Py;
        }

        MatrixXd Hi(n_comp, n_comp);
        for(int i = 0; i < n_comp; i++){
            VectorXd cur_APy = APy.col(i);
            VectorXd cvec = solverV.solve(cur_APy) - ViX * (b_proj_t.transpose() * cur_APy);
            Hi(i, i) = 0.5 * cur_APy.dot(cvec);
            for(int j = 0; j < i; j++){
                Hi(j, i) = 0.5 * APy.col(j).dot(cvec);
                Hi(i, j) = Hi(j, i);
            }
        }
        INVmethod mtd_hi = INV_FQR;
        double logdet_hi;
        int rank;
        if(!SquareMatrixInverse(Hi, logdet_hi, rank, mtd_hi)){
            LOGGER.e(0, "the Hi matrix is not invertible.");
        }
        memcpy(Hinv, Hi.data(), n_comp * n_comp * sizeof(double));
    }
}


double FastFAM::spREML(const Ref<const SpMat> fam, const Ref<const VectorXd> pheno, bool &isSig){
    int n_comp = 2;

    int n = pheno.size();
    int n_covar = covar.cols();

    A.resize(2);
    A[0] = fam;
    A[1].resize(n, n); 
    A[1].setIdentity();

    double Vp = pheno.array().square().sum() / (n - 1);

    //vector<int> trails = {101, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};
    vector<int> trails = {101, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};
    double MAX_Vg = Vp * options_d["h2_limit"];
    double start = 0;
    double end = MAX_Vg;
    double end_logL = 0;
    vector<bool> bEndNAN(trails.size());
    vector<double> starts(trails.size());
    vector<double> ends(trails.size());
    vector<double> steps(trails.size());
    int total_infeasible = 0;
    int total_points = 0;

    bool b_reml = false;
    std::ofstream out;
    if(options.find("reml_detail") != options.end()){
        b_reml = true;
        out.open(options["out"] + ".reml");
        if(!out){
            LOGGER.e(0, "failed to open the " + options["out"] + ".reml");
        }

        out << std::setprecision( std::numeric_limits<double>::digits10);
        out << "index\tlogL\tVg\tVe" << std::endl;

    }

    for(int iter = 0; iter < trails.size(); iter++){
        int n_trails = trails[iter];
        double step = (end - start) / (n_trails - 1);
        vector<double> varcomps(n_trails);
        vector<double> logLs(n_trails);

        #pragma omp parallel for
        for(int i = 0; i < n_trails; i++){
            vector<double> varcomp(n_comp);
            varcomp[0] = start + i * step;
            varcomp[1] = Vp - varcomp[0];
            double logL;
            logLREML(pheno, varcomp, logL);
            varcomps[i] = varcomp[0];
            logLs[i] = logL;
        }

        double max_logL = -1e300;
        double max_varcomp = 0;

        int max_index = 0;
        int n_infeasible = 0;
        for(int i = 0; i < n_trails; i++){
            double logL = logLs[i];
            if(!std::isfinite(logL)){
                n_infeasible++;
                continue;
            }
            if(logL > max_logL){
                max_logL = logL;
                max_varcomp = varcomps[i];
                max_index = i;
            }
        }
        total_infeasible += n_infeasible;
        total_points += n_trails;

        if(n_infeasible == n_trails){
            LOGGER.e(0, "all " + std::to_string(n_trails) + " Vg values in iteration "
                + std::to_string(iter + 1) + " gave a non-PD V or a singular XtV^-1X over ["
                + std::to_string(start) + ", " + std::to_string(end) + "]. "
                "Check for collinear covariates, or try a denser or PSD-regularised GRM.");
        }
        if(max_index == 0){
            start = varcomps[max_index];
            end = varcomps[max_index + 1];
            end_logL = logLs[max_index + 1];
        }else if(max_index == n_trails - 1){
            start = varcomps[max_index - 1];
            end = varcomps[max_index];
            end_logL = logLs[max_index];
        }else{
            start = varcomps[max_index - 1];
            end = varcomps[max_index + 1];
            end_logL = logLs[max_index + 1];
        }

        if(b_reml){
            for(int i = 0; i < logLs.size(); i++){
                out << i << "\t" << logLs[i] << "\t" << varcomps[i] << "\t" << Vp - varcomps[i] << std::endl;
            }
            out << "----------------------------" << std::endl;
        }

        if(!std::isfinite(end_logL)){
            bEndNAN[iter] = true;
        }else{
            bEndNAN[iter] = false;
        }
        starts[iter] = start;
        ends[iter] = end;
        steps[iter] = step;


        LOGGER << "Iteration " << iter + 1 << ", step size: " << step << ", logL: " << max_logL
               << ". Vg: " << max_varcomp
               << ", searching range: " << start << " to " << end;
        if(n_infeasible > 0){
            LOGGER << ", infeasible: " << n_infeasible << "/" << n_trails;
        }
        LOGGER << std::endl;
    }

    if(b_reml){
        out.close();
    }

    if(total_infeasible > 0){
        LOGGER << "Grid search skipped " << total_infeasible << "/" << total_points
               << " Vg values because V was not positive definite or XtV^-1X was singular."
               << " If the converged Vg lies close to the infeasible region, consider a denser"
               << " or PSD-regularised GRM." << std::endl;
    }

    double Vg = (start + end) / 2.0;
    double Ve = Vp - Vg;

    if(b_reml){
        LOGGER << "\n" << "Upper boundary detail: " << std::endl;
        LOGGER << "  " << "iter\tstep\tstart\tend\tendNAN" << std::endl;
        LOGGER << std::boolalpha;
        for(int iter = 0; iter < trails.size(); iter++){
            LOGGER << "  " << iter << "\t" << steps[iter] << "\t" << starts[iter] << "\t" << ends[iter] << "\t" << bEndNAN[iter] << std::endl; 
        }
        LOGGER << std::endl;
    }

    if(bEndNAN[0] && Ve < 0){
        std::streamsize ss = std::cout.precision();
        LOGGER << "Best guess Vg " 
            << std::setprecision( std::numeric_limits<double>::digits10) 
            << "range: " << start << " to " << end 
            << ", Vp: " << Vp << std::setprecision(ss) << std::endl;
        LOGGER.e(0, "fastGWA-REML can't converge.");
    }

    if(MAX_Vg - end <= 1e-10){
        std::streamsize ss = std::cout.precision();
        LOGGER << "Best guess Vg " 
            << std::setprecision( std::numeric_limits<double>::digits10) 
            << "range: " << start << " to " << end 
            << ", Vp: " << Vp << std::setprecision(ss) << std::endl;

        LOGGER.e(0, "fastGWA-REML can't converge – hit the upper limit!");
    }

    LOGGER << "fastGWA-REML converged." << std::endl;

    //std::streamsize ss = std::cout.precision();
    //LOGGER << "Estimated Vg: " << std::setprecision( std::numeric_limits<double>::digits10) << Vg << ", range: " << start << " to " << end << std::setprecision(ss) << std::endl;

    double logL;
    MatrixXd Hinv(n_comp, n_comp);
    vector<double> varcomp(n_comp);
    varcomp[0] = Vg;
    varcomp[1] = Ve;
    logLREML(pheno, varcomp, logL, Hinv.data());

    LOGGER << "logL: " << logL << std::endl;

    LOGGER <<"Sampling variance/covariance of the estimates of Vg and Ve:" << std::endl; 
    LOGGER << Hinv << std::endl;

    LOGGER << "\nSource\tVariance\tSE" << std::endl;
    LOGGER << "Vg" << "\t" << varcomp[0] << "\t" << sqrt(Hinv(0, 0)) << std::endl;
    LOGGER << "Ve" << "\t" << varcomp[1] << "\t" << sqrt(Hinv(1, 1)) << std::endl;
    LOGGER << "Vp" << "\t" << Vp << std::endl;

    for(int i = 0; i < n_comp; i++){
        A[i].resize(0, 0);
    }
    A.resize(0);

    double zsq = Vg * Vg / Hinv(0, 0);
    double p = StatLib::pchisqd1(zsq);

    LOGGER << "\nHeritability = " << Vg / Vp << " (Pval = " << p << ")" << std::endl;
    
    if(p>0.05){
        isSig = false;
    }else{
        isSig = true;
    }

    return Vg;
}

double FastFAM::spREML_2df(const Ref<const SpMat> fam, const Ref<const VectorXd> pheno, bool &isSig, const double rho, double &logLikelihood){
    int n_comp = 2;
    
    int n = pheno.size();
    int n_covar = covar.cols();
    
    A.resize(2);
    A[0] = fam;
    A[1].resize(n, n);
    A[1].setIdentity();
    
    double Vp = pheno.array().square().sum() / (n - 1);
    
    //vector<int> trails = {101, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};
    vector<int> trails = {101, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};
    double MAX_Vg = Vp * options_d["h2_limit"];
    double start = 0;
    double end = MAX_Vg;
    double end_logL = 0;
    vector<bool> bEndNAN(trails.size());
    vector<double> starts(trails.size());
    vector<double> ends(trails.size());
    vector<double> steps(trails.size());
    int total_infeasible = 0;
    int total_points = 0;
    
    bool b_reml = false;
    std::ofstream out;
    if(options.find("reml_detail") != options.end()){
        b_reml = true;
        out.open(options["out"] + ".reml");
        if(!out){
            LOGGER.e(0, "Error to open the " + options["out"] + ".reml");
        }
        
        out << std::setprecision( std::numeric_limits<double>::digits10);
        out << "index\tlogL\tVg\tVe" << std::endl;
        
    }
    
    for(int iter = 0; iter < trails.size(); iter++){
        int n_trails = trails[iter];
        double step = (end - start) / (n_trails - 1);
        vector<double> varcomps(n_trails);
        vector<double> logLs(n_trails);
        
#pragma omp parallel for
        for(int i = 0; i < n_trails; i++){
            vector<double> varcomp(n_comp);
            varcomp[0] = start + i * step;
            varcomp[1] = Vp - varcomp[0];
            double logL;
            logLREML(pheno, varcomp, logL);
            varcomps[i] = varcomp[0];
            logLs[i] = logL;
        }
        
        double max_logL = -1e300;
        double max_varcomp = 0;
        
        int max_index = 0;
        int n_infeasible = 0;
        for(int i = 0; i < n_trails; i++){
            double logL = logLs[i];
            if(!std::isfinite(logL)){
                n_infeasible++;
                continue;
            }
            if(logL > max_logL){
                max_logL = logL;
                max_varcomp = varcomps[i];
                max_index = i;
            }
        }
        total_infeasible += n_infeasible;
        total_points += n_trails;

        if(n_infeasible == n_trails){
            LOGGER.e(0, "all " + std::to_string(n_trails) + " Vg values in iteration "
                + std::to_string(iter + 1) + " gave a non-PD V or a singular XtV^-1X over ["
                + std::to_string(start) + ", " + std::to_string(end) + "]. "
                "Check for collinear covariates, or try a denser or PSD-regularised GRM.");
        }
        if(max_index == 0){
            start = varcomps[max_index];
            end = varcomps[max_index + 1];
            end_logL = logLs[max_index + 1];
        }else if(max_index == n_trails - 1){
            start = varcomps[max_index - 1];
            end = varcomps[max_index];
            end_logL = logLs[max_index];
        }else{
            start = varcomps[max_index - 1];
            end = varcomps[max_index + 1];
            end_logL = logLs[max_index + 1];
        }
        
        if(b_reml){
            for(int i = 0; i < logLs.size(); i++){
                out << i << "\t" << logLs[i] << "\t" << varcomps[i] << "\t" << Vp - varcomps[i] << std::endl; 
            }
            out << "----------------------------" << std::endl;
        }
        
        if(!std::isfinite(end_logL)){
            bEndNAN[iter] = true;
        }else{
            bEndNAN[iter] = false;
        }
        starts[iter] = start;
        ends[iter] = end;
        steps[iter] = step;
        
        
        LOGGER << "Iteration " << iter + 1 << ", step size: " << step << ", logL: " << max_logL
        << ". Vg: " << max_varcomp
        << ", searching range: " << start << " to " << end;
        if(n_infeasible > 0){
            LOGGER << ", infeasible: " << n_infeasible << "/" << n_trails;
        }
        LOGGER << std::endl;
    }
    
    if(b_reml){
        out.close();
    }
    
    if(total_infeasible > 0){
        LOGGER << "Grid search skipped " << total_infeasible << "/" << total_points
               << " Vg values because V was not positive definite or XtV^-1X was singular."
               << " If the converged Vg lies close to the infeasible region, consider a denser"
               << " or PSD-regularised GRM." << std::endl;
    }
    
    double Vg = (start + end) / 2.0;
    double Ve = Vp - Vg;
    
    if(b_reml){
        LOGGER << "\n" << "Up boundary detail: " << std::endl;
        LOGGER << "  " << "iter\tstep\tstart\tend\tendNAN" << std::endl;
        LOGGER << std::boolalpha;
        for(int iter = 0; iter < trails.size(); iter++){
            LOGGER << "  " << iter << "\t" << steps[iter] << "\t" << starts[iter] << "\t" << ends[iter] << "\t" << bEndNAN[iter] << std::endl;
        }
        LOGGER << std::endl;
    }
    
    if(bEndNAN[0] && Ve < 0){
        std::streamsize ss = std::cout.precision();
        LOGGER << "Best guess Vg "
        << std::setprecision( std::numeric_limits<double>::digits10)
        << "range: " << start << " to " << end
        << ", Vp: " << Vp << std::setprecision(ss) << std::endl;
        LOGGER.e(0, "fastGWA-REML can't converge.");
    }
    
    if(MAX_Vg - end <= 1e-10){
        std::streamsize ss = std::cout.precision();
        LOGGER << "Best guess Vg "
        << std::setprecision( std::numeric_limits<double>::digits10)
        << "range: " << start << " to " << end
        << ", Vp: " << Vp << std::setprecision(ss) << std::endl;
        
        LOGGER.e(0, "fastGWA-REML can't converge, hit upper limit!");
    }
    
    LOGGER << "fastGWA-REML converged." << std::endl;
    
    //std::streamsize ss = std::cout.precision();
    //LOGGER << "Estimated Vg: " << std::setprecision( std::numeric_limits<double>::digits10) << Vg << ", range: " << start << " to " << end << std::setprecision(ss) << std::endl;
    
    double logL;
    MatrixXd Hinv(n_comp, n_comp);
    vector<double> varcomp(n_comp);
    varcomp[0] = Vg;
    varcomp[1] = Ve;
    logLREML(pheno, varcomp, logL, Hinv.data());
    
    LOGGER << "logL: " << logL << std::endl;
    
    logLikelihood = logL ;

    
    LOGGER <<"Sampling variance/covariance of the estimates of Vg and Ve:" << std::endl;
    LOGGER << Hinv << std::endl;
    
    LOGGER << "\nSource\tVariance\tSE" << std::endl;
    LOGGER << "Vg" << "\t" << varcomp[0] << "\t" << sqrt(Hinv(0, 0)) << std::endl;
    LOGGER << "Ve" << "\t" << varcomp[1] << "\t" << sqrt(Hinv(1, 1)) << std::endl;
    LOGGER << "Vp" << "\t" << Vp << std::endl;
    
    for(int i = 0; i < n_comp; i++){
        A[i].resize(0, 0);
    }
    A.resize(0);
    
    double zsq = Vg * Vg / Hinv(0, 0);
    double p = StatLib::pchisqd1(zsq);
    
    LOGGER << "\nHeritability = " << Vg / Vp << " (Pval = " << p << ")" << std::endl;
    
    if(p>0.05){
        isSig = false;
    }else{
        isSig = true;
    }
    
    return Vg;
}

double FastFAM::HEreg(vector<double> &Zij, vector<double> &Aij, bool &isSig){
    double p_val;
    return HEreg(Zij, Aij, isSig, p_val);
}

double FastFAM::HEreg(vector<double> &Zij, vector<double> &Aij, bool &isSig, double &pvalue){
    Map<VectorXd> ZVec(Zij.data(), Zij.size());
    Map<VectorXd> AVec(Aij.data(), Aij.size());

    double Zmean = ZVec.mean();
    double Amean = AVec.mean();
    ZVec -= (VectorXd::Ones(ZVec.size())) * Zmean;
    AVec -= (VectorXd::Ones(AVec.size())) * Amean;

    double A2v = (AVec.transpose() * AVec)(0, 0);
    if(A2v < 1e-6){
        LOGGER.e(0, "can't solve the regression");
    }
    double AZ = (AVec.transpose() * ZVec)(0, 0);
    double hsq = (1.0 / A2v) * AZ;

    VectorXd RZVec = ZVec - AVec * hsq;

    double delta = RZVec.array().square().sum() / (RZVec.size() - 2);
    double se = sqrt(delta / A2v);

    double z = hsq / se;

    double p = StatLib::pchisqd1(z * z);

    pvalue = p + 0.0;

    LOGGER.i(2, "Vg = " + to_string(hsq) + ", se = " + to_string(se) +  ", P = " + to_string(p));

    if(p > 0.05){
        isSig = false;
    }else{
        isSig = true;
    }

    return hsq;
}
    

void FastFAM::readFAM(string filename, SpMat& fam, const vector<string> &ids, vector<uint32_t> &remain_index){
    LOGGER.i(0, "Reading the sparse GRM file from [" + filename + "]...");
    vector<string> sublist = Pheno::read_sublist(filename + ".grm.id");
    vector<uint32_t> fam_index;
    vector_commonIndex(sublist, ids, fam_index, remain_index);

    // Fix index order to outside, preserving phenotype and covariate sample order
    vector<size_t> index_list_order = sort_indexes(remain_index);
    const size_t n_keep = remain_index.size();
    vector<uint32_t> ordered_fam_index(n_keep, 0);
    vector<uint32_t> ordered_remain_index(n_keep, 0);
    std::transform(index_list_order.begin(), index_list_order.end(), ordered_fam_index.begin(), [&fam_index](size_t pos){
            return fam_index[pos];});
    std::transform(index_list_order.begin(), index_list_order.end(), ordered_remain_index.begin(), [&remain_index](size_t pos){
            return remain_index[pos];});
    remain_index = ordered_remain_index;

    string sp_path = filename + ".grm.sp";
    FILE *fp = std::fopen(sp_path.c_str(), "rb");
    if(!fp){
        LOGGER.e(0, "can't read [" + sp_path + "]");
    }
    vector<char> io_buf(1 << 20);
    std::setvbuf(fp, io_buf.data(), _IOFBF, io_buf.size());

    uint32_t max_fam_id = static_cast<uint32_t>(sublist.size());
    for(size_t idx = 0; idx < n_keep; ++idx){
        if(ordered_fam_index[idx] >= max_fam_id){
            max_fam_id = ordered_fam_index[idx] + 1;
        }
    }
    vector<int32_t> map_index(max_fam_id, -1);
    for(size_t idx = 0; idx < n_keep; ++idx){
        map_index[ordered_fam_index[idx]] = static_cast<int32_t>(idx);
    }

    vector<uint32_t> id1;
    vector<uint32_t> id2;
    vector<double> grm;
    id1.reserve(n_keep * 2);
    id2.reserve(n_keep * 2);
    grm.reserve(n_keep * 2);

    vector<int64_t> num_elements(n_keep, 0);

    char line_buf[4096];
    while(std::fgets(line_buf, sizeof(line_buf), fp)){
        const char *p = line_buf;
        while(*p == ' ' || *p == '\t') ++p;
        if(*p == '\0' || *p == '\n' || *p == '\r') continue;

        char *endp = NULL;
        unsigned long raw_id1 = std::strtoul(p, &endp, 10);
        if(endp == p) continue;
        p = endp;

        unsigned long raw_id2 = std::strtoul(p, &endp, 10);
        if(endp == p) continue;
        p = endp;

        if(raw_id1 < max_fam_id && raw_id2 < max_fam_id){
            int32_t mapped1 = map_index[raw_id1];
            int32_t mapped2 = map_index[raw_id2];
            if(mapped1 >= 0 && mapped2 >= 0){
                double tmp_grm = std::strtod(p, &endp);
                uint32_t u1 = static_cast<uint32_t>(mapped1);
                uint32_t u2 = static_cast<uint32_t>(mapped2);
                id1.push_back(u1);
                id2.push_back(u2);
                num_elements[u2] += 1;
                grm.push_back(tmp_grm);
                if(u1 != u2){
                    id1.push_back(u2);
                    id2.push_back(u1);
                    num_elements[u1] += 1;
                    grm.push_back(tmp_grm);
                }
            }
        }
    }
    std::fclose(fp);

    // Direct compressed column-major assembly via counting bucket placement + per-column sort
    fam.resize(n_keep, n_keep);
    const int64_t total_nnz = static_cast<int64_t>(id1.size());
    fam.resizeNonZeros(total_nnz);

    long long *outer_ptr = fam.outerIndexPtr();
    long long *inner_ptr = fam.innerIndexPtr();
    double *val_ptr = fam.valuePtr();

    vector<int64_t> write_pos(n_keep, 0);
    int64_t offset = 0;
    for(size_t col = 0; col < n_keep; ++col){
        outer_ptr[col] = offset;
        write_pos[col] = offset;
        offset += num_elements[col];
    }
    outer_ptr[n_keep] = offset;

    for(int64_t k = 0; k < total_nnz; ++k){
        uint32_t col = id2[k];
        int64_t pos = write_pos[col]++;
        inner_ptr[pos] = static_cast<long long>(id1[k]);
        val_ptr[pos] = grm[k];
    }

    vector<std::pair<long long, double>> col_entries;
    for(size_t col = 0; col < n_keep; ++col){
        int64_t c_start = outer_ptr[col];
        int64_t c_end = outer_ptr[col + 1];
        int64_t len = c_end - c_start;
        if(len <= 1) continue;
        if(len == 2){
            if(inner_ptr[c_start] > inner_ptr[c_start + 1]){
                std::swap(inner_ptr[c_start], inner_ptr[c_start + 1]);
                std::swap(val_ptr[c_start], val_ptr[c_start + 1]);
            }
            continue;
        }
        col_entries.resize(len);
        for(int64_t r = 0; r < len; ++r){
            col_entries[r] = std::make_pair(inner_ptr[c_start + r], val_ptr[c_start + r]);
        }
        std::stable_sort(col_entries.begin(), col_entries.end(),
                         [](const std::pair<long long, double> &a, const std::pair<long long, double> &b){
                             return a.first < b.first;
                         });
        for(int64_t r = 0; r < len; ++r){
            inner_ptr[c_start + r] = col_entries[r].first;
            val_ptr[c_start + r] = col_entries[r].second;
        }
    }
}

void FastFAM::grammar_func(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    int nMarker = markerIndex.size();
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < nMarker; i++){
        int index_cur_marker = num_grammar_markers + i;
        GenoBufItem item;
        item.extractedMarkerIndex = markerIndex[i];
        geno->getGenoDouble(genobuf, i, &item);
        bValids[index_cur_marker] = item.valid;
        if(item.valid){
            Map< VectorXd > curGeno(item.geno.data(), num_indi);
            conditionCovarReg(curGeno);
            VectorXd Vg = solver.solve(curGeno);
            double gt_Vg = curGeno.dot(Vg);
            double g_Vi_y = curGeno.dot(Vi_y);
            double temp_chisq = g_Vi_y * g_Vi_y / gt_Vg;

            v_chisq[index_cur_marker] = temp_chisq;

            if(temp_chisq < 5){
                double gt_g = curGeno.dot(curGeno);
                double tmp_cinf = gt_Vg / gt_g;
                v_c_infs[index_cur_marker] = tmp_cinf;
            }else{
                bValids[index_cur_marker] = false;
            }
        }
    }
    num_grammar_markers += nMarker;
}


void FastFAM::grammar(SpMat& fam, double VG, double VR){
    int num_marker_rand = 2000; //1000 -> 2000, longda
    int soft_cap = 1000; // a soft cap to stop the grammar-gamma approx, longda


    LOGGER.i(0, "\nTuning parameters using " + to_string(num_marker_rand) + " null SNPs...");

    SpMat eye(fam.rows(), fam.cols());
    eye.setIdentity();

    fam *= VG;
    fam += eye * VR;

    //LOGGER.i(0, "Estimating conjugate gradient...");
    LOGGER.ts("tuning");
    solver.compute(fam);
    if(solver.info() != Eigen::Success){
        LOGGER.e(0, "the V matrix is not invertible.");
    }
    //LOGGER << "TCG compute time: " << LOGGER.tp("TCG") << std::endl;

    //LOGGER.i(0, "Solving Vi * y via conjugate gradient...");
    //LOGGER.ts("vi_y");
    Vi_y = solver.solve(phenoVec);
    //LOGGER << "  time: " << LOGGER.tp("vi_y") << std::endl;


    // get 1000 random SNPs
    auto total_markers_index = marker->get_extract_index_autosome();
    if(total_markers_index.size() < num_marker_rand){
        LOGGER.e(0, "can't read " + to_string(num_marker_rand) + " SNPs from autosomes for tuning.");
    }
    std::default_random_engine generator;
    //generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    uint32_t seed;
    if(options_d["seed"] == 0){
        seed = pheno->getSeed();
    }else{
        seed = options_d["seed"];
        LOGGER << "Seed used: " << seed << std::endl;
    }
    //LOGGER << "Setting seed to: " << seed << std::endl;
    generator.seed(seed);
    std::uniform_int_distribution<uint32_t> distribution(0, total_markers_index.size() - 1);
    vector<uint32_t> seq_marker_index(num_marker_rand);

    for(int i = 0; i < num_marker_rand; i++){
        seq_marker_index[i] = distribution(generator);
    }
    std::sort(seq_marker_index.begin(), seq_marker_index.end());
    auto last = std::unique(seq_marker_index.begin(), seq_marker_index.end());
    seq_marker_index.erase(last, seq_marker_index.end());

    num_marker_rand = seq_marker_index.size();
    vector<uint32_t> marker_index(num_marker_rand);
    std::transform(seq_marker_index.begin(), seq_marker_index.end(), marker_index.begin(), [&total_markers_index](size_t pos){return total_markers_index[pos];});

    int nMarker = 100;

    //get previous threshold
    double preAF = geno->getMAF();
    double preInfo = geno->getFilterInfo();
    double preMiss = geno->getFilterMiss();
    if(options.find("c-inf-no-filter") != options.end()){
        if(preAF < 0.01){
            geno->setMAF(0.01);
        }
        if(preInfo < 0.3 && hasInfo){
            geno->setFilterInfo(0.3);
        }
        if(preMiss < 0.9){
            geno->setFilterMiss(0.9);
        }
    }

    v_chisq.resize(num_marker_rand);
    v_c_infs.resize(num_marker_rand);
    bValids.resize(num_marker_rand);
    num_grammar_markers = 0;

    LOGGER << "  reading genotypes..." << std::endl; 
    vector<function<void (uintptr_t *, const vector<uint32_t> &)>> callBacks;
    callBacks.push_back(bind(&FastFAM::grammar_func, this, _1, _2));
    geno->loopDouble(marker_index, nMarker, true, true, false, false, callBacks);

    if(num_marker_rand != num_grammar_markers){
        LOGGER.e(0, "some SNPs cannot be read successfully!");
    }

    //reset back
    geno->setMAF(preAF);
    geno->setFilterInfo(preInfo);
    geno->setFilterMiss(preMiss);

    
    double tmp_cinf = 0;
    int n_valid_null = 0;

    std::ofstream o_inf;
    bool out_inf = false;
    if(options.find("c-inf") != options.end()){
        o_inf.open(options["out"] + ".cinf");
        o_inf << "CHR\tSNP\tPOS\tA1\tA2\tcinf\tchisq" << std::endl;
        out_inf = true;
    }
    for(int i=0; i<num_marker_rand; i++){
        if(bValids[i]){
            tmp_cinf += v_c_infs[i];
            n_valid_null++;
            if(out_inf){
                o_inf << marker->getMarkerStrExtract(marker_index[i]) << "\t" << v_c_infs[i] << "\t" << v_chisq[i] << std::endl;
            }
           // added by longda.
            if(i >= soft_cap && n_valid_null >= nMarker){
                LOGGER << "Tuning of Gamma finished at the " << i << "th SNP." << std::endl;
                break;
            }
        }
    }
    if(out_inf){
        o_inf.close();
    }
    //LOGGER.i(0, "Got " + to_string(n_valid_null) + " null SNPs");

    if(n_valid_null < 100){
        LOGGER.e(0, "not enough valid null SNPs (<100). \nYou may check if too variants are removed by a filter, e.g., MAF.");
    }

    c_inf = tmp_cinf / n_valid_null;
    LOGGER.i(0, "Mean GRAMMAR-Gamma value = " + to_string(c_inf));
    LOGGER << "Tuning of Gamma finished " << LOGGER.tp("tuning") << " seconds." << std::endl;
    Vi_y_cinf = Vi_y.array() / c_inf;
    v_chisq.resize(0);
    v_c_infs.resize(0);
    bValids.resize(0);

    if(Vi_y_cinf.size() != num_indi){
        LOGGER.e(0, "inconsistent sample size - there may be some unknown bugs.");
    }
    if(options.find("save_resi") != options.end()){
        std::ofstream oresi((options["out"] + ".fastGWA.residual").c_str());
        std::ofstream ogam((options["out"] + ".fastGWA.gamma").c_str());
        if((!ogam) || (!oresi)){
            LOGGER.e(0, "can't write fastGWA-MLM parameters to the file.");
        }
        ogam << c_inf << std::endl;
        ogam.close();

        for(int i = 0; i < num_indi; i++){
            oresi << (pheno->get_id(i,i, "\t")[0]) << "\t" << Vi_y_cinf[i] << "\n";
        }
        oresi.close();
        LOGGER << "Saved GRAMMAR-Gamma residual to [" << options["out"] << ".fastGWA(.residual, .gamma)" << "]." << std::endl;
    }
}
 

void FastFAM::inverseFAM(SpMat& fam, double VG, double VR){
    LOGGER.i(0, "\nInverting the variance-covariance matrix by " + options["inv_method"] + " (This may take a long time)...");
    //LOGGER.i(0, "DEUBG: Inverse Threads " + to_string(Eigen::nbThreads()));
    LOGGER.ts("INVERSE_FAM");
    SpMat eye(fam.rows(), fam.cols());
    //LOGGER.i(0, "FAM " + to_string(fam.rows()) + " * " + to_string(fam.cols()));
    eye.setIdentity();

    // V

    fam *= VG;
    fam += eye * VR;

    // inverse
    if(options["inv_method"] == "ldlt"){
        Eigen::SimplicialLDLT<SpMat> solver;
        solver.compute(fam);

        if(solver.info() != Eigen::Success){
            LOGGER.e(0, "the sparse GRM is not invertible.");
        }

        V_inverse = solver.solve(eye); 
    }else{
        LOGGER.e(0, "unknown matrix inverse method.");
    }


    LOGGER.i(0, "The V matrix inverted in " + to_string(LOGGER.tp("INVERSE_FAM")) + " sec.");
}

void FastFAM::calculate_gwa(uintptr_t * genobuf, const vector<uint32_t> &markerIndex){

    static double iN = 1.0 /(num_indi - (covarFlag ? covar.cols() : 1.0) - 1.0);
    static double SSy = phenoVec.dot(phenoVec);

    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;

        geno->getGenoDouble(genobuf, i, &item);

        isValids[i] = item.valid;
        if(!item.valid) {
            continue;
        }

        Map< VectorXd > xMat(item.geno.data(), num_indi);

        conditionCovarReg(xMat);


        double xMat_V_x = 1.0 / xMat.dot(xMat);
        double xMat_V_p = xMat.dot(phenoVec);

        double temp_beta =  xMat_V_x * xMat_V_p;
        double sse = (SSy - temp_beta * xMat_V_p) * iN;
        double temp_se = sqrt(sse * xMat_V_x);


        double temp_z = temp_beta / temp_se;

        beta[i] = (float)temp_beta; //* geno->RDev[cur_raw_marker]; 
        se[i] = (float)temp_se;
        p[i] = StatLib::pchisqd1(temp_z * temp_z); 

        af[i] = (float)item.af;
        countMarkers[i] = item.nValidN;
        info[i] = item.info; 
        //}else{
            //beta[index] = nan;
            //se[index] = nan;
            //p[index] = nan;
        //}
    }

    output_res(isValids, markerIndex);
   /*
        std::ofstream o_geno(options["out"] + "_geno.txt");
        o_geno << dealGeno << std::endl;
        o_geno.close();
    */
}

void FastFAM::calculate_gwa_2df(uintptr_t * genobuf, const vector<uint32_t> &markerIndex){
    static double iN = 1.0 /(num_indi - (covarFlag ? covar.cols() : 1.0) - 2.0);
    static double SSy = phenoVec.dot(phenoVec);

    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;

        geno->getGenoDouble(genobuf, i, &item);

        isValids[i] = item.valid;
        if(!item.valid) {
            continue;
        }

        Map< VectorXd > xMat(item.geno.data(), num_indi);

        conditionCovarReg(xMat);
        VectorXd g_times_envir(num_indi); 
        g_times_envir = xMat.cwiseProduct(envirVec_scaled);
        //center g_times_envir
        g_times_envir -=VectorXd::Ones(g_times_envir.size()) * (g_times_envir.mean()) ; 
        
        
        double x_x = xMat.dot(xMat);
        double xenvir_x = g_times_envir.dot(xMat);
        double xenvir_xenvir = g_times_envir.dot(g_times_envir);
        
        double x_p = phenoVec.dot(xMat);
        double xenvir_p = phenoVec.dot(g_times_envir);
        
        VectorXd first_deri(2);
        // here the first_deri is not the first derivative of likelihood function. first_deri times variance of residual will be the first derivative 
        first_deri(0)=x_p;    
        first_deri(1)=xenvir_p; 
        
        // here the ngtv_2nd_deri is not the negative second derivative of likelihood function. ngtv_2nd_deri times variance of residual will be the negative second derivative 
        MatrixXd ngtv_2nd_deri(2,2);
        ngtv_2nd_deri(0,0)=x_x;
        ngtv_2nd_deri(0,1)=ngtv_2nd_deri(1,0)=xenvir_x;
        ngtv_2nd_deri(1,1)=xenvir_xenvir;
        
        MatrixXd ngtv_2nd_deri_inv=ngtv_2nd_deri.inverse();
        
        VectorXd temp_beta = ngtv_2nd_deri_inv * first_deri;
        
        double sse = (SSy - temp_beta[0] * x_p - temp_beta[1] * xenvir_p ) * iN;
        
        VectorXd temp_se(2); 
        temp_se(0) = sqrt( sse * ngtv_2nd_deri_inv(0,0) );
        temp_se(1) = sqrt( sse * ngtv_2nd_deri_inv(1,1) );
        
        
        double temp_chisq = first_deri.dot( ngtv_2nd_deri_inv * first_deri ) / sse;
        double temp_chisq_x = pow(temp_beta[0]/temp_se[0] , 2);
        double temp_chisq_xenvir = pow(temp_beta[1]/temp_se[1] , 2);
        
        beta_geno[i]=(float)temp_beta(0);
        se_geno[i]=(float)temp_se(0);
        score_geno[i]=(float)temp_chisq_x;
        
        if (temp_chisq_x > 0){
          p_geno[i] = StatLib::pchisqd1(temp_chisq_x);
        }else{
          p_geno[i] = 1;
        }
        
        beta_interaction[i]=(float)temp_beta[1];
        se_interaction[i]=(float)temp_se[1];
        score_interaction[i]=(float)temp_chisq_xenvir;
        
        if (temp_chisq_xenvir > 0){
          p_interaction[i] = StatLib::pchisqd1(temp_chisq_xenvir);
        }else{
          p_interaction[i] = 1;
        }
        
        cov_geno_interaction[i] = (float)(sse * ngtv_2nd_deri_inv(0,1)) ;
        
        score[i] = (float)temp_chisq;
        if (temp_chisq > 0){
            p[i] = StatLib::pchisqd2(temp_chisq);
        }else{
            p[i] = 1;
        }
        
        af[i] = (float)item.af;
        countMarkers[i] = item.nValidN;
        info[i] = item.info; 

    }
  
    output_res_2df(isValids, markerIndex);
}

void FastFAM::calculate_gwa_2df_sandwich(uintptr_t * genobuf, const vector<uint32_t> &markerIndex){

    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;

        geno->getGenoDouble(genobuf, i, &item);

        isValids[i] = item.valid;
        if(!item.valid) {
            continue;
        }

        Map< VectorXd > xMat(item.geno.data(), num_indi);
        conditionCovarReg(xMat);
        VectorXd g_times_envir(num_indi);
        g_times_envir = xMat.cwiseProduct(envirVec_scaled);
        //center g_times_envir
        g_times_envir -=VectorXd::Ones(g_times_envir.size()) * (g_times_envir.mean());
        
        double x_x = xMat.dot(xMat);
        double xenvir_x = g_times_envir.dot(xMat);
        double xenvir_xenvir = g_times_envir.dot(g_times_envir);
        
        double x_p = phenoVec.dot(xMat);
        double xenvir_p = phenoVec.dot(g_times_envir);
        
        VectorXd first_deri(2);
        // here the first_deri is not the first derivative of likelihood function. first_deri times variance of residual will be the first derivative
        first_deri(0)=x_p;
        first_deri(1)=xenvir_p;
        
        // here the ngtv_2nd_deri is not the negative second derivative of likelihood function. ngtv_2nd_deri times variance of residual will be the negative second derivative
        // -A matrix is ngtv_2nd_deri
        MatrixXd ngtv_2nd_deri(2,2);
        ngtv_2nd_deri(0,0)=x_x;
        ngtv_2nd_deri(0,1)=ngtv_2nd_deri(1,0)=xenvir_x;
        ngtv_2nd_deri(1,1)=xenvir_xenvir;
        
        // (-A)^(-1) is ngtv_2nd_deri_inv
        MatrixXd ngtv_2nd_deri_inv=ngtv_2nd_deri.inverse();
        
        MatrixXd z_mat(xMat.size(),2);
        z_mat.col(0)=xMat;
        z_mat.col(1)=g_times_envir;
        
        MatrixXd z_ngtv_2nd_deri_inv = z_mat * ngtv_2nd_deri_inv;
        MatrixXd hatDiagonal = (z_ngtv_2nd_deri_inv.cwiseProduct(z_mat)).rowwise().sum();
        
        VectorXd temp_beta = ngtv_2nd_deri_inv * first_deri;
        
        VectorXd residual = phenoVec - xMat * temp_beta[0] - g_times_envir * temp_beta[1];
        VectorXd residualSquare = residual.cwiseProduct(residual);
        VectorXd correctedResidualSquare = residualSquare.cwiseProduct( (VectorXd::Ones(hatDiagonal.size()) * (1.0) - hatDiagonal).cwiseInverse() );
        
        // B matrix is B_mat
        MatrixXd B_mat(2,2);
        VectorXd x_correctedResidualSquare = xMat.cwiseProduct(correctedResidualSquare);
        VectorXd xenvir_correctedResidualSquare = g_times_envir.cwiseProduct(correctedResidualSquare);
        B_mat(0,0) = x_correctedResidualSquare.dot(xMat);
        B_mat(0,1) = B_mat(1,0) = x_correctedResidualSquare.dot(g_times_envir);
        B_mat(1,1) = xenvir_correctedResidualSquare.dot(g_times_envir);
        
        //sandwich variance estimator is (-A)^(-1) * B * (-A)^(-1)
        MatrixXd sandwich_variance = ngtv_2nd_deri_inv * B_mat * ngtv_2nd_deri_inv;
        
        if (std::isnan(sandwich_variance.sum()) == 0){
            double temp_chisq = temp_beta.dot(sandwich_variance.inverse() * temp_beta);
            
            VectorXd temp_se(2);
            temp_se(0) = sqrt(sandwich_variance(0,0));
            temp_se(1) = sqrt(sandwich_variance(1,1));
            
            double temp_chisq_x = pow(temp_beta[0]/temp_se[0] , 2);
            double temp_chisq_xenvir = pow(temp_beta[1]/temp_se[1] , 2);
            
            beta_geno[i]=(float)temp_beta(0);
            se_geno[i]=(float)temp_se(0);
            score_geno[i]=(float)temp_chisq_x;
            
            if (temp_chisq_x > 0){
                p_geno[i] = StatLib::pchisqd1(temp_chisq_x);
            }else{
                p_geno[i] = 1;
            }
            
            beta_interaction[i]=(float)temp_beta[1];
            se_interaction[i]=(float)temp_se[1];
            score_interaction[i]=(float)temp_chisq_xenvir;
            
            if (temp_chisq_xenvir > 0){
                p_interaction[i] = StatLib::pchisqd1(temp_chisq_xenvir);
            }else{
                p_interaction[i] = 1;
            }
            
            cov_geno_interaction[i] = (float)sandwich_variance(0,1) ;
            
            score[i] = (float)temp_chisq;
            if (temp_chisq > 0){
                p[i] = StatLib::pchisqd2(temp_chisq);
            }else{
                p[i] = 1;
            }
            
        }

        af[i] = (float)item.af;
        countMarkers[i] = item.nValidN;
        info[i] = item.info; 
    }
    
    output_res_2df(isValids, markerIndex);
}

void FastFAM::output_res_spa(const vector<uint8_t> &isValids, const vector<uint32_t> markerIndex){
    int numKept = 0;
    int num_marker = markerIndex.size();
    if(bSaveBin){
        int num_write = 1;
        for(int i = 0; i != num_marker; i++){
            if(isValids[i]){
                numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\n";
                if(fwrite(&af[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write allele frequency to [" + sFileName + ".bin].");
                }

                if(fwrite(&beta[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write beta to [" + sFileName + ".bin].");
                }
                if(fwrite(&se[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write se to [" + sFileName + ".bin].");
                }
                if(fwrite(&p[i], sizeof(double), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write p to [" + sFileName + ".bin].");
                }
                if(fwrite(&padj[i], sizeof(double), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write Padj to [" + sFileName + ".bin].");
                }

                if(fwrite(&countMarkers[i], sizeof(uint32_t), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write N to [" + sFileName + ".bin].");
                }
                if(hasInfo){
                    if(fwrite(&info[i], sizeof(float), num_write, bOut) != num_write){
                        LOGGER.e(0, "can't write INFO score to [" + sFileName + ".bin].");
                    }
                }


            }
        }
    }else{
        for(int i = 0; i != num_marker; i++){
            if(isValids[i]){
                numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t" << countMarkers[i]
                    << "\t" << af[i] << "\t" << Tscore[i] << "\t" << Tse[i] << "\t" << p[i] 
                    << "\t" << beta[i] << "\t" << se[i] << "\t" << padj[i] << "\t" << ((int)rConverge[i]);
                if(hasInfo){
                    osOut << "\t" << info[i];
                }
                osOut << "\n";
            }else{
                if(bOutResAll){
                    numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t" << countMarkers[i]
                    << "\t" << af[i] << "\tNA\tNA\tNA\tNA\tNA\tNA\tNA";
                if(hasInfo){
                    osOut << "\t" << info[i];
                }
                osOut << "\n";
 
                }
            }
        }
    }

    numMarkerOutput += numKept;
}
 


void FastFAM::output_res(const vector<uint8_t> &isValids, const vector<uint32_t> markerIndex){
    int numKept = 0;
    int num_marker = markerIndex.size();
    if(bSaveBin){
        int num_write = 1;
        for(int i = 0; i != num_marker; i++){
            if(isValids[i]){
                numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\n";
                if(fwrite(&af[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write allele frequency to [" + sFileName + ".bin].");
                }

                if(fwrite(&beta[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write beta to [" + sFileName + ".bin].");
                }
                if(fwrite(&se[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write se to [" + sFileName + ".bin].");
                }
                if(fwrite(&p[i], sizeof(double), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write p to [" + sFileName + ".bin].");
                }

                if(fwrite(&countMarkers[i], sizeof(uint32_t), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write N to [" + sFileName + ".bin].");
                }
                if(hasInfo){
                    if(fwrite(&info[i], sizeof(float), num_write, bOut) != num_write){
                        LOGGER.e(0, "can't write INFO score to [" + sFileName + ".bin].");
                    }
                }


            }
        }
    }else{
        for(int i = 0; i != num_marker; i++){
            if(isValids[i]){
                numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t" << countMarkers[i]
                    << "\t" << af[i] << "\t" << beta[i] << "\t" << se[i] << "\t" << p[i];
                if(hasInfo){
                    osOut << "\t" << info[i];
                }
                osOut << "\n";
            }else{
                if(bOutResAll){
                    numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t" << countMarkers[i]
                    << "\t" << af[i] << "\tNA\tNA\tNA";
                if(hasInfo){
                    osOut << "\t" << info[i];
                }
                osOut << "\n";
 
                }
            }
        }
    }

    numMarkerOutput += numKept;
}


void FastFAM::output_res_2df(const vector<uint8_t> &isValids, const vector<uint32_t> markerIndex){
    int numKept = 0;
    int num_marker = markerIndex.size();
    if(bSaveBin){
        int num_write = 1;
        for(int i = 0; i != num_marker; i++){
            if(isValids[i]){
                numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\n";
                if(fwrite(&af[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write allele frequency to [" + sFileName + ".bin].");
                }

                if(fwrite(&beta_geno[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write beta_geno to [" + sFileName + ".bin].");
                }
                if(fwrite(&beta_interaction[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write beta_interaction to [" + sFileName + ".bin].");
                }
                if(fwrite(&se_geno[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write se_geno to [" + sFileName + ".bin].");
                }
                if(fwrite(&se_interaction[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write se_interaction to [" + sFileName + ".bin].");
                }
                if(fwrite(&cov_geno_interaction[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write cov_geno_interaction to [" + sFileName + ".bin].");
                }
                if(fwrite(&score_geno[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write score_geno to [" + sFileName + ".bin].");
                }
                if(fwrite(&score_interaction[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write score_interaction to [" + sFileName + ".bin].");
                }
                if(fwrite(&score[i], sizeof(float), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write score to [" + sFileName + ".bin].");
                }
                if(fwrite(&p_geno[i], sizeof(double), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write p_geno to [" + sFileName + ".bin].");
                }   
                if(fwrite(&p_interaction[i], sizeof(double), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write p_interaction to [" + sFileName + ".bin].");
                }                             
                if(fwrite(&p[i], sizeof(double), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write p to [" + sFileName + ".bin].");
                }

                if(fwrite(&countMarkers[i], sizeof(uint32_t), num_write, bOut) != num_write){
                    LOGGER.e(0, "can't write N to [" + sFileName + ".bin].");
                }
                if(hasInfo){
                    if(fwrite(&info[i], sizeof(float), num_write, bOut) != num_write){
                        LOGGER.e(0, "can't write INFO score to [" + sFileName + ".bin].");
                    }
                }


            }
        }
    }else{
        for(int i = 0; i != num_marker; i++){
            if(isValids[i]){
                numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t" << countMarkers[i]
                    << "\t" << af[i] << "\t" << beta_geno[i] << "\t" << beta_interaction[i]  << "\t" << se_geno[i] << "\t" << se_interaction[i] << "\t" << cov_geno_interaction[i] << "\t" << score_geno[i] << "\t" << score_interaction[i] << "\t" << score[i] << "\t" << p_geno[i] << "\t" << p_interaction[i] << "\t" << p[i];
                if(hasInfo){
                    osOut << "\t" << info[i];
                }
                osOut << "\n";
            }else{
                if(bOutResAll){
                    numKept++;
                osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t" << countMarkers[i]
                    << "\t" << af[i] << "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA";
                if(hasInfo){
                    osOut << "\t" << info[i];
                }
                osOut << "\n";
 
                }
            }
        }
    }

    numMarkerOutput += numKept;
}
 


void FastFAM::calculate_fam(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){

    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;

        geno->getGenoDouble(genobuf, i, &item);

        isValids[i] = item.valid;
        if(!item.valid) {
            continue;
        }
 
        Map< VectorXd > xMat(item.geno.data(), num_indi);

        conditionCovarReg(xMat);

        VectorXd xMat_V = V_inverse * xMat;
        double xMat_V_x = 1.0 / xMat_V.dot(xMat);
        double xMat_V_p = xMat_V.dot(phenoVec);

        double temp_beta =  xMat_V_x * xMat_V_p;
        double temp_se = sqrt(xMat_V_x);
        double temp_z = temp_beta / temp_se;


        beta[i] = (float)temp_beta; //* geno->RDev[cur_raw_marker]; 
        se[i] = (float)temp_se;
        p[i] = StatLib::pchisqd1(temp_z * temp_z); 

        af[i] = (float)item.af;
        countMarkers[i] = item.nValidN;
        info[i] = item.info;
    }
    output_res(isValids, markerIndex);
}

void FastFAM::calculate_grammar(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;
        geno->getGenoDouble(genobuf, i, &item);
        
        isValids[i] = item.valid;
        if(!item.valid){
            continue;
        }
        
        Map< VectorXd > xvec(item.geno.data(), num_indi);

        conditionCovarReg(xvec);

        double gtg = xvec.dot(xvec);
        //double gt_Vi_y = xvec.dot(Vi_y);
        double gt_Vi_y = xvec.dot(Vi_y_cinf);

        //double temp_beta = gt_Vi_y / gtg / c_inf;
        double temp_beta = gt_Vi_y / gtg;
        //double temp_chisq = temp_beta * gt_Vi_y;
        double temp_chisq = temp_beta * gt_Vi_y * c_inf;
        double temp_se = sqrt(temp_beta * temp_beta / temp_chisq);

        beta[i] = (float)temp_beta; //* geno->RDev[cur_raw_marker]; 
        se[i] = (float)temp_se;
        p[i] = StatLib::pchisqd1(temp_chisq); 
        af[i] = (float)item.af;
        countMarkers[i] = item.nValidN;
        info[i] = item.info;
    }

    output_res(isValids, markerIndex);
}


void FastFAM::calculate_mixed_2df(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    //calculate residualized phenotype
    phenoVec = VR_copy * Vi_y;
    //center residualized phenotype
    phenoVec -=VectorXd::Ones(phenoVec.size()) * (phenoVec.mean()) ;

    calculate_gwa_2df(genobuf, markerIndex);
}

void FastFAM::calculate_mixed_2df_sandwich(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    //calculate residualized phenotype
    phenoVec = VR_copy * Vi_y;
    //center residualized phenotype
    phenoVec -=VectorXd::Ones(phenoVec.size()) * (phenoVec.mean()) ;

    calculate_gwa_2df_sandwich(genobuf, markerIndex);
}

void FastFAM::processFAM(vector<function<void (uintptr_t *, const vector<uint32_t> &)>> callBacks){
    sFileName = options["out"]; 
    int buf_size = 23068672;
    osBuf.resize(buf_size);
    osOut.rdbuf()->pubsetbuf(&osBuf[0], buf_size);
    if(options.find("save_bin") == options.end()){
        bSaveBin = false;
        LOGGER << "fastGWA results will be saved in text format to [" << sFileName << "]." << std::endl;
        osOut.open(sFileName.c_str());
        vector<string> header = {"CHR", "SNP", "POS", "A1", "A2", "N", "AF1", "BETA", "SE", "P"};
        if(hasInfo)header.push_back("INFO");
        if(bBinary){
            header = {"CHR", "SNP", "POS", "A1", "A2", "N", "AF1", "T", "SE_T", "P_noSPA", "BETA", "SE",  "P",  "CONVERGE"};
            if(hasInfo)header.push_back("INFO");
        }
        if(has_envir){
            header = {"CHR", "SNP", "POS", "A1", "A2", "N", "AF1", "BETA_G", "BETA_G_by_E", "SE_G", "SE_G_by_E", "Cov_BETA_G_and_G_by_E", "chisq_G", "chisq_G_by_E", "chisq_2df", "P_G", "P_G_by_E", "P_2df"};
            if(hasInfo)header.push_back("INFO");
        }
        string header_string = boost::algorithm::join(header, "\t");
        if(osOut.bad()){
            LOGGER.e(0, "can't open [" + sFileName + "] to write.");
        }
        osOut << header_string << std::endl;
    }else{
        bSaveBin = true;
        LOGGER << "fastGWA results will be saved in binary format to [" << sFileName << "(.snpinfo, .bin)]" << std::endl;
        osOut.open((sFileName + ".snpinfo").c_str());
        if(osOut.bad()){
            LOGGER.e(0, "can't open [" + sFileName + ".snpinfo] to write.");
        }
        osOut << "CHR\tSNP\tPOS\tA1\tA2" << std::endl;
        bOut = fopen((sFileName + ".bin").c_str(), "wb");
        if(bOut == NULL){
            LOGGER.e(0, "can't open [" + sFileName + ".bin] to write.");
        }
    }


    if(options.find("no_filter") == options.end()){
        bOutResAll = false;
        double preAF = geno->getMAF();
        double preInfo = geno->getFilterInfo();
        double preMiss = geno->getFilterMiss();
        if(preAF < 1e-10){
            geno->setMAF(0.0001);
            LOGGER << "  Filtering out variants with MAF < 0.0001, or customise it with --maf flag." << std::endl; 
        }
        /*
        if(preInfo < 1e-10 && hasInfo){
            geno->setFilterInfo(0.3);
            LOGGER << "  Filtering out variants with imputation INFO score < 0.30, or customise it with --info flag." << std::endl;
        }
        */
        if(preMiss < 1e-10){
            geno->setFilterMiss(0.9);
            LOGGER << "  Filtering out variants with missingness rate > 0.10, or customise it with --geno flag." << std::endl;
        }
    }else{
        bOutResAll = true;
    }

    vector<uint32_t> extractIndex(marker->count_extract());
    std::iota(extractIndex.begin(), extractIndex.end(), 0);

    int nMarker = 1024;
 
    beta = new float[nMarker];
    se = new float[nMarker];
    p = new double[nMarker];
   
    countMarkers = new uint32_t[nMarker];
    af = new float[nMarker];
    info = new float[nMarker];

    numMarkerOutput = 0;

    bool bCenter = true;
    if(bBinary){
        Tscore = new float[nMarker];
        Tse = new float[nMarker];
        padj = new double[nMarker];
        rConverge = new uint8_t[nMarker];
    }
    if(has_envir){
        beta_geno = new float[nMarker];
        beta_interaction = new float[nMarker];
        se_geno = new float[nMarker];
        se_interaction = new float[nMarker];
        cov_geno_interaction = new float[nMarker];
        score_geno = new float[nMarker];
        score_interaction = new float[nMarker];
        score = new float[nMarker];
        p_geno = new double[nMarker];
        p_interaction = new double[nMarker];
    }
    geno->loopDouble(extractIndex, nMarker, true, bCenter, false, false, callBacks);

    osOut.flush();
    osOut.close();
    if(bOut){
        fflush(bOut);
        fclose(bOut);
    }
    LOGGER << "Saved " << numMarkerOutput << " SNPs." << std::endl;

    delete[] beta;
    delete[] se;
    delete[] p;
    delete[] countMarkers;
    delete[] af;
    delete[] info;
    if(bBinary){
        delete[] padj;
        delete[] rConverge;
        delete[] Tscore;
        delete[] Tse;
    }
    if(has_envir){
        delete[] beta_geno;
        delete[] beta_interaction;
        delete[] se_geno;
        delete[] se_interaction;
        delete[] cov_geno_interaction;
        delete[] score_geno;
        delete[] score_interaction;
        delete[] score;
        delete[] p_geno;
        delete[] p_interaction;
    }
}

int FastFAM::registerOption(map<string, vector<string>>& options_in){
    int returnValue = 0;
    //DEBUG: change to .fastFAM
    options["out"] = options_in["out"][0] + ".fastGWA";

    string curFlag = "--fastGWA";
    if(options_in.find(curFlag) != options_in.end()){
        processFunctions.push_back("fast_fam");
        returnValue++;
        options_in.erase(curFlag);
        LOGGER.e(0, "obsoleted flag. Try --fastGWA-mlm or --fastGWA-mlm-exact.");
    }

    curFlag = "--fastGWA-gram";
    if(options_in.find(curFlag) != options_in.end()){
        processFunctions.push_back("fast_fam");
        options["grammar"] = "true";
        returnValue++;
        options_in.erase(curFlag);
        LOGGER.e(0, "obsoleted flag. Try --fastGWA-mlm or --fastGWA-mlm-exact.");
    }

    curFlag = "--grm-sparse";
    if(options_in.find(curFlag) != options_in.end()){
        if(options_in[curFlag].size() == 1){
            options["grmsparse_file"] = options_in[curFlag][0];
        }else{
            LOGGER.e(0, curFlag + "can't deal with 0 or > 1 files.");
        }
        options_in.erase(curFlag);
    }

    curFlag = "--fastGWA-mlm";
    if(options_in.find(curFlag) != options_in.end()){
        processFunctions.push_back("fast_fam");
        options["grammar"] = "true";
        if(options.find("grmsparse_file") == options.end()){
            LOGGER.e(0, "--fastGWA-mlm only works with --grm-sparse.");
        }
        returnValue++;
        options_in.erase(curFlag);
    }

    curFlag = "--fastGWA-mlm-binary";
    if(options_in.find(curFlag) != options_in.end()){
        processFunctions.push_back("fast_fam");
        options["binary"] = "true";
        if(options.find("grmsparse_file") == options.end()){
            LOGGER.e(0, "--fastGWA-mlm-binary only works with --grm-sparse.");
        }
        returnValue++;
        options_in.erase(curFlag);
    }

    curFlag = "--save-fastGWA-mlm-residual";
    if(options_in.find(curFlag) != options_in.end()){
        if(options.find("grammar") == options.end()){
            LOGGER.e(0, curFlag + " can only work with --fastGWA-mlm.");
        }
        options["save_resi"] = "true";
        options_in.erase(curFlag);
    }

    curFlag = "--fastGWA-mlm-exact";
    if(options_in.find(curFlag) != options_in.end()){
        processFunctions.push_back("fast_fam");
        if(options.find("grmsparse_file") == options.end()){
            LOGGER.e(0, "--fastGWA-mlm-exact only works with --grm-sparse.");
        }
        returnValue++;
        options_in.erase(curFlag);
    }

    curFlag = "--fastGWA-lr";
    if(options_in.find(curFlag) != options_in.end()){
        processFunctions.push_back("fast_fam");
        if(options.find("grmsparse_file") != options.end()){
            LOGGER.e(0, "--fastGWA-lr is incompatible with --grm-sparse.");
        }
        returnValue++;
        options_in.erase(curFlag);
    }

    curFlag = "--ge";
    if(options_in.find(curFlag) != options_in.end()){
        if(options_in[curFlag].size() == 2){
            options["G"] = options_in[curFlag][0];
            options["E"] = options_in[curFlag][1];
        }else{
            LOGGER.e(0, curFlag + " can't handle more than 2 values.");
        }
        options_in.erase(curFlag);
    }

    /*
    curFlag = "--qcovar";
    if(options_in.find(curFlag) != options_in.end()){
        if(options_in[curFlag].size() == 1){
            options["concovar"] = options_in[curFlag][0];
        }else{
            LOGGER.e(0, curFlag + "can't deal with covar other than 1");
        }
    }
    */

    options["inv_method"] = "ldlt";
    vector<string> flags = {"--cg", "--ldlt", "--llt", "--pardiso", "--tcg", "--lscg", "--inv-t1"};
    for(auto curFlag : flags){
        if(options_in.find(curFlag) != options_in.end()){
            boost::erase_all(curFlag, "--");
            options["inv_method"] = curFlag;
            options_in.erase(curFlag);
        }
    }

    options["VgEstMethod"] = "REML";
    curFlag = "--est-vg";
    if(options_in.find(curFlag) != options_in.end()){
        auto cur_option = options_in[curFlag];
        if(cur_option.size() >=1){
            options["VgEstMethod"] = cur_option[0];
        }
    }

    curFlag = "--save-inv";
    if(options_in.find(curFlag) != options_in.end()){
        options["save_inv"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--save-bin";
    if(options_in.find(curFlag) != options_in.end()){
        options["save_bin"] = "yes";
        //options_in.erase(curFlag);
    }

    curFlag = "--no-marker";
    if(options_in.find(curFlag) != options_in.end()){
        options["no_marker"] = "yes";
        //options_in.erase(curFlag);
    }

    curFlag = "--nofilter";
    if(options_in.find(curFlag) != options_in.end()){
        options["no_filter"] = "yes";
        //options_in.erase(curFlag);
    }


    curFlag = "--c-inf";
    if(options_in.find(curFlag) != options_in.end()){
        options["c-inf"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--c-inf-no-filter";
    if(options_in.find(curFlag) != options_in.end()){
        options["c-inf-no-filter"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--load-inv";
    if(options_in.find(curFlag) != options_in.end()){
        if(options_in[curFlag].size() == 1){
            options["inv_file"] = options_in[curFlag][0];
        }else{
            LOGGER.e(0, "--load-inv can't load multiple files.");
        }
        options_in.erase(curFlag);
    }

    curFlag = "--model-only";
    bool model_only = false;
    if(options_in.find(curFlag) != options_in.end()){
        options["model_only"] = "yes";
        model_only = true;
        options_in.erase(curFlag);
    }

    curFlag = "--load-model";
    options["model_file"] = "";
    if(options_in.find(curFlag) != options_in.end()){
        if(options_in[curFlag].size() == 1){
           options["model_file"] = options_in[curFlag][0];
           if(!checkFileReadable(options["model_file"] + ".mdl.id")){
              LOGGER.e(0, "can't read " + options["model_file"] + ".mdl.id.");
           }
           if(checkFileReadable(options["model_file"] + ".mdl.bin")){
              processFunctions.push_back("fast_fam");
           }else if(checkFileReadable(options["model_file"] + ".mdl.bin2")){
              processFunctions.push_back("fast_fam");
              options["binary"] = "yes";
           }else{
              LOGGER.e(0, "can't read the model binary file.");
           }
           returnValue++;
        }else{
           LOGGER.e(0, "--load-model can't load multiple files.");
        }
       options_in.erase(curFlag);
    }

    if(model_only && options["model_file"] != ""){
        LOGGER.e(0, "can't generate and load model at the same time.");
    }
    
    curFlag = "--burden";
    if(options_in.find(curFlag) != options_in.end()){
        options["regiontest"] = "burden";
    }

    curFlag = "--set-list";
    if(options_in.find(curFlag) != options_in.end()){
        if(options_in[curFlag].size() >= 1){
            options["geneset"] = options_in[curFlag][0];
            if(!checkFileReadable(options["geneset"])){
                LOGGER.e(0, "can't read " + options["geneset"]);
            }
        }
    }


    if(options.find("regiontest") != options.end()){
        if(options.find("model_file") == options.end()){
            LOGGER.e(0, "set based test can only be applied to the 2nd step with the --load-model flag.");
        }
        if(options.find("geneset") == options.end()){
            LOGGER.e(0, "set based test requires to load the set list by the --set-list flag.");
        }
    }


    curFlag = "--save-pheno";
    if(options_in.find(curFlag) != options_in.end()){
        options["save_pheno"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--joint-covar";
    if(options_in.find(curFlag) != options_in.end()){
        options["adj_covar"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--force-gwa";
    if(options_in.find(curFlag) != options_in.end()){
        options["force_gwa"] = "yes";
        options_in.erase(curFlag);
    }

    // no contrain by default
    options["no_constrain"] = "yes";
    curFlag = "--gwa-no-constrain";
    if(options_in.find(curFlag) != options_in.end()){
        options["no_constrain"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--reml-detail";
    if(options_in.find(curFlag) != options_in.end()){
        options["reml_detail"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--h2-limit";
    if(options_in.find(curFlag) != options_in.end()){
        auto cur_option = options_in[curFlag];
        if(cur_option.size() >=1){
            options_d["h2_limit"] = stod(cur_option[0]);
        }
    }else{
        //default value 1.6
        options_d["h2_limit"] = 1.6;
    }

    //random seed
    curFlag = "--seed";
    if(options_in.find(curFlag) != options_in.end()){
        auto cur_option = options_in[curFlag];
        if(cur_option.size() >=1){
            options_d["seed"] = stod(cur_option[0]);
        }
    }else{
        options_d["seed"] = 0;
    }


    curFlag = "--rel-only";
    if(options_in.find(curFlag) != options_in.end()){
        options["rel_only"] = "yes";
        options_in.erase(curFlag);
    }else{
        options["rel_only"] = "no";
    }

    curFlag = "--cv-threshold";
    if(options_in.find(curFlag) != options_in.end()){
        auto cur_option = options_in[curFlag];
        if(cur_option.size() >=1){
            options_d["cv_threshold"] = stod(cur_option[0]);
        }
    }else{
        options_d["cv_threshold"] = 0.1;
    }

    curFlag = "--tao-start";
    if(options_in.find(curFlag) != options_in.end()){
        auto cur_option = options_in[curFlag];
        if(cur_option.size() >=1){
            options_d["tao_start"] = stod(cur_option[0]);
        }
    }else{
        options_d["tao_start"] = 0.0;
    }


    curFlag = "--optimal-rho";
    if(options_in.find(curFlag) != options_in.end()){
        options["optimal-rho"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--envir";
    if(options_in.find(curFlag) != options_in.end()){
      options["envir"] = "yes";
      options_in.erase(curFlag);
    }
        
    curFlag = "--noSandwich";
    if(options_in.find(curFlag) != options_in.end()){
        options["noSandwich"] = "yes";
        options_in.erase(curFlag);
    }

    curFlag = "--grid-size";
    if(options_in.find(curFlag) != options_in.end()){
        auto cur_option = options_in[curFlag];
        if(cur_option.size() >=1){
            options_d["grid_size"] = stod(cur_option[0]);
        }
    }else{
        //default value 11
        options_d["grid_size"] = 11;
    }

    return returnValue;
}


void FastFAM::processMain(){
    vector<function<void (uintptr_t *, const vector<uint32_t> &)>> callBacks;
    for(auto &process_function : processFunctions){
        if(process_function == "fast_fam"){
            FastFAM ffam;
            if(options.find("save_inv") != options.end()){
                LOGGER.i(0, "Use --load-inv to load the V inverse file for fastGWA");
                return;
            }
            if(options.find("model_only") != options.end()){
                LOGGER.i(0, "Use \"--load-model " + options["out"] + "\" to load the model file for further analysis.");
                LOGGER << "Note: the sample IDs (FID IID) in the model file must also present in the genotype file for further association step." << std::endl;
                return;
            }
            bool bBinary = false;
            if(options.find("binary") != options.end()){
                bBinary = true;
            }
            if(options.find("grmsparse_file") != options.end() && ffam.fam_flag){
                if(options.find("envir") != options.end()){
                    if (options.find("noSandwich") != options.end()){
                        LOGGER.i(0, "\nPerforming fastGWA-GE mixed model association analysis using model-based variance estimator...");
                        callBacks.push_back(bind(&FastFAM::calculate_mixed_2df, &ffam, _1, _2));
                    }else{
                        LOGGER.i(0, "\nPerforming fastGWA-GE mixed model association analysis using model-robust (sandwich) variance estimator...");
                        callBacks.push_back(bind(&FastFAM::calculate_mixed_2df_sandwich, &ffam, _1, _2));                        
                    }
                }else{
                    if(bBinary){
                        LOGGER.i(0, "\nPerforming fastGWA generalized linear mixed model association analysis...");
                        callBacks.push_back(bind(&FastFAM::calculate_spa, &ffam, _1, _2));
                    }else{
                        if(options.find("grammar") == options.end()){
                            LOGGER.i(0, "\nPerforming fastGWA mixed model association analysis (exact test)...");
                            callBacks.push_back(bind(&FastFAM::calculate_fam, &ffam, _1, _2));
                        }else{
                            LOGGER.i(0, "\nPerforming fastGWA mixed model association analysis...");
                            callBacks.push_back(bind(&FastFAM::calculate_grammar, &ffam, _1, _2));
                        }
                    }
                }
            }else{
                if (options.find("envir") != options.end()){
                    if (options.find("noSandwich") != options.end()){
                        LOGGER.i(0, "\nPerforming fastGWA-GE linear regression analysis using model-based variance estimator...");
                        callBacks.push_back(bind(&FastFAM::calculate_gwa_2df, &ffam, _1, _2));
                    }else{
                        LOGGER.i(0, "\nPerforming fastGWA-GE linear regression analysis using model-robust (sandwich) variance estimator...");
                        callBacks.push_back(bind(&FastFAM::calculate_gwa_2df_sandwich, &ffam, _1, _2));
                    }
                }else{
                    LOGGER.i(0, "\nPerforming fastGWA linear regression analysis...");
                    callBacks.push_back(bind(&FastFAM::calculate_gwa, &ffam, _1, _2));
                }
            }
            if(options.find("regiontest") != options.end()){
                ffam.processFAMreg();
            }else{
                ffam.processFAM(callBacks);
            }
            callBacks.clear();
        }
    }
}

void FastFAM::processFAMreg(){
    if(options.find("geneset") == options.end()){
        LOGGER.e(0, "can't find the region set. Plese specify it by the --set-list flag.");
    }
    sFileName = options["out"]; 

    LOGGER << "fastGWA-BB results will be saved in text format to [" << sFileName << "]." << std::endl;
    osOut.open(sFileName.c_str());
    vector<string> header = {"GENE", "START", "END", "VAR_N", "MAF_MEAN", "BETA", "SE", "P"};
    if(bBinary){
        header = {"GENE", "START", "END", "VAR_N", "MAF_MEAN", "T", "SE_T", "P_noSPA", "BETA", "SE",  "P",  "CONVERGE"};
    }
    string header_string = boost::algorithm::join(header, "\t");
    if(osOut.bad()){
        LOGGER.e(0, "can't open [" + sFileName + "] to write.");
    }
    osOut << header_string << std::endl;

    string gfile = options["geneset"];
    LOGGER.i(0, "Reading gene list file from [" + gfile + "]...");
    vector<pair<string, vector<uint32_t>>> genelist = marker->read_gene(gfile);
    int numGeneBlock = genelist.size();
    if(numGeneBlock == 0){
        LOGGER.e(0, "can't find a valid gene in the genotype file.");
    }else{
        LOGGER << "Processing " << numGeneBlock << " genes by " << options["regiontest"] << " test..." << std::endl;
    }

    double preMiss = geno->getFilterMiss();
    if(preMiss < 1e-10){
        geno->setFilterMiss(0.9);
        LOGGER << "  Filtering out variants with missingness rate > 0.10, or customise it with --geno flag." << std::endl;
    }

    //
    vector<function<void (uintptr_t *, const vector<uint32_t> &)>> calls;
    calls.push_back(bind(&FastFAM::getX, this, _1, _2));
    int nMarker = 100;

    LOGGER.ts("LOOP_SET_TOT");
 
    for(int i = 0; i < numGeneBlock; i++){
        string genename = genelist[i].first;
        vector<uint32_t> &glist = genelist[i].second;
        int curNumMarker = glist.size();
        gAF.resize(curNumMarker);
        gX.resize(num_indi, curNumMarker);
        bValids.resize(curNumMarker);
        gN.resize(curNumMarker);
        num_gene_index = 0;
        geno->loopDouble(glist, nMarker, true, false, false, false, calls, false);
        
        int move_seg = 0;
        vector<uint32_t> keptIndex;
        for(int j = 0; j < curNumMarker; j++){
            if(!bValids[j]){
                move_seg++;
            }else{
                if(move_seg != 0){
                    gX.col(j - move_seg) = gX.col(j);
                    gAF[j - move_seg] = gAF(j);
                    gN[j - move_seg] = gN(j);
                }
                keptIndex.push_back(glist[j]);
            }
        }
        int nValidMarker = curNumMarker - move_seg;
        if(nValidMarker == 0){
            continue;
        }
        gAF.conservativeResize(nValidMarker);
        gX.conservativeResize(num_indi, nValidMarker);
        gN.conservativeResize(nValidMarker);

        vector<bool> bFlip(nValidMarker, false);
        for(int j = 0; j < nValidMarker; j++){
            if(gAF(j) > 0.5){
                double af = 1.0 - gAF(j);
                gAF(j) = af;
                gX.col(j) = -gX.col(j).array() + 2.0;
                bFlip[j] = true;
            }

        }

        VectorXd weight = StatLib::weightBetaMAF(gAF, 1, 25);
        SpMat weights(nValidMarker, nValidMarker);
        weights.setIdentity();
        weights.diagonal() = weight; 

        MatrixXd GW = gX * weights;

        VectorXd GW1 = GW * MatrixXd::Ones(nValidMarker, 1);
        VectorXd GW2 = GW1;
        if(bPreciseCovar){
            conditionCovarBinReg(GW1);
        }else{
            GW1 = GW1.array() - GW1.mean();
        }
        double varSNP = std::sqrt(GW1.dot(dWp.cwiseProduct(GW1)) * c_inf);
        SPARes res;
        res.score = GW1.dot(phenoVecMu);
        double chisq = std::abs(res.score) / varSNP;
        res.p = StatLib::pchisqd1(chisq * chisq);

        res.bConverge = true;
        if( chisq < spaCutOff){
            res.p_adj = res.p;
        }else{
            vector<uint32_t> index0;
            index0.reserve(num_indi);
            double thresh = 1e-6;
            for(uint32_t i = 0; i < num_indi; i++){
                if(GW2[i] > thresh){
                    index0.push_back(i);
                }
            }

            if(!bPreciseCovar) conditionCovarBinReg(GW1);
            double q = GW1.dot(phenoVec);
            double qinv = q - res.score - res.score;
            SPA spa(q, qinv, GW1, index0);
            spa.saddleProb(&res);
        }

        int rConverge = (int)res.bConverge;
        double temp_beta = res.score / (varSNP *varSNP);
        double se = std::abs(temp_beta) / sqrt(StatLib::qchisqd1(res.p_adj));

        osOut << genename << "\t" << gAF.size() << "\t" << gAF.mean() << "\t"
            << res.score << "\t" << varSNP << "\t" << res.p << "\t"
            << temp_beta << "\t" << se << "\t" << res.p_adj << "\t" << rConverge << std::endl;

        int curNGene = i + 1;
        if(curNGene % 2000 == 0) {
            float elapse_time = LOGGER.tp("LOOP_SET_TOT");
            float finished_percent = (float) curNGene / numGeneBlock;
            float remain_time = (1.0 / finished_percent - 1) * elapse_time / 60;

            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << "proceeded " << curNGene << " genes. Estimated time remaining " << remain_time << " min"; 

            LOGGER.i(1, ss.str());
        }

    }
    osOut.close();
    //osSets.close();

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << "All " << numGeneBlock << " genes finished in " << LOGGER.tp("LOOP_SET_TOT") << " sec";
    LOGGER.i(0, ss.str());
}

void FastFAM::getX(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);
    for(int i = 0; i < num_marker; i++){
        int index_cur_marker = num_gene_index + i;
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;
        geno->getGenoDouble(genobuf, i, &item);
        
        bValids[index_cur_marker] = item.valid;
        if(item.valid){
            Map< VectorXd > xvec(item.geno.data(), num_indi);
            gX.col(index_cur_marker) = xvec;
            gAF[index_cur_marker] = item.af;
            gN[index_cur_marker] = item.nValidN;
        }
    }
    num_gene_index += num_marker;
}

bool FastFAM::covarGLM(const VectorXd& phenoVec, const MatrixXd& covar, Ref<VectorXd> est_beta, int maxIter, double thresh){
    bool bConverge = false;
    int curIter = 0;

    const Eigen::Index n_rows = covar.rows();
    const Eigen::Index n_cols = covar.cols();
    if (n_rows == 0 || n_cols == 0) {
        return true;
    }

    // Warm-start intercept at marginal logit(mean(y)) when starting from zero
    if (est_beta.squaredNorm() == 0.0) {
        Eigen::Index intercept_col = -1;
        const Eigen::Index cand_cols[2] = { n_cols - 1, 0 };
        for (int ci = 0; ci < 2; ++ci) {
            const Eigen::Index c_idx = cand_cols[ci];
            const double * __restrict cp = covar.col(c_idx).data();
            bool all_ones = true;
            for (Eigen::Index r = 0; r < n_rows; ++r) {
                if (cp[r] != 1.0) {
                    all_ones = false;
                    break;
                }
            }
            if (all_ones) {
                intercept_col = c_idx;
                break;
            }
        }
        if (intercept_col >= 0) {
            const double y_mean = phenoVec.mean();
            if (y_mean > 0.0 && y_mean < 1.0) {
                est_beta[intercept_col] = std::log(y_mean / (1.0 - y_mean));
            }
        }
    }

    const Eigen::Index kTile = 512;
    const Eigen::Index n_tiles = (n_rows + kTile - 1) / kTile;
    vector<double> ll_partials(static_cast<size_t>(n_tiles), 0.0);
    vector<MatrixXd> xtwx_partials(static_cast<size_t>(n_tiles));
    vector<VectorXd> comp2_partials(static_cast<size_t>(n_tiles));

    MatrixXd XtWX(n_cols, n_cols);
    VectorXd comp2(n_cols);
    VectorXd new_beta(n_cols);

    auto eval_glm_state = [&](const Ref<const VectorXd> &beta_vec, bool need_derivs, double &out_ll) {
        ScopedSingleThreadBLAS blas_guard;
        #pragma omp parallel if(n_rows >= 4096)
        {
            ScopedSingleThreadBLAS::ThreadLocal tl_guard;
            const Eigen::Index max_b = std::min<Eigen::Index>(kTile, n_rows);
            VectorXd theta_blk(max_b);
            VectorXd diags_blk(max_b);
            VectorXd resi_blk(max_b);
            MatrixXd scaled_blk(max_b, need_derivs ? n_cols : 1);

            #pragma omp for schedule(static)
            for (Eigen::Index t = 0; t < n_tiles; ++t) {
                const Eigen::Index r0 = t * kTile;
                const Eigen::Index rows = std::min<Eigen::Index>(kTile, n_rows - r0);
                auto X_blk = covar.middleRows(r0, rows);
                auto th = theta_blk.head(rows);
                th.noalias() = X_blk * beta_vec;

                const double * __restrict y_ptr = phenoVec.data() + r0;
                const double * __restrict th_ptr = th.data();
                double * __restrict d_ptr = diags_blk.data();
                double * __restrict r_ptr = resi_blk.data();

                double tile_ll = 0.0;
                for (Eigen::Index r = 0; r < rows; ++r) {
                    const double eta = th_ptr[r];
                    const double y_i = y_ptr[r];
                    const double abs_eta = std::abs(eta);
                    const double softplus = (eta > 0.0 ? eta : 0.0) + std::log1p(std::exp(-abs_eta));
                    tile_ll += y_i * eta - softplus;
                    if (need_derivs) {
                        const double m = 1.0 / (std::exp(-eta) + 1.0);
                        d_ptr[r] = m * (1.0 - m);
                        r_ptr[r] = y_i - m;
                    }
                }
                ll_partials[static_cast<size_t>(t)] = tile_ll;
                if (need_derivs) {
                    auto S_blk = scaled_blk.topRows(rows);
                    S_blk = X_blk.array().colwise() * diags_blk.head(rows).array();
                    xtwx_partials[static_cast<size_t>(t)].noalias() = X_blk.transpose() * S_blk;
                    comp2_partials[static_cast<size_t>(t)].noalias() = X_blk.transpose() * resi_blk.head(rows);
                }
            }
        }
        double total_ll = 0.0;
        if (need_derivs) {
            XtWX.setZero();
            comp2.setZero();
        }
        for (Eigen::Index t = 0; t < n_tiles; ++t) {
            total_ll += ll_partials[static_cast<size_t>(t)];
            if (need_derivs) {
                XtWX.noalias() += xtwx_partials[static_cast<size_t>(t)];
                comp2.noalias() += comp2_partials[static_cast<size_t>(t)];
            }
        }
        out_ll = total_ll;
    };

    double cur_ll = -std::numeric_limits<double>::infinity();
    eval_glm_state(est_beta, true, cur_ll);

    while(curIter < maxIter){
        auto XtWX_solver = XtWX.colPivHouseholderQr();
        if(!XtWX_solver.isInvertible()){
            LOGGER.e(0, "XtWX is not invertible.");
        }
        VectorXd step = XtWX_solver.solve(comp2);
        if (!step.array().isFinite().all()) {
            break;
        }

        double alpha = 1.0;
        new_beta = est_beta + step;
        double next_ll = -std::numeric_limits<double>::infinity();
        const double raw_thresh = ((est_beta - new_beta).array().abs() /
                                   (est_beta.array().abs() + new_beta.array().abs() + 1e-12)).maxCoeff();

        if (raw_thresh < thresh) {
            eval_glm_state(new_beta, false, next_ll);
            if (std::isfinite(next_ll) && next_ll >= cur_ll - 1e-10 * (1.0 + std::abs(cur_ll))) {
                est_beta = new_beta;
                bConverge = true;
                break;
            }
        }

        // Evaluate next state with Armijo monotonicity safeguard against rare-trait / separation overshoot
        bool accepted = false;
        for (int bt = 0; bt < 12; ++bt) {
            new_beta = est_beta + alpha * step;
            eval_glm_state(new_beta, true, next_ll);
            if (std::isfinite(next_ll) && next_ll >= cur_ll - 1e-10 * (1.0 + std::abs(cur_ll))) {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            break;
        }

        double cur_thresh = ((est_beta - new_beta).array().abs() /
                             (est_beta.array().abs() + new_beta.array().abs() + 1e-12)).maxCoeff();
        est_beta = new_beta;
        cur_ll = next_ll;
        if(cur_thresh < thresh){
            bConverge = true;
            break;
        }
        curIter++;
    }
    return bConverge;
}

double varVector(const Ref<VectorXd> vec){
    const Eigen::Index n = vec.size();
    if (n <= 1) return 0.0;
    const double mean = vec.mean();
    return (vec.array() - mean).matrix().squaredNorm() / static_cast<double>(n - 1);
}

bool checkNAN(const MatrixXd &mat){
    return (!mat.array().isFinite()).any();
}


// REML log-likelihood at a single tao grid point for fastGWA-BB-REML.
// Returns -infinity (rather than aborting) when V = W + tao*K is not
// positive definite or when XtV^-1X is singular: the sparse GRM from
// --make-bK thresholding can have negative eigenvalues, so some tao
// values legitimately have no valid log-likelihood. SimplicialLDLT
// reports Success on indefinite V, so we must also check d.minCoeff().
// binGridREML already skips non-finite logL values via bEndNAN. A
// converged tao near the infeasibility boundary may indicate a biased
// estimate and warrants a denser or PSD-regularised GRM.
double FastFAM::binLogL(double cur_tao, const SpMat& fam, const SpMat& W, const Ref<VectorXd> Y, const Ref<MatrixXd> X){
    SpMat V = W + cur_tao * fam;
    Eigen::SimplicialLDLT<SpMat> solverV;
    solverV.compute(V);
    if(solverV.info() != Eigen::Success){
        return -std::numeric_limits<double>::infinity();
    }

    VectorXd d = solverV.vectorD();
    if(d.minCoeff() <= 0.0){
        return -std::numeric_limits<double>::infinity();
    }
    double logdet_V = d.array().log().sum();

    MatrixXd ViX = solverV.solve(X); // n*c
    if(solverV.info() != Eigen::Success){
        return -std::numeric_limits<double>::infinity();
    }

    auto XtVX_solver = (X.transpose() * ViX).colPivHouseholderQr();
    if(!XtVX_solver.isInvertible()){
        return -std::numeric_limits<double>::infinity();
    }
    double logdet_XtVX = XtVX_solver.logAbsDeterminant();

    MatrixXd inv_XtVX_ViX = XtVX_solver.solve(ViX.transpose()); // c*n
    VectorXd PY = solverV.solve(Y) - ViX * (inv_XtVX_ViX * Y);

    return(-0.5 * (logdet_V + logdet_XtVX + Y.dot(PY)));

}

bool FastFAM::binGridREML(const SpMat& fam, Ref<VectorXd> est_a, int maxIter, double threshold){
    LOGGER << "Performing GLM to get the starting values of beta for the covariates..." << std::endl;
    if(!covarGLM(phenoVec, covar, est_a, maxIter, threshold)){
        LOGGER.w(0, "GLM didn't converge.");
    }
    LOGGER << "GLM finished, fixed effects: " << est_a.transpose() << std::endl;

    binSolverV.reset(new BinSparseLDLT());
    binSolverV->init(fam);
    binSolverV->prepareDesign(covar);
    vector<double> w_s, w_r_perm;

   //init first run
    VectorXd Xa_b = covar * est_a;
    VectorXd var_mu_i;
    VectorXd Y;
    updateLogisticWorkingFromEta(Xa_b, phenoVec, mu, var_mu_i, Y);
    const double init_varY = varVector(Y);
    LOGGER << "Initial Var(Y): " << init_varY << std::endl;

    auto run_blup_step = [&](double tao_val,
                             const Ref<const VectorXd> &in_var_mu,
                             const Ref<const VectorXd> &in_Y,
                             VectorXd &out_a,
                             VectorXd &out_eta) -> bool {
        return binSolverV->solveBlupStep(covar, in_var_mu, in_Y, tao_val, w_s, w_r_perm, out_a, out_eta);
    };

    auto polish_pirls_at_tao = [&](double tao_val,
                                   Ref<VectorXd> io_a,
                                   VectorXd &io_eta,
                                   VectorXd &io_mu,
                                   VectorXd &io_var_mu,
                                   VectorXd &io_Y,
                                   int max_polish_steps) -> bool {
        for (int p_it = 0; p_it < max_polish_steps; ++p_it) {
            VectorXd cand_a, cand_eta;
            if (!run_blup_step(tao_val, io_var_mu, io_Y, cand_a, cand_eta)) {
                return false;
            }
            const double max_deta = (cand_eta - io_eta).lpNorm<Eigen::Infinity>();
            double damp = 1.0;
            if (max_deta > 2.5) {
                damp = std::max(0.35, 2.5 / max_deta);
            }
            io_a = io_a + damp * (cand_a - io_a);
            io_eta = io_eta + damp * (cand_eta - io_eta);
            updateLogisticWorkingFromEta(io_eta, phenoVec, io_mu, io_var_mu, io_Y);
            if (max_deta < 1e-3 && damp == 1.0) {
                break;
            }
        }
        return true;
    };

    double cur_tao = options_d["tao_start"] * init_varY;
    bool have_init_zero_score = false;
    double cached_ll0 = 0.0, cached_s0 = 0.0, cached_f0 = 0.0;
    if (options_d["tao_start"] == 0.0) {
        // Compute exact REML score & Fisher info at tao = 0 to warm-start tao^(0)
        binSolverV->updateWorkingState(var_mu_i, Y, w_s, w_r_perm);
        if (binSolverV->evalZeroTaoScoreAndInfo(w_s.data(), w_r_perm.data(), cached_ll0, cached_s0, cached_f0)) {
            have_init_zero_score = true;
            if (cached_s0 > 0.0 && cached_f0 > 1e-30) {
                const double raw_fs = cached_s0 / cached_f0;
                // Padé logit-curvature attenuation prevents initial Fisher-scoring overshoot on rare traits
                double warm_tao = std::min(0.5 * init_varY, raw_fs / (1.0 + 0.45 * raw_fs));
                for (int try_k = 0; try_k < 10 && warm_tao > 1e-8; ++try_k) {
                    double ld_try = 0.0;
                    if (binSolverV->factorizeRelatedOnly(w_r_perm.data(), warm_tao, binSolverV->main_ws, ld_try) &&
                        (binSolverV->n_s == 0 || binSolverV->min_w_s + warm_tao * binSolverV->min_k_diag_s > 0.0)) {
                        cur_tao = warm_tao;
                        break;
                    }
                    warm_tao *= 0.5;
                }
            }
        }
    }

    bool state_modified_at_init = false;
    if (cur_tao > 0.0 || options_d["tao_start"] != 0.0) {
        if (!polish_pirls_at_tao(cur_tao, est_a, Xa_b, mu, var_mu_i, Y, 4)) {
            LOGGER.e(0, "can't invert the V matrix!");
        }
        state_modified_at_init = true;
    }
    LOGGER << "Initializing with tao: " << cur_tao << ", Var(Y): " << varVector(Y) << ", fixed effects: " << est_a.transpose() << "." << std::endl;
 

    bool b_reml = false;
    std::ofstream out;
    if(options.find("reml_detail") != options.end()){
        b_reml = true;
        out.open(options["out"] + ".reml");
        if(!out){
            LOGGER.e(0, "failed to open the " + options["out"] + ".reml.");
        }

        out << std::setprecision( std::numeric_limits<double>::digits10);
        out << "index\tlogL\tVg\tVe" << std::endl;

    }

    //double cur_tao = -100;
    int curIter = 0;
    bool bConverge = false;
    double tol = 5e-5;
    double abTol = 1e-6;
    int numAbTol = 0;
    const double init_tao_ref = cur_tao;
    double saved_tao_scale = 0.0;
    vector<double> hisTao(maxIter);
    int max_omp_threads = 1;
#ifdef _OPENMP
    max_omp_threads = std::max(1, omp_get_max_threads());
#endif
    const int32_t n_cols_aug = static_cast<int32_t>(covar.cols()) + 1;
    vector<BinSparseLDLT::Workspace> thread_ws(max_omp_threads);
    {
        ScopedSingleThreadBLAS blas_guard;
        #pragma omp parallel num_threads(max_omp_threads)
        {
            ScopedSingleThreadBLAS::ThreadLocal tl_guard;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            if (tid < max_omp_threads) {
                thread_ws[tid].initEval(binSolverV->n, binSolverV->nnz_L, n_cols_aug);
            }
        }
    }

    bool has_prev_fp = false;
    double prev_fp_tao = 0.0;
    double prev_fp_res = 0.0;
    bool prev_was_secant = false;

    while(curIter < maxIter && (!bConverge) && numAbTol < 5){
        LOGGER << "------------------------------------" << std::endl;
        double pre_tao = cur_tao;
        VectorXd pre_est_a = est_a;
        binSolverV->updateWorkingState(var_mu_i, Y, w_s, w_r_perm);

        double varY = varVector(Y);
        const double max_tao_limit = std::max(varY * 1.01, pre_tao * 2.0);
        int total_infeasible = 0;
        int total_points = 0;

        vector<std::pair<double, double>> eval_cache;
        eval_cache.reserve(256);

        auto find_cached = [&](double val, double &out_ll) -> bool {
            if (eval_cache.empty()) return false;
            const double eps = 1e-15 * std::max(1.0, std::abs(val));
            auto it = std::lower_bound(
                eval_cache.begin(), eval_cache.end(), val - eps,
                [](const std::pair<double, double> &a, double v){ return a.first < v; });
            if (it != eval_cache.end() && std::abs(it->first - val) <= eps) {
                out_ll = it->second;
                return true;
            }
            return false;
        };

        auto insert_cached = [&](double val, double ll) {
            auto it = std::lower_bound(
                eval_cache.begin(), eval_cache.end(), val,
                [](const std::pair<double, double> &a, double v){ return a.first < v; });
            if (it == eval_cache.end() || it->first != val) {
                eval_cache.insert(it, std::make_pair(val, ll));
            }
            total_points++;
            if (!std::isfinite(ll)) {
                total_infeasible++;
            }
        };

        auto eval_single = [&](double val) -> double {
            val = std::max(0.0, std::min(max_tao_limit, val));
            double ll = 0.0;
            if (find_cached(val, ll)) return ll;
            ScopedSingleThreadBLAS blas_guard;
            ScopedSingleThreadBLAS::ThreadLocal tl_guard;
            ll = binSolverV->evalLogL(w_s.data(), w_r_perm.data(), val, thread_ws[0]);
            insert_cached(val, ll);
            return ll;
        };

        auto eval_batch_parallel = [&](const vector<double> &cands) {
            vector<double> todo;
            todo.reserve(cands.size());
            for (size_t i = 0; i < cands.size(); ++i) {
                double v = std::max(0.0, std::min(max_tao_limit, cands[i]));
                double dummy = 0.0;
                if (find_cached(v, dummy)) continue;
                bool dup = false;
                const double eps = 1e-15 * std::max(1.0, std::abs(v));
                for (size_t j = 0; j < todo.size(); ++j) {
                    if (std::abs(todo[j] - v) <= eps) { dup = true; break; }
                }
                if (!dup) todo.push_back(v);
            }
            const int n_todo = static_cast<int>(todo.size());
            if (n_todo == 0) return;
            vector<double> res_ll(n_todo, -std::numeric_limits<double>::infinity());
            {
                ScopedSingleThreadBLAS blas_guard;
                #pragma omp parallel
                {
                    #pragma omp single
                    {
#ifdef _OPENMP
                        int active_threads = omp_get_num_threads();
                        if (active_threads > static_cast<int>(thread_ws.size())) {
                            int old_sz = static_cast<int>(thread_ws.size());
                            thread_ws.resize(active_threads);
                            for (int t = old_sz; t < active_threads; ++t) {
                                thread_ws[t].initEval(binSolverV->n, binSolverV->nnz_L, n_cols_aug);
                            }
                        }
#endif
                    }
                    ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                    int tid = 0;
#ifdef _OPENMP
                    tid = omp_get_thread_num();
#endif
                    BinSparseLDLT::Workspace &ws = thread_ws[tid];
                    #pragma omp for schedule(dynamic)
                    for (int k = 0; k < n_todo; ++k) {
                        res_ll[k] = binSolverV->evalLogL(w_s.data(), w_r_perm.data(), todo[k], ws);
                    }
                }
            }
            for (int k = 0; k < n_todo; ++k) {
                insert_cached(todo[k], res_ll[k]);
            }
            if (b_reml) {
                for (int k = 0; k < n_todo; ++k) {
                    out << k << "\t" << res_ll[k] << "\t" << todo[k] << std::endl;
                }
                out << "----------------------------" << std::endl;
            }
        };

        // 1. Exact logL(0), score(0), and Fisher info I(0) at tao = 0 when needed
        double logL_0 = -std::numeric_limits<double>::infinity();
        double score_0 = 1.0;
        double fisher_0 = 0.0;
        bool evaled_zero = false;
        auto ensure_zero_eval = [&]() {
            if (evaled_zero) return;
            evaled_zero = true;
            if (curIter == 0 && !state_modified_at_init && have_init_zero_score) {
                logL_0 = cached_ll0;
                score_0 = cached_s0;
                fisher_0 = cached_f0;
                insert_cached(0.0, logL_0);
            } else if (binSolverV->evalZeroTaoScoreAndInfo(w_s.data(), w_r_perm.data(), logL_0, score_0, fisher_0)) {
                insert_cached(0.0, logL_0);
            } else {
                logL_0 = eval_single(0.0);
                score_0 = 0.0;
                fisher_0 = 0.0;
            }
        };

        if (curIter == 0 || pre_tao == 0.0 || saved_tao_scale <= 0.0) {
            ensure_zero_eval();
            const double fs_t = (fisher_0 > 1e-30 && score_0 > 0.0) ? (score_0 / fisher_0) : 0.0;
            const double info_sc = (fisher_0 > 1e-30) ? (1.0 / std::sqrt(fisher_0)) : (0.01 * max_tao_limit);
            saved_tao_scale = std::max(1e-8, std::min(max_tao_limit, std::max(fs_t, info_sc)));
        }

        const double fs_tao = (evaled_zero && fisher_0 > 1e-30 && score_0 > 0.0) ? (score_0 / fisher_0) : pre_tao;
        const double tao_scale = std::min(max_tao_limit, saved_tao_scale);
        const int wave_width = 10;

        auto make_uniform_grid = [](double lo, double hi, int count, vector<double> &grid) {
            grid.clear();
            if (count <= 1 || hi <= lo) {
                grid.push_back(lo);
                return;
            }
            grid.reserve(count);
            const double st = (hi - lo) / static_cast<double>(count - 1);
            for (int i = 0; i < count; ++i) {
                grid.push_back(lo + i * st);
            }
            grid.back() = hi;
        };

        // 2. Deterministic parallel bracketing coupled to warm-start / outer PQL trajectory
        if (curIter == 0) {
            vector<double> cands;
            if (pre_tao > 0.0 && pre_tao < max_tao_limit) {
                const int n_near = wave_width + 2;
                const int n_far = wave_width - 2;
                cands.reserve(n_near + n_far + 4);
                cands.push_back(0.0);
                cands.push_back(pre_tao);
                if (fs_tao > 0.0 && fs_tao < max_tao_limit) cands.push_back(fs_tao);
                const double near_hi = std::min(max_tao_limit, std::max(2.5 * pre_tao, 2.0 * tao_scale));
                for (int i = 1; i <= n_near; ++i) {
                    const double frac = static_cast<double>(i) / static_cast<double>(n_near);
                    cands.push_back(near_hi * frac);
                }
                if (near_hi < max_tao_limit) {
                    for (int i = 1; i <= n_far; ++i) {
                        const double frac = static_cast<double>(i) / static_cast<double>(n_far);
                        cands.push_back(near_hi + (max_tao_limit - near_hi) * frac * frac);
                    }
                }
            } else {
                const int coarse_width = 2 * wave_width;
                const int n_low = coarse_width / 2;
                const int n_high = coarse_width - n_low;
                const double pivot = std::min(max_tao_limit * 0.25, std::max(8.0 * tao_scale, 1e-4 * max_tao_limit));
                cands.reserve(coarse_width + 3);
                cands.push_back(0.0);
                if (fs_tao > 0.0 && fs_tao < max_tao_limit) {
                    cands.push_back(fs_tao);
                }
                for (int i = 1; i <= n_low; ++i) {
                    const double frac = static_cast<double>(i) / static_cast<double>(n_low);
                    cands.push_back(pivot * frac * frac);
                }
                for (int i = 1; i <= n_high; ++i) {
                    const double frac = static_cast<double>(i) / static_cast<double>(n_high);
                    cands.push_back(pivot + (max_tao_limit - pivot) * frac);
                }
            }
            LOGGER << "Fine tuning within 0 ~ " << max_tao_limit << " with adaptive parallel waves (width " << wave_width << ")." << std::endl;
            eval_batch_parallel(cands);
        } else {
            if (pre_tao == 0.0 && score_0 <= 0.0) {
                const double probe_eps = std::min(max_tao_limit, 1e-7 * std::max(1.0, tao_scale));
                eval_single(probe_eps);
            } else {
                const double center = (pre_tao > 0.0) ? pre_tao : std::max(fs_tao, 0.1 * tao_scale);
                double z_lo = 0.0;
                double z_hi = max_tao_limit;
                if (curIter == 1) {
                    const double init_jump = std::abs(hisTao[0] - init_tao_ref);
                    const double rad1 = std::max(3.0 * init_jump, std::max(0.25 * center, 1e-4 * tao_scale));
                    z_lo = (center - rad1 < 0.20 * center) ? 0.0 : (center - rad1);
                    z_hi = std::min(max_tao_limit, center + rad1);
                } else {
                    const double prev_jump = std::abs(hisTao[curIter - 1] - hisTao[curIter - 2]);
                    const double rad = std::max(3.5 * prev_jump, std::max(2.5e-3 * center, 1e-5 * std::max(1.0, tao_scale)));
                    z_lo = (center - rad < 0.10 * center) ? 0.0 : (center - rad);
                    z_hi = std::min(max_tao_limit, center + rad);
                }
                if (z_lo == 0.0) {
                    ensure_zero_eval();
                }
                vector<double> cands;
                make_uniform_grid(z_lo, z_hi, wave_width + 1, cands);
                cands.push_back(pre_tao);
                eval_batch_parallel(cands);

                // Expand dynamically if maximum hits an interior search-window boundary
                for (int exp_it = 0; exp_it < 8; ++exp_it) {
                    int best_k = -1, last_finite_k = -1;
                    double best_ll = -std::numeric_limits<double>::infinity();
                    for (int k = 0; k < static_cast<int>(eval_cache.size()); ++k) {
                        if (std::isfinite(eval_cache[k].second)) {
                            last_finite_k = k;
                            if (eval_cache[k].second > best_ll) {
                                best_ll = eval_cache[k].second;
                                best_k = k;
                            }
                        }
                    }
                    if (best_k < 0) break;
                    const double best_t = eval_cache[best_k].first;
                    if (best_t >= z_hi - 1e-14 * std::max(1.0, z_hi) && z_hi < max_tao_limit && best_k == last_finite_k) {
                        const double new_hi = std::min(max_tao_limit, std::max(2.5 * z_hi, z_hi + 2.0 * tao_scale));
                        make_uniform_grid(z_hi, new_hi, wave_width + 1, cands);
                        z_hi = new_hi;
                        eval_batch_parallel(cands);
                    } else if (z_lo > 0.0 && best_t <= z_lo + 1e-14 * std::max(1.0, z_lo)) {
                        const double new_lo = (z_lo < 0.05 * tao_scale) ? 0.0 : (0.25 * z_lo);
                        if (new_lo == 0.0) ensure_zero_eval();
                        make_uniform_grid(new_lo, z_lo, wave_width + 1, cands);
                        z_lo = new_lo;
                        eval_batch_parallel(cands);
                    } else {
                        break;
                    }
                }
            }
        }

        // Gather finite evaluated points and check PD feasibility cliff
        auto get_best_bracket = [&](int &out_best_idx, vector<std::pair<double, double>> &finite_pts, double &first_infeas_above) {
            finite_pts.clear();
            first_infeas_above = std::numeric_limits<double>::infinity();
            out_best_idx = -1;
            double best_ll = -std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < eval_cache.size(); ++k) {
                if (std::isfinite(eval_cache[k].second)) {
                    const int idx = static_cast<int>(finite_pts.size());
                    finite_pts.push_back(eval_cache[k]);
                    if (eval_cache[k].second > best_ll) {
                        best_ll = eval_cache[k].second;
                        out_best_idx = idx;
                    }
                }
            }
            if (out_best_idx >= 0) {
                const double best_t = finite_pts[out_best_idx].first;
                for (size_t k = 0; k < eval_cache.size(); ++k) {
                    if (!std::isfinite(eval_cache[k].second) && eval_cache[k].first > best_t) {
                        first_infeas_above = eval_cache[k].first;
                        break;
                    }
                }
            }
        };

        vector<std::pair<double, double>> finite_pts;
        int best_idx = -1;
        double first_infeas_above = std::numeric_limits<double>::infinity();
        get_best_bracket(best_idx, finite_pts, first_infeas_above);

        if (best_idx < 0) {
            LOGGER.e(0, "all " + std::to_string(total_points) + " tao values gave a non-PD V matrix over [0, "
                + std::to_string(max_tao_limit) + "]. The sparse GRM is likely not PSD; try a denser or PSD-regularised GRM.");
        }

        // If optimum on curIter == 0 is not trivially certified at tao = 0, run one parallel zoom wave
        bool certified_zero = false;
        if (best_idx == 0 && score_0 <= 0.0 && finite_pts.size() >= 2) {
            const double eps_probe = std::min(finite_pts[1].first * 0.1, 1e-7 * std::max(1.0, tao_scale));
            if (eps_probe <= 0.0 || eval_single(eps_probe) <= logL_0) {
                certified_zero = true;
            }
            get_best_bracket(best_idx, finite_pts, first_infeas_above);
        }

        if (!certified_zero && curIter == 0) {
            const double z_lo = (best_idx > 0) ? finite_pts[best_idx - 1].first : finite_pts[0].first;
            double z_hi = finite_pts.back().first;
            if (best_idx + 1 < static_cast<int>(finite_pts.size())) {
                z_hi = finite_pts[best_idx + 1].first;
            } else if (std::isfinite(first_infeas_above)) {
                z_hi = first_infeas_above;
            }
            if (z_hi > z_lo) {
                vector<double> zoom_cands;
                make_uniform_grid(z_lo, z_hi, wave_width + 2, zoom_cands);
                eval_batch_parallel(zoom_cands);
                get_best_bracket(best_idx, finite_pts, first_infeas_above);
            }
        }

        // If the best point abuts an infeasible non-PD cliff, refine the feasibility boundary
        if (!certified_zero && best_idx == static_cast<int>(finite_pts.size()) - 1 && std::isfinite(first_infeas_above)) {
            double lo_feas = finite_pts[best_idx].first;
            double hi_infeas = first_infeas_above;
            for (int bi = 0; bi < 24 && (hi_infeas - lo_feas) > 1e-8 * std::max(1.0, lo_feas); ++bi) {
                const double mid = 0.5 * (lo_feas + hi_infeas);
                const double ll_mid = eval_single(mid);
                if (!std::isfinite(ll_mid)) {
                    hi_infeas = mid;
                } else if (ll_mid >= finite_pts[best_idx].second) {
                    lo_feas = mid;
                    get_best_bracket(best_idx, finite_pts, first_infeas_above);
                } else {
                    break;
                }
            }
            get_best_bracket(best_idx, finite_pts, first_infeas_above);
        }

        // 3. Safeguarded Brent parabolic interpolation polish to machine-tight tolerance
        double opt_tao = finite_pts[best_idx].first;
        double opt_logL = finite_pts[best_idx].second;

        if (certified_zero) {
            opt_tao = 0.0;
            opt_logL = logL_0;
        } else if (finite_pts.size() >= 2) {
            if (best_idx == 0 && !evaled_zero) {
                ensure_zero_eval();
                get_best_bracket(best_idx, finite_pts, first_infeas_above);
            }
            if (best_idx == 0 && score_0 <= 0.0) {
                const double tiny_t = std::min(0.1 * finite_pts[1].first, 1e-8 * std::max(1.0, tao_scale));
                const double tiny_ll = eval_single(tiny_t);
                if (tiny_ll <= logL_0) {
                    opt_tao = 0.0;
                    opt_logL = logL_0;
                } else {
                    get_best_bracket(best_idx, finite_pts, first_infeas_above);
                }
            }

            if (!(best_idx == 0 && score_0 <= 0.0 && opt_tao == 0.0)) {
                double a = (best_idx > 0) ? finite_pts[best_idx - 1].first : 0.0;
                double fa = (best_idx > 0) ? -finite_pts[best_idx - 1].second : -logL_0;
                double b = max_tao_limit;
                double fb = std::numeric_limits<double>::infinity();
                if (best_idx + 1 < static_cast<int>(finite_pts.size())) {
                    b = finite_pts[best_idx + 1].first;
                    fb = -finite_pts[best_idx + 1].second;
                } else if (std::isfinite(first_infeas_above)) {
                    b = first_infeas_above;
                    fb = std::numeric_limits<double>::infinity();
                } else {
                    b = finite_pts.back().first;
                    fb = -finite_pts.back().second;
                }

                double x = finite_pts[best_idx].first;
                double fx = -finite_pts[best_idx].second;
                if (x <= a || x >= b) {
                    x = 0.5 * (a + b);
                    const double ll_x = eval_single(x);
                    fx = std::isfinite(ll_x) ? -ll_x : std::numeric_limits<double>::infinity();
                    if (-fa < fx) { x = a + 0.3819660112501051 * (b - a); const double ll2 = eval_single(x); fx = std::isfinite(ll2) ? -ll2 : std::numeric_limits<double>::infinity(); }
                }

                double w = x, fw = fx;
                double v = x, fv = fx;
                if (std::isfinite(fa) && std::isfinite(fb) && a < x && x < b) {
                    if (fa <= fb) {
                        w = a; fw = fa;
                        v = b; fv = fb;
                    } else {
                        w = b; fw = fb;
                        v = a; fv = fa;
                    }
                }

                double d = 0.0;
                double e = b - a;
                const double cgold = 0.3819660112501051;

                for (int brent_it = 0; brent_it < 24; ++brent_it) {
                    const double xm = 0.5 * (a + b);
                    const double tol1 = 2e-7 * std::max(1.0, std::abs(x)) + 1e-9;
                    const double tol2 = 2.0 * tol1;
                    if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
                        break;
                    }

                    bool did_parabolic = false;
                    if (std::abs(e) > tol1 && std::isfinite(fw) && std::isfinite(fv) && w != x && v != x && w != v) {
                        const double r = (x - w) * (fx - fv);
                        double q = (x - v) * (fx - fw);
                        double p = (x - v) * q - (x - w) * r;
                        q = 2.0 * (q - r);
                        if (q > 0.0) p = -p;
                        q = std::abs(q);
                        const double etemp = e;
                        e = d;
                        if (std::abs(p) < std::abs(0.5 * q * etemp) && p > q * (a - x) && p < q * (b - x)) {
                            d = p / q;
                            const double u_try = x + d;
                            if ((u_try - a) < tol2 || (b - u_try) < tol2) {
                                d = (xm >= x) ? tol1 : -tol1;
                            }
                            did_parabolic = true;
                        }
                    }
                    if (!did_parabolic) {
                        e = (x >= xm) ? (a - x) : (b - x);
                        d = cgold * e;
                    }

                    const double u = (std::abs(d) >= tol1) ? (x + d) : (x + (d >= 0.0 ? tol1 : -tol1));
                    const double ll_u = eval_single(u);
                    const double fu = std::isfinite(ll_u) ? -ll_u : std::numeric_limits<double>::infinity();

                    if (fu <= fx) {
                        if (u >= x) a = x; else b = x;
                        v = w; fv = fw;
                        w = x; fw = fx;
                        x = u; fx = fu;
                    } else {
                        if (u < x) a = u; else b = u;
                        if (fu <= fw || w == x) {
                            v = w; fv = fw;
                            w = u; fw = fu;
                        } else if (fu <= fv || v == x || v == w) {
                            v = u; fv = fu;
                        }
                    }
                }

                // Compare interior Brent optimum against exact boundaries if adjacent
                opt_tao = x;
                opt_logL = -fx;
                if (a == 0.0 && logL_0 >= opt_logL) {
                    opt_tao = 0.0;
                    opt_logL = logL_0;
                }
                if (b == max_tao_limit) {
                    const double ll_end = eval_single(max_tao_limit);
                    if (std::isfinite(ll_end) && ll_end > opt_logL) {
                        opt_tao = max_tao_limit;
                        opt_logL = ll_end;
                    }
                }
            }
        }

        LOGGER << "REML 1D search evaluated " << total_points << " points, max logL: " << opt_logL
               << ", Tao: " << opt_tao << std::endl;
        if(total_infeasible > 0){
            LOGGER << "Grid search skipped " << total_infeasible << "/" << total_points
                   << " tao values because V was not positive definite. If the converged tao lies"
                   << " close to the infeasible region, consider a denser or PSD-regularised GRM." << std::endl;
        }
        const double raw_tao = opt_tao;
        const double cur_fp_res = raw_tao - pre_tao;

        // First perform the exact, unextrapolated GCTA BLUP update at raw_tao so convergence
        // is verified strictly against GCTA's exact PQL-REML fixed-point equations.
        VectorXd raw_a, raw_eta;
        if (!run_blup_step(raw_tao, var_mu_i, Y, raw_a, raw_eta)) {
            LOGGER.e(0, "can't invert the V matrix.");
        }
        const double raw_max_deta = (raw_eta - Xa_b).lpNorm<Eigen::Infinity>();

        double thresh = ((raw_a - pre_est_a).array().abs() / (raw_a.array().abs() + pre_est_a.array().abs() + tol)).maxCoeff();
        thresh = std::max(thresh, std::abs((raw_tao - pre_tao) / (raw_tao + pre_tao + tol)) );

        if (thresh < tol) {
            cur_tao = raw_tao;
            hisTao[curIter] = cur_tao;
            est_a = raw_a;
            Xa_b = raw_eta;
            updateLogisticWorkingFromEta(Xa_b, phenoVec, mu, var_mu_i, Y);
            LOGGER << "Iter " << curIter << ", tao: " << cur_tao << ", Var(Y): " << varVector(Y) << ", fixed effects: " << est_a.transpose() << "." << std::endl;
            bConverge = true;
            break;
        }

        // Safeguarded Secant / Steffensen fixed-point acceleration on r(tao) = F(tao) - tao = 0
        double next_tao = raw_tao;
        bool used_secant = false;
        if (has_prev_fp && !prev_was_secant) {
            const double d_tao = pre_tao - prev_fp_tao;
            const double d_res = cur_fp_res - prev_fp_res;
            if (std::abs(d_tao) > 1e-12 && std::abs(d_res) > 1e-14) {
                const bool oscillating = (cur_fp_res * prev_fp_res < 0.0);
                const double ratio = cur_fp_res / prev_fp_res;
                const bool monotone_slow = (cur_fp_res * prev_fp_res > 0.0 && ratio > 0.04 && ratio < 0.995);
                if (oscillating || monotone_slow) {
                    double sec_tao = pre_tao - cur_fp_res * (d_tao / d_res);
                    if (oscillating) {
                        const double lo_b = std::min(pre_tao, prev_fp_tao);
                        const double hi_b = std::max(pre_tao, prev_fp_tao);
                        sec_tao = std::max(lo_b, std::min(hi_b, sec_tao));
                    } else {
                        const double max_jump = 5.0 * std::abs(cur_fp_res);
                        sec_tao = std::max(0.0, std::min(max_tao_limit,
                                  std::max(raw_tao - max_jump, std::min(raw_tao + max_jump, sec_tao))));
                    }
                    if (std::isfinite(sec_tao) && sec_tao >= 0.0 &&
                        std::abs(sec_tao - raw_tao) > 0.25 * tol * (raw_tao + tol)) {
                        next_tao = sec_tao;
                        used_secant = true;
                    }
                }
            }
        }

        prev_fp_tao = pre_tao;
        prev_fp_res = cur_fp_res;
        has_prev_fp = true;

        bool applied_next = false;
        if (used_secant) {
            VectorXd sec_a = est_a, sec_eta = Xa_b, sec_mu = mu, sec_v = var_mu_i, sec_Y = Y;
            if (polish_pirls_at_tao(next_tao, sec_a, sec_eta, sec_mu, sec_v, sec_Y, 3)) {
                cur_tao = next_tao;
                est_a = sec_a;
                Xa_b = sec_eta;
                mu = sec_mu;
                var_mu_i = sec_v;
                Y = sec_Y;
                prev_was_secant = true;
                applied_next = true;
            }
        }

        if (!applied_next) {
            cur_tao = raw_tao;
            prev_was_secant = false;
            if (raw_max_deta > 2.5) {
                const double damp = std::max(0.35, 2.5 / raw_max_deta);
                est_a = est_a + damp * (raw_a - est_a);
                Xa_b = Xa_b + damp * (raw_eta - Xa_b);
                updateLogisticWorkingFromEta(Xa_b, phenoVec, mu, var_mu_i, Y);
                polish_pirls_at_tao(cur_tao, est_a, Xa_b, mu, var_mu_i, Y, 2);
            } else {
                est_a = raw_a;
                Xa_b = raw_eta;
                updateLogisticWorkingFromEta(Xa_b, phenoVec, mu, var_mu_i, Y);
                if (raw_max_deta > 0.04 && cur_tao > 0.0) {
                    polish_pirls_at_tao(cur_tao, est_a, Xa_b, mu, var_mu_i, Y, 2);
                }
            }
        }

        hisTao[curIter] = cur_tao;
        if(cur_tao < abTol){
            numAbTol++;
        }else{
            numAbTol = 0;
        }
        LOGGER << "Iter " << curIter << ", tao: " << cur_tao << ", Var(Y): " << varVector(Y) << ", fixed effects: " << est_a.transpose() << "." << std::endl;

        curIter++;
    }

    if(b_reml){
        out.close();
    }

    if(!bConverge){
        if(numAbTol >=5){
            LOGGER << "fastGWA-BB-REML stopped at a very small tao value." << std::endl;
        }else{
            LOGGER.w(0, "fastGWA-BB-REML did not converge, although the results are reliable in most cases.");
        }
    }else{
        LOGGER << "fastGWA-BB-REML converged." << std::endl;
    }
    // for analysis: tiled parallel XtWX and H without allocating an N*c XtW matrix
    const Eigen::Index n_rows_h = covar.rows();
    const Eigen::Index c_cols = covar.cols();
    VectorXd w_glm(n_rows_h);
    {
        const double * __restrict vm = var_mu_i.data();
        double * __restrict wg = w_glm.data();
        #pragma omp parallel for schedule(static) if(n_rows_h >= 4096)
        for (Eigen::Index i = 0; i < n_rows_h; ++i) {
            wg[i] = 1.0 / vm[i];
        }
    }
    const Eigen::Index kHTile = 1024;
    const Eigen::Index n_h_tiles = (n_rows_h + kHTile - 1) / kHTile;
    vector<MatrixXd> xtwx_partials(static_cast<size_t>(n_h_tiles));
    {
        ScopedSingleThreadBLAS blas_guard;
        #pragma omp parallel if(n_rows_h >= 4096)
        {
            ScopedSingleThreadBLAS::ThreadLocal tl_guard;
            MatrixXd scaled_blk(std::min<Eigen::Index>(kHTile, n_rows_h), c_cols);
            #pragma omp for schedule(static)
            for (Eigen::Index t = 0; t < n_h_tiles; ++t) {
                const Eigen::Index r0 = t * kHTile;
                const Eigen::Index rows = std::min<Eigen::Index>(kHTile, n_rows_h - r0);
                auto X_blk = covar.middleRows(r0, rows);
                auto S_blk = scaled_blk.topRows(rows);
                S_blk = X_blk.array().colwise() * w_glm.segment(r0, rows).array();
                xtwx_partials[static_cast<size_t>(t)].noalias() = X_blk.transpose() * S_blk;
            }
        }
    }
    MatrixXd XtWX = MatrixXd::Zero(c_cols, c_cols);
    for (Eigen::Index t = 0; t < n_h_tiles; ++t) {
        XtWX.noalias() += xtwx_partials[static_cast<size_t>(t)];
    }
    auto t_solver = XtWX.colPivHouseholderQr();
    if(!t_solver.isInvertible()){
        LOGGER.e(0, "XtWX is not invertible.");
    }
    MatrixXd inv_XtWX = t_solver.solve(MatrixXd::Identity(c_cols, c_cols));
    H.resize(c_cols, n_rows_h);
    {
        ScopedSingleThreadBLAS blas_guard;
        #pragma omp parallel if(n_rows_h >= 4096)
        {
            ScopedSingleThreadBLAS::ThreadLocal tl_guard;
            MatrixXd xtw_blk(c_cols, std::min<Eigen::Index>(kHTile, n_rows_h));
            #pragma omp for schedule(static)
            for (Eigen::Index t = 0; t < n_h_tiles; ++t) {
                const Eigen::Index r0 = t * kHTile;
                const Eigen::Index rows = std::min<Eigen::Index>(kHTile, n_rows_h - r0);
                auto X_blk = covar.middleRows(r0, rows);
                auto W_blk = xtw_blk.leftCols(rows);
                W_blk = (X_blk.array().colwise() * w_glm.segment(r0, rows).array()).matrix().transpose();
                H.middleCols(r0, rows).noalias() = inv_XtWX * W_blk;
            }
        }
    }

    // for 
    binSolverV->permuteWeights(var_mu_i, w_s, w_r_perm);
    taoVal = cur_tao;
    double logdet_final = 0.0;
    if(!binSolverV->factorize(w_s.data(), w_r_perm.data(), cur_tao, binSolverV->main_ws, logdet_final)){
        LOGGER.e(0, "can't invert the V matrix.");
    }

    if(!bGLMMExact){
        binSolverV->prepareGrammarTuning(covar, ViX);
    }
    //VectorXd PY = ViY - ViX * est_a;
    //template of P
    //VectorXd PY = solverV.solve(Y) - ViX * inv_XtVX_ViX * Y;

    return bConverge;


}

void FastFAM::initVar(){
    numi_indi = num_indi;
    phenoVecMu = phenoVec - mu;
    SPA::setMu(mu);
    dWp = mu.array() * (-mu.array() + 1);

    if(spaCutOff < 0.1){
        spaCutOff = 0.1;
    }
}

void FastFAM::initBinary(const SpMat &fam){

    num_covar = covar.cols();

    VectorXd est_beta = VectorXd::Zero(covar.cols());
    //est_beta.setZero();
    double thresh = 1e-6;
    int maxIter = 200;

    LOGGER.ts("binREML");
    binGridREML(fam, est_beta, maxIter, thresh);
    LOGGER << "fastGWA-BB-REML finished in " << LOGGER.tp("binREML") << " seconds." << std::endl;

    initVar();

    if(!bGLMMExact){
        estBinGamma();
    }
    
    if(options.find("model_only") != options.end()){
        LOGGER << "Saving fastGWA-BB model information..." << std::endl;
        string id_file = options["out"] + ".mdl.id";
        std::ofstream inv_id(id_file.c_str());
        if(!inv_id) LOGGER.e(0, "failed to write " + options["out"]+".grm.id");
        vector<string> ids = pheno->get_id(0, num_indi - 1, "\t");
        for(auto & t_id : ids){
            inv_id << t_id << "\n";
        }
        inv_id.close();
        LOGGER << "Sample information have been saved to [" << id_file << "]." << std::endl;

        char magic[5];
        magic[0] = 'f';
        magic[1] = 'G';
        magic[2] = 'L';
        magic[3] = 'M';
        magic[4] = '\0';
 
        string bin_file = options["out"] + ".mdl.bin2";
        FILE * pFile = fopen(bin_file.c_str(), "wb");
        // TODO check the output
        fwrite(magic, 5, sizeof(char), pFile);
        fwrite(&num_indi, 1, sizeof(uint32_t), pFile);
        uint32_t col_covar = covar.cols();
        fwrite(&col_covar, 1, sizeof(uint32_t), pFile);
        fwrite(&bPreciseCovar, 1, sizeof(bool), pFile);

        fwrite(&taoVal, 1, sizeof(double), pFile);
        fwrite(&c_inf, 1, sizeof(double), pFile);
        fwrite(mu.data(), mu.size(), sizeof(double), pFile);
        fwrite(phenoVec.data(), phenoVec.size(), sizeof(double), pFile);
        fwrite(covar.data(), covar.rows() * covar.cols(), sizeof(double), pFile);
        fwrite(H.data(), H.rows() * H.cols(), sizeof(double), pFile);
        fclose(pFile);
        LOGGER << "Data for model have been saved to [" << bin_file << "]." << std::endl;
    }
    ViX.resize(0,0);
    binSolverV.reset();
    //solverV.~SimplicialLDLT<SpMat>();
}

void FastFAM::binGrammar_func(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    int nMarker = markerIndex.size();
    ScopedSingleThreadBLAS blas_guard;
    #pragma omp parallel
    {
        ScopedSingleThreadBLAS::ThreadLocal tl_guard;
        GenoBufItem item;
        item.geno.resize(num_indi);
        VectorXd u_r_buf;
        VectorXd w_buf(covar.cols());
        #pragma omp for schedule(dynamic)
        for(int i = 0; i < nMarker; i++){
            int index_cur_marker = num_grammar_markers + i;
            item.extractedMarkerIndex = markerIndex[i];
            item.valid = false;
            geno->getGenoDouble(genobuf, i, &item);
            bValids[index_cur_marker] = item.valid;
            if(item.valid){
                Map< VectorXd > curGeno(item.geno.data(), num_indi);
                if(bPreciseCovar) conditionCovarBinReg(curGeno);
                v_c_infs[index_cur_marker] = binSolverV->evalGrammarGamma(curGeno, dWp, ViX, u_r_buf, w_buf);
            }
        }
    }
    num_grammar_markers += nMarker;
}


void FastFAM::estBinGamma(){
    int num_marker_rand = 2000;
    LOGGER.i(0, "\nTuning parameters using null SNPs...");  
    auto total_markers_index = marker->get_extract_index_autosome();
    if(total_markers_index.size() < num_marker_rand){
        LOGGER.e(0, "can't read " + to_string(num_marker_rand) + " SNPs from the autosomes for tuning.");
    }
    std::random_device rd;
    std::mt19937 gen(seed);
    std::shuffle(total_markers_index.begin(), total_markers_index.end(), gen);
    
   //get previous threshold
    double preAF = geno->getMAF();
    double preInfo = geno->getFilterInfo();
    double preMiss = geno->getFilterMiss();
    if(options.find("c-inf-no-filter") != options.end()){
        double mac20 = 10.0 / num_indi;
        if(preAF < mac20){
            // mac 20
            geno->setMAF(mac20);
        }
        if(preInfo < 0.3 && hasInfo){
            geno->setFilterInfo(0.3);
        }
        if(preMiss < 0.9){
            geno->setFilterMiss(0.9);
        }
    }

    num_grammar_markers = 0;

    int nModelSNP = 200;
    int nMarker = nModelSNP;
    int curNumModel = 0;

    int max_omp_threads = 1;
#ifdef _OPENMP
    max_omp_threads = std::max(1, omp_get_max_threads());
#endif
    const size_t max_stage_bytes = 128ULL * 1024ULL * 1024ULL;
    const size_t bytes_per_snp = static_cast<size_t>(std::max<uint32_t>(1, num_indi)) * sizeof(double);
    const int stage_cap = std::min<int>(
        nModelSNP,
        std::max<int>(16, std::min<int>(max_omp_threads * 2, static_cast<int>(max_stage_bytes / bytes_per_snp))));
    MatrixXd staged_genos;
    vector<int> staged_slots;
    staged_slots.reserve(stage_cap);

    auto flush_staged = [&]() {
        const int n_staged = static_cast<int>(staged_slots.size());
        if (n_staged == 0) return;
        ScopedSingleThreadBLAS blas_guard;
        #pragma omp parallel
        {
            ScopedSingleThreadBLAS::ThreadLocal tl_guard;
            VectorXd u_r_buf;
            VectorXd w_buf(covar.cols());
            #pragma omp for schedule(dynamic)
            for (int k = 0; k < n_staged; ++k) {
                const int out_slot = staged_slots[k];
                Eigen::Ref<VectorXd> curGeno = staged_genos.col(k);
                if (bPreciseCovar) conditionCovarBinReg(curGeno);
                v_c_infs[out_slot] = binSolverV->evalGrammarGamma(curGeno, dWp, ViX, u_r_buf, w_buf);
            }
        }
        staged_slots.clear();
    };

    vector<function<void (uintptr_t *, const vector<uint32_t> &)>> callBacks;
    callBacks.push_back([&](uintptr_t *genobuf, const vector<uint32_t> &mIndex) {
        const int curBatch = static_cast<int>(mIndex.size());
        if (curBatch * 2 >= max_omp_threads && staged_slots.empty()) {
            this->binGrammar_func(genobuf, mIndex);
            return;
        }
        if (staged_genos.cols() < stage_cap) {
            staged_genos.resize(num_indi, stage_cap);
        }
        int offset = 0;
        while (offset < curBatch) {
            const int space = stage_cap - static_cast<int>(staged_slots.size());
            const int take = std::min(space, curBatch - offset);
            const int base_stage = static_cast<int>(staged_slots.size());
            vector<uint8_t> local_valid(take, 0);
            {
                ScopedSingleThreadBLAS blas_guard;
                #pragma omp parallel if(take > 1)
                {
                    ScopedSingleThreadBLAS::ThreadLocal tl_guard;
                    GenoBufItem item;
                    item.geno.resize(num_indi);
                    #pragma omp for schedule(dynamic)
                    for (int t = 0; t < take; ++t) {
                        const int src_i = offset + t;
                        const int index_cur_marker = num_grammar_markers + src_i;
                        item.extractedMarkerIndex = mIndex[src_i];
                        item.valid = false;
                        geno->getGenoDouble(genobuf, src_i, &item);
                        bValids[index_cur_marker] = item.valid;
                        local_valid[t] = item.valid ? 1 : 0;
                        if (item.valid) {
                            std::memcpy(staged_genos.col(base_stage + t).data(),
                                        item.geno.data(),
                                        static_cast<size_t>(num_indi) * sizeof(double));
                        }
                    }
                }
            }
            int write_pos = base_stage;
            for (int t = 0; t < take; ++t) {
                if (local_valid[t]) {
                    if (write_pos != base_stage + t) {
                        staged_genos.col(write_pos) = staged_genos.col(base_stage + t);
                    }
                    staged_slots.push_back(num_grammar_markers + offset + t);
                    write_pos++;
                }
            }
            offset += take;
            if (static_cast<int>(staged_slots.size()) >= std::min(stage_cap, std::max(max_omp_threads * 2, 32))) {
                flush_staged();
            }
        }
        num_grammar_markers += curBatch;
    });
    double cvThreshold = options_d["cv_threshold"];

    LOGGER.ts("tuning");
    while(true){
        curNumModel += nModelSNP;
        if(curNumModel > num_marker_rand){
            LOGGER.e(0, "failed to estimate gamma. There are too many signals. You may use --cv-threshold to lower down the threshold.");
        }

        LOGGER << "  reading " << nModelSNP << " SNP..." << std::endl; 
        v_c_infs.resize(curNumModel);
        bValids.resize(curNumModel);

        vector<uint32_t> marker_index(total_markers_index.begin() + curNumModel - nModelSNP, total_markers_index.begin() + curNumModel);
        std::sort(marker_index.begin(), marker_index.end());
        geno->loopDouble(marker_index, nMarker, true, true, false, false, callBacks);
        flush_staged();

        if(curNumModel != num_grammar_markers){
            LOGGER.e(0, "some SNPs were not read successfully!");
        }

        double tmp_cinf = 0;
        double temp_cinf2 = 0;
        int n_valid_null = 0;

        for(int i=0; i < curNumModel; i++){
            if(bValids[i]){
                double curInf = v_c_infs[i];
                tmp_cinf += curInf;
                temp_cinf2 += curInf * curInf;
                n_valid_null++;
            }
        }
        if(n_valid_null < 100){
            LOGGER << "Not enough null SNPs." << std::endl;
            continue;
        }

        c_inf = tmp_cinf / n_valid_null;

        double sumsq = temp_cinf2 - c_inf * c_inf * n_valid_null;
        if(sumsq < 0 && sumsq > -1e-8){
            sumsq = 0;
        }

        double sd_c_inf = std::sqrt(sumsq / (n_valid_null - 1));
        double cv = sd_c_inf / c_inf;
        if(cv < cvThreshold){
            break;
        }else{
            LOGGER << "  GRAMMAR-Gamma value = " << c_inf << ", CV = " << cv << ", rerun " << std::endl;
        }

    }

    //reset back
    geno->setMAF(preAF);
    geno->setFilterInfo(preInfo);
    geno->setFilterMiss(preMiss);


    LOGGER.i(0, "Mean GRAMMAR-Gamma value = " + to_string(c_inf));
    LOGGER << "Tuning time: " << LOGGER.tp("tuning") << " seconds." << std::endl;
    std::ofstream o_inf;
    bool out_inf = false;
    if(options.find("c-inf") != options.end()){
        o_inf.open(options["out"] + ".cinf");
        o_inf << "CHR\tSNP\tPOS\tA1\tA2\tcinf" << std::endl;
        out_inf = true;
    }
    for(int i=0; i<curNumModel; i++){
        if(bValids[i]){
            if(out_inf){
                //o_inf << marker->getMarkerStrExtract(marker_index[i]) << "\t" << v_c_infs[i] << std::endl;
            }
        }
    }
    if(out_inf){
        o_inf.close();
    }
    //LOGGER.i(0, "Got " + to_string(n_valid_null) + " null SNPs");
    v_c_infs.resize(0);
    bValids.resize(0);
}

void FastFAM::calculate_spa(uintptr_t *genobuf, const vector<uint32_t> &markerIndex){
    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;
        geno->getGenoDouble(genobuf, i, &item);
        
        isValids[i] = item.valid;
        if(!item.valid){
            continue;
        }

        Map< VectorXd > xvec(item.geno.data(), num_indi);
        VectorXd xvec2 = xvec;

        if(bPreciseCovar) conditionCovarBinReg(xvec);

        double varSNP = std::sqrt(xvec.dot(dWp.cwiseProduct(xvec)) * c_inf);
 
        SPARes res;
        res.score = xvec.dot(phenoVecMu);
        double chisq = std::abs(res.score) / varSNP;

        res.p = StatLib::pchisqd1(chisq * chisq);

        res.bConverge = true;
        if( chisq < spaCutOff){
            res.p_adj = res.p;
        }else{
            vector<uint32_t> index0;
            index0.reserve(num_indi);
            double thresh = -item.mean + 1e-6;
            //double thresh = 1e-6;
            for(uint32_t i = 0; i < num_indi; i++){
                if(xvec2[i] > thresh){
                    index0.push_back(i);
                }
            }

            if(!bPreciseCovar) conditionCovarBinReg(xvec);

            double q = xvec.dot(phenoVec);
            double qinv = q - res.score - res.score;
            SPA spa(q, qinv, xvec, index0);
            spa.saddleProb(&res);
        }

        Tscore[i] = (float)res.score; //* geno->RDev[cur_raw_marker]; 
        Tse[i] = (float)varSNP;
        p[i] = res.p; 
        padj[i] = res.p_adj;
        rConverge[i] = res.bConverge;
        af[i] = (float)item.af;
        countMarkers[i] = item.nValidN;
        info[i] = item.info;
        double temp_beta = res.score / (varSNP *varSNP);
        beta[i] = (float) temp_beta;
        se[i] = std::abs(temp_beta) / sqrt(StatLib::qchisqd1(res.p_adj));
    }

    output_res_spa(isValids, markerIndex);

}
