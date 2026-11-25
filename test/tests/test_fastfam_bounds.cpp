#include <gtest/gtest.h>
#include "FastFAM.h"
#include "Covar.h"
#include "Geno.h"
#include "Marker.h"
#include "Pheno.h"
#include "Logger.h"
#include "test_helpers.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <omp.h>
#include <random>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern int test_argc;
extern char** test_argv;

// Regression test for upstream out-of-bounds read in FastFAM::estBinGamma:
// When GRAMMAR-Gamma tuning converges on the first 200 null SNPs (curNumModel == 200),
// bValids has size 200, yet master looped i from 0 to num_marker_rand - 1 (1999) and
// indexed bValids[i], reading 1,800 bytes past the end of the vector.
// Compiled with standard-library vector bounds assertions enabled, this test catches
// the OOB access on master and passes once the loop bound is fixed to curNumModel.
TEST(FastFamBoundsRegressionTest, EstBinGammaDoesNotReadPastEndOfBValidsVector) {
    const char* child_prefix = std::getenv("GCTA_OOB_CHILD_PREFIX");
    if (child_prefix != nullptr) {
        const std::string prefix(child_prefix);
        const std::string out_pref = prefix + "_mdl";
        omp_set_num_threads(1);
        LOGGER.open(out_pref + ".log");
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bfile", {prefix}},
            {"--grm-sparse", {prefix}},
            {"--pheno", {prefix + ".phen"}},
            {"--fastGWA-mlm-binary", {}},
            {"--model-only", {}},
            {"--seed", {"12345"}},
            {"out", {out_pref}}
        };
        Pheno::registerOption(opts);
        Marker::registerOption(opts);
        Geno::registerOption(opts);
        Covar::registerOption(opts);
        FastFAM::registerOption(opts);
        FastFAM::processMain();
        LOGGER.close();
        return;
    }

    gcta_test::TempDir tmp("gcta_oob_");
    const int N = 120, M = 2050;
    const std::string prefix = tmp.file("oob");

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<gcta_test::SampleSpec> samples(N);
    for (int i = 0; i < N; ++i) {
        samples[i] = {"F" + std::to_string(i + 1), "I" + std::to_string(i + 1), (i % 2) + 1, -9.0};
    }

    {
        std::ofstream fid(prefix + ".grm.id"), fsp(prefix + ".grm.sp"), fp(prefix + ".phen");
        for (int i = 0; i < N; ++i) {
            fid << samples[i].fid << "\t" << samples[i].iid << "\n";
            fsp << i << "\t" << i << "\t1.000000\n";
            // Concordant pairs for 0..39 (Vg > 0) and ~35% prevalence so intercept != 0
            const int y = (i < 40) ? (((i / 2) % 3 == 0) ? 1 : 0) : ((i % 3 == 0) ? 1 : 0);
            fp << samples[i].fid << "\t" << samples[i].iid << "\t" << y << "\n";
        }
        for (int p = 0; p < 20; ++p) {
            fsp << (2 * p + 1) << "\t" << (2 * p) << "\t0.500000\n";
        }
    }

    std::vector<gcta_test::SnpSpec> snps(M);
    std::vector<std::vector<int>> geno(M, std::vector<int>(N, 0));
    for (int m = 0; m < M; ++m) {
        snps[m] = {"1", "rs" + std::to_string(m + 1), static_cast<uint32_t>((m + 1) * 100), "A", "G"};
        for (int i = 0; i < N; ++i) {
            geno[m][i] = (u01(rng) < 0.35 ? 1 : 0) + (u01(rng) < 0.35 ? 1 : 0);
        }
        geno[m][0] = 0;
        geno[m][1] = 2;
    }
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);

    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        setenv("GCTA_OOB_CHILD_PREFIX", prefix.c_str(), 1);
        setenv("OMP_NUM_THREADS", "1", 1);
        setenv("OPENBLAS_NUM_THREADS", "1", 1);
        std::vector<char*> argv_exec = {test_argv[0], nullptr};
        execv(test_argv[0], argv_exec.data());
        _exit(127);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : (128 + WTERMSIG(status));
    EXPECT_EQ(rc, 0) << "Child terminated with status " << rc << " (OOB vector access aborts child)";
}
