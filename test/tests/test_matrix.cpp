#include <gtest/gtest.h>
#include "Matrix.hpp"
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;

namespace {
MatrixXd MakeWellConditionedSpd(int n, unsigned seed = 1337) {
    MatrixXd M(n, n);
    unsigned state = seed;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            M(i, j) = (static_cast<double>(state & 0xFFFFu) / 32768.0) - 1.0;
        }
    }
    return M * M.transpose() + 2.5 * MatrixXd::Identity(n, n);
}
} // namespace

TEST(MatrixInverseTest, SpdInverseMatchesIdentityAndLogDetAcrossMethods) {
    const int n = 32;
    const MatrixXd A0 = MakeWellConditionedSpd(n);
    const double expected_logdet = 2.0 * A0.llt().matrixL().toDenseMatrix().diagonal().array().log().sum();
    const MatrixXd I = MatrixXd::Identity(n, n);

    for (INVmethod req : {INV_LLT, INV_LU, INV_QR, INV_FQR}) {
        MatrixXd A_inv = A0;
        double logdet = 0.0;
        int rank = n;
        INVmethod used = req;
        ASSERT_TRUE(SquareMatrixInverse(A_inv, logdet, rank, used));
        EXPECT_NEAR(logdet, expected_logdet, 1e-8);
        const double max_err = (A0 * A_inv - I).cwiseAbs().maxCoeff();
        EXPECT_LT(max_err, 1e-9);
    }
}

TEST(MatrixInverseTest, IndefiniteSymmetricFallsBackFromLltCleanly) {
    MatrixXd A(4, 4);
    A <<  2.0,  0.5,  0.0,  0.1,
          0.5, -3.0,  0.2,  0.0,
          0.0,  0.2,  4.0, -0.3,
          0.1,  0.0, -0.3,  1.5;

    MatrixXd A_inv = A;
    double logdet = 0.0;
    int rank = 0;
    INVmethod method = INV_LLT;
    ASSERT_TRUE(SquareMatrixInverse(A_inv, logdet, rank, method));
    EXPECT_EQ(method, INV_LU);
    EXPECT_NEAR(logdet, std::log(std::abs(A.determinant())), 1e-10);
    const double max_err = (A * A_inv - MatrixXd::Identity(4, 4)).cwiseAbs().maxCoeff();
    EXPECT_LT(max_err, 1e-10);
}

TEST(MatrixInverseTest, RankDeficientSingularMatrixFqrDetectsRankAndRejectsInversion) {
    MatrixXd B(2, 2);
    B << 1.0, 1.0,
         1.0, 1.0;

    MatrixXd B_work = B;
    double logdet = 0.0;
    int rank = 0;
    INVmethod method = INV_LLT;
    EXPECT_FALSE(SquareMatrixInverse(B_work, logdet, rank, method));
    EXPECT_EQ(rank, 1);

    // Rank-3 6x6 matrix (rank/n = 0.5 < 0.99 -> FQR reports exact rank 3 and returns false)
    MatrixXd U(6, 3);
    U << 1, 0, 1,
         0, 1, 1,
         1, 1, 0,
         2, -1, 1,
        -1, 2, 0,
         0, 1, -1;
    MatrixXd R3 = U * U.transpose();
    int rank3 = 0;
    INVmethod m2 = INV_FQR;
    EXPECT_FALSE(SquareMatrixInverse(R3, logdet, rank3, m2));
    EXPECT_EQ(rank3, 3);
}
