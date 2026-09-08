module tsvc_2_s2233_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s2233_fp64")
    ! Arguments
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout), dimension(*), target :: aa
    real(c_double), intent(inout), dimension(*), target :: bb
    real(c_double), intent(in),    dimension(*), target :: cc
    ! Locals
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx_current, idx_prev, idx_cc
    !DIR$ ASSUME_ALIGNED aa:64, bb:64, cc:64
    ! Compute aa: outer loop over j (sequential), inner loop over i (vectorized)
    do j = 8_c_int64_t, LEN_2D - 1_c_int64_t
      !$omp simd
      do i = 8_c_int64_t, LEN_2D - 1_c_int64_t
        idx_current = j * LEN_2D + i + 1_c_int64_t
        idx_prev    = (j - 1_c_int64_t) * LEN_2D + i + 1_c_int64_t
        idx_cc      = j * LEN_2D + i + 1_c_int64_t
        aa(idx_current) = aa(idx_prev) + cc(idx_cc)
      end do
    end do

    ! Compute bb: each column (j) independent; parallelize over columns for large sizes
    !$omp parallel do schedule(static) private(i, idx_current, idx_prev, idx_cc) if (LEN_2D > 2048)
    do j = 8_c_int64_t, LEN_2D - 1_c_int64_t
      do i = 8_c_int64_t, LEN_2D - 1_c_int64_t
        idx_current = i * LEN_2D + j + 1_c_int64_t
        idx_prev    = (i - 1_c_int64_t) * LEN_2D + j + 1_c_int64_t
        idx_cc      = i * LEN_2D + j + 1_c_int64_t
        bb(idx_current) = bb(idx_prev) + cc(idx_cc)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s2233_fp64
end module tsvc_2_s2233_mod
