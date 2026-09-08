module tsvc_2_s119_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s119_fp64")
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(in) :: bb
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx_ij, idx_im1j

    do i = 1_c_int64_t, LEN_2D - 1_c_int64_t
!$omp simd
      do j = 1_c_int64_t, LEN_2D - 1_c_int64_t
        idx_ij = i * LEN_2D + j
        idx_im1j = (i - 1_c_int64_t) * LEN_2D + (j - 1_c_int64_t)
        aa(idx_ij + 1_c_int64_t) = aa(idx_im1j + 1_c_int64_t) + bb(idx_ij + 1_c_int64_t)
      end do
    end do

  end subroutine tsvc_2_s119_fp64
end module tsvc_2_s119_mod
