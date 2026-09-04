module tsvc_2_s119_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s119_fp64")
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(in) :: bb
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx_ij, idx_im1j
    do i = 1, LEN_2D - 1
      !$omp simd
      do j = 1, LEN_2D - 1
        idx_ij = i * LEN_2D + j
        idx_im1j = (i - 1) * LEN_2D + (j - 1)
        aa(idx_ij + 1) = aa(idx_im1j + 1) + bb(idx_ij + 1)
      end do
    end do
  end subroutine tsvc_2_s119_fp64
end module tsvc_2_s119_mod
