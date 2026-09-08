! Fortran implementation of TSVC 2 kernel s231 (fp64)
module tsvc_2_s231_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s231_fp64")
    ! Arguments: aa is input-output array, bb is input, LEN_2D is dimension size (int64)
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: bb(*)
    integer(c_int64_t), value    :: LEN_2D
    integer(c_int64_t) :: i, j
    ! Parallelize outer loop over columns (i). Each column is independent.
    !$omp parallel do default(none) shared(aa, bb, LEN_2D) private(i, j)
    do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      do j = 1_c_int64_t, LEN_2D - 1_c_int64_t
        aa(j*LEN_2D + i + 1_c_int64_t) = aa((j-1_c_int64_t)*LEN_2D + i + 1_c_int64_t) + bb(j*LEN_2D + i + 1_c_int64_t)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s231_fp64
end module tsvc_2_s231_mod
