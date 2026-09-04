module tsvc_2_s231_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s231_fp64")
    ! Arguments: aa - input/output array, bb - input array, LEN_2D - matrix dimension
    real(C_DOUBLE), intent(inout) :: aa(*)
    real(C_DOUBLE), intent(in)    :: bb(*)
    integer(C_INT64_T), value    :: LEN_2D
    integer(C_INT64_T) :: i, j, idx, idx_prev
    if (LEN_2D <= 0) return
    !$omp parallel do default(none) shared(aa, bb, LEN_2D) private(i, j, idx, idx_prev)
    do i = 0, LEN_2D - 1
      do j = 1, LEN_2D - 1
        idx = j * LEN_2D + i + 1
        idx_prev = (j - 1) * LEN_2D + i + 1
        aa(idx) = aa(idx_prev) + bb(idx)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s231_fp64
end module tsvc_2_s231_mod
