module tsvc_2_s235_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name="tsvc_2_s235_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: b(*), bb(*), c(*)
    integer(c_int64_t), value    :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx, idx_prev
    real(c_double) :: a_i
    !$omp parallel do private(i, j, idx, idx_prev, a_i) schedule(static)
    do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      a_i = a(i+1) + b(i+1) * c(i+1)
      a(i+1) = a_i
      idx_prev = i
      idx = i + LEN_2D
      do j = 1_c_int64_t, LEN_2D - 1_c_int64_t
        aa(idx + 1) = aa(idx_prev + 1) + bb(idx + 1) * a_i
        idx_prev = idx
        idx = idx + LEN_2D
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s235_fp64
end module tsvc_2_s235_mod
