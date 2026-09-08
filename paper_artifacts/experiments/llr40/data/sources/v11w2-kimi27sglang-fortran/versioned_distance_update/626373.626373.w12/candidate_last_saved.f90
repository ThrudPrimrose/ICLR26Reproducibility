module kernel_mod
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
contains
  subroutine versioned_distance_update_fp64(a, b, c, LEN_1D, K, workspace, workspace_bytes) bind(c)
    integer(c_int64_t), value :: LEN_1D, K, workspace_bytes
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
    integer(c_int8_t), intent(inout) :: workspace(*)
    integer(c_int64_t) :: i
    do i = K + 1, LEN_1D
      a(i) = 0.75_c_double * a(i - K) + b(i) * c(i)
    end do
  end subroutine versioned_distance_update_fp64
end module kernel_mod
