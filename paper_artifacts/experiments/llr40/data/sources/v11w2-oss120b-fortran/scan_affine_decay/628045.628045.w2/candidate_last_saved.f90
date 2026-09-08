module scan_affine_decay_mod
  use iso_c_binding
contains
  subroutine scan_affine_decay_fp64(y, c, x, len_1d) bind(C, name="scan_affine_decay_fp64")
    real(c_double), intent(out) :: y(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: x(*)
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i
    real(c_double) :: tmp

    if (len_1d <= 0) return
    tmp = x(1)
    y(1) = tmp
    !$omp parallel
    !$omp single
    do i = 2_c_int64_t, len_1d
      tmp = c(i) * tmp + x(i)
      y(i) = tmp
    end do
    !$omp end single
    !$omp end parallel
  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
