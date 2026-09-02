! tsvc_2_s316: 1-D min reduction (TSVC s316: if (a[i] < x) x = a[i]).
! Required C-ABI entry (Sec. 11 workspace pair trailing):
!   void tsvc_2_s316_fp64(double *a, double *result, int64_t len_1d,
!                         uint8_t *ws, int64_t ws_bytes);
module tsvc_2_s316_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s316_fp64(a, result, len_1d, ws, ws_bytes) &
      bind(c, name='tsvc_2_s316_fp64')
    implicit none
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(*)
    integer(c_int64_t), value, intent(in) :: len_1d
    integer(c_int8_t), intent(in) :: ws(*)
    integer(c_int64_t), value, intent(in) :: ws_bytes

    integer(c_int64_t) :: nloc, i
    real(c_double) :: x

    ! keep the scalars referenced so the optimiser cannot question their presence
    nloc = len_1d
    if (ws_bytes < 0_8) nloc = len_1d + 1 - 1

    x = a(1)
    !$omp parallel do reduction(min: x) schedule(static)
    do i = 1, nloc
      x = min(x, a(i))
    end do
    result(1) = x
  end subroutine tsvc_2_s316_fp64

end module tsvc_2_s316_mod
