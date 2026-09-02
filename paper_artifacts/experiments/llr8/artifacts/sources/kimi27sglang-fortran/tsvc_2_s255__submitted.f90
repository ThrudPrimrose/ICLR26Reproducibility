module s255_m
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
contains
  subroutine tsvc_2_s255_fp64(a, b, LEN_1D, workspace, workspace_size) &
       bind(C, name="tsvc_2_s255_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int8_t), intent(inout) :: workspace(*)
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: n, i

    n = LEN_1D
    if (n <= 0) return

    ! Scalar expansion of the two carry-around scalars.
    ! Reference uses x = b[n-1], y = b[n-2] at i=0, then rotates.
    ! In 1-based Fortran this is the 3-point wrapped stencil:
    !   a(i) = (b(i) + b(i-1) + b(i-2)) * 0.333
    ! with b(0) == b(n) and b(-1) == b(n-1).

    a(1) = (b(1) + b(n) + b(n - 1)) * 0.333_c_double
    if (n < 2) return
    a(2) = (b(2) + b(1) + b(n)) * 0.333_c_double
    if (n < 3) return

    !$omp parallel do proc_bind(spread) if(n > 100000_c_int64_t) schedule(static)
    do i = 3, n
       a(i) = (b(i) + b(i - 1) + b(i - 2)) * 0.333_c_double
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s255_fp64
end module s255_m
