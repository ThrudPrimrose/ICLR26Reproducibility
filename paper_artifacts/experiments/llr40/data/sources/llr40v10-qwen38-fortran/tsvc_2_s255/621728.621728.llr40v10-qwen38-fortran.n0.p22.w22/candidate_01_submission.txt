subroutine tsvc_2_s255_fp64(a, b, len_1d) bind(c, name="tsvc_2_s255_fp64")
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(out) :: a
  real(c_double), dimension(*), intent(in)  :: b
  integer(c_int64_t), value :: len_1d

  integer(kind=8) :: i, n

  n = len_1d
  if (n <= 0) return
  if (n == 1) then
    ! numpy oracle: b[-1] wraps to b(1)
    a(1) = (b(1) + b(1) + b(1)) * 0.333d0
    return
  end if
  a(1) = (b(1) + b(n) + b(n-1)) * 0.333d0
  a(2) = (b(2) + b(1) + b(n)) * 0.333d0
  !$omp parallel do schedule(static)
  do i = 3, n
    a(i) = (b(i) + b(i-1) + b(i-2)) * 0.333d0
  end do
end subroutine tsvc_2_s255_fp64
