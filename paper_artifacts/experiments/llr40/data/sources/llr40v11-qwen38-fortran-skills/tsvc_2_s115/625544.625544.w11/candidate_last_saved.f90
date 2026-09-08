subroutine tsvc_2_s115_fp64(a, aa, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(in) :: aa(len_2d, len_2d)
  integer(c_int64_t) :: i, j, n, jj
  real(c_double) :: aj

  n = len_2d
  if (n < 2048) then
    do j = 1, n
      aj = a(j)
      do i = j + 1, n
        a(i) = a(i) - aa(i, j) * aj
      end do
    end do
    return
  end if

  jj = 1
  !$omp parallel shared(n, jj, a, aa) private(i, aj)
    do while (jj <= n)
      aj = a(jj)
      !$omp do schedule(static)
      do i = jj + 1, n
        a(i) = a(i) - aa(i, jj) * aj
      end do
      !$omp master
        jj = jj + 1
      !$omp end master
    end do
  !$omp end parallel
end subroutine tsvc_2_s115_fp64
