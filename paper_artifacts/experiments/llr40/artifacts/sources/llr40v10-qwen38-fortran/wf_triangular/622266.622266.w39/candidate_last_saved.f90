subroutine wf_triangular_fp64(a, len_2d) bind(C, name="wf_triangular_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  real(c_double), intent(inout) :: a(*)
  integer(c_int64_t), intent(in) :: len_2d
  integer(c_int64_t) :: i, j, n, base, pbase
  print '(a,i0)', 'LEN_2D=', len_2d
  flush(6)
  print '(a,f0.3)', 'FIRST=', a(1)
  flush(6)
  n = len_2d
  do i = 1, n - 1
    base = i * n
    pbase = base - n
    do j = i, n - 1
      a(base + j + 1) = a(base + j + 1) + a(pbase + j + 1) + a(base + j)
    end do
  end do
  print '(a,i0)', 'DONE'
  flush(6)
end subroutine
