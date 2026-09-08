subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(C, name="tsvc_2_s2275_fp64")
  use iso_c_binding, only : c_double, c_int64_t
  real(c_double), intent(out)   :: a(*)
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: b(*), bb(*), c(*), cc(*), d(*)
  integer(c_int64_t), value :: LEN_2D
  integer(c_int64_t) :: n, n2, k

  n = LEN_2D
  n2 = n * n

  ! Over all outer i, the inner indices {j*N + i} cover every element of the
  ! N-by-N arrays exactly once (each element is updated exactly once, so the
  ! column-strided walk is equivalent to one contiguous elementwise pass).
  !$omp parallel do schedule(static)
  do k = 1, n2
    aa(k) = aa(k) + bb(k) * cc(k)
  end do

  !$omp parallel do schedule(static)
  do k = 1, n
    a(k) = b(k) + c(k) * d(k)
  end do
end subroutine tsvc_2_s2275_fp64
