subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a
  integer(c_int64_t), value :: LEN_2D
  real(c_double), pointer :: p(:)
  integer(c_int64_t) :: i, j, n, base, pbase

  n = LEN_2D
  call c_f_pointer(a, p, [n * n])
  do i = 2, n
    base = (i - 1) * n
    pbase = (i - 2) * n
    do j = 0, n - 2
      p(base + j + 1) = p(base + j + 1) + p(pbase + j + 1) + p(pbase + j + 2)
    end do
  end do
end subroutine wf_diff_skew_fp64
