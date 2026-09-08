subroutine tsvc_2_s115_fp64(a, aa, len_2d) bind(c, name='tsvc_2_s115_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a, aa
  integer(c_int64_t), value :: len_2d
  real(c_double), pointer :: ap(:)
  real(c_double), pointer :: qp(:)
  integer(c_int64_t) :: n, i, j
  real(c_double) :: bj
  n = len_2d
  call c_f_pointer(a, ap, [n])
  call c_f_pointer(aa, qp, [n*n])
  do j = 0, n - 1
     bj = ap(j+1)
     do i = j + 1, n - 1
        ap(i+1) = ap(i+1) - bj * qp(j*n + i + 1)
     end do
  end do
end subroutine tsvc_2_s115_fp64
