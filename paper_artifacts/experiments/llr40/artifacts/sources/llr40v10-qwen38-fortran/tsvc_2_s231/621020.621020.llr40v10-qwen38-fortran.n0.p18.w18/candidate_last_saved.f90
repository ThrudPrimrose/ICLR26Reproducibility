subroutine tsvc_2_s231_fp64(aa, bb, len_2d) bind(c, name="tsvc_2_s231_fp64")
  use iso_c_binding
  implicit none
  double precision, intent(inout), dimension(*), target :: aa
  double precision, intent(in),    dimension(*), target :: bb
  integer(c_int64_t), value :: len_2d

  real(kind=8), dimension(:,:), pointer :: a, b
  integer(c_int64_t) :: n, i, j, c, nch, i0, i1
  integer, parameter :: CW = 128

  n = len_2d
  if (n < 2) return
  call c_f_pointer(c_loc(aa(1)), a, [n, n])
  call c_f_pointer(c_loc(bb(1)), b, [n, n])

  nch = n / CW
  !$omp parallel do schedule(static)
  do c = 1, nch
    i0 = (c - 1) * CW + 1
    i1 = c * CW
    do j = 2, n
      do i = i0, i1
        a(i, j) = a(i, j-1) + b(i, j)
      end do
    end do
  end do
  ! remainder columns (n - nch*CW < CW)
  do j = 2, n
    do i = nch*CW + 1, n
      a(i, j) = a(i, j-1) + b(i, j)
    end do
  end do
end subroutine tsvc_2_s231_fp64
