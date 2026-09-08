subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(1)
  double precision :: acc, acc1, acc2
  integer(c_int64_t) :: j, k, cnt

  if (len_1d < 65536) then
    acc = 0.0d0
    do j = 2, len_1d, 2
      acc = acc + a(j)
    end do
  else
    cnt = len_1d / 2
    acc1 = 0.0d0
    acc2 = 0.0d0
    !$omp parallel do simd reduction(+:acc1, acc2)
    do k = 3, cnt, 2
      acc1 = acc1 + a(2*k)
      acc2 = acc2 + a(2*k+2)
    end do
    if (len_1d >= 4) acc2 = acc2 + a(4)
    if (len_1d >= 2) acc1 = acc1 + a(2)
    acc = acc1 + acc2
  end if
  out(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
