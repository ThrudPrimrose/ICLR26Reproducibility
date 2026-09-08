subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(c, name="tsvc_2_s1232_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: len_2d, vlen
  real(c_double), intent(inout), dimension(*) :: aa
  real(c_double), intent(in), dimension(*) :: bb
  real(c_double), intent(in), dimension(*) :: cc

  integer :: n, v, q, jp, k

  n = int(len_2d)
  v = int(vlen)

  !$omp parallel do schedule(dynamic, 4)
  do q = 1, n
     if (v > 0) then
        jp = min((q - 1) / v + 1, n)
     else
        jp = n
     end if
     do k = (q - 1) * n + 1, (q - 1) * n + jp
        aa(k) = bb(k) + cc(k)
     end do
  end do
end subroutine
