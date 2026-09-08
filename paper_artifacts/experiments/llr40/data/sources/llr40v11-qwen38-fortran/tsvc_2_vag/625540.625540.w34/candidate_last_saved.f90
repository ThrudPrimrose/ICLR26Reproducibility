subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), dimension(*), intent(out) :: a
  real(c_double), dimension(*), intent(in)  :: b
  integer(c_int32_t), dimension(*), intent(in) :: ip
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t) :: i, m32
  real(c_double) :: t0, t1
  integer :: nt
  nt = -1
  t0 = omp_get_wtime()
  m32 = (len_1d / 32) * 32
!$omp parallel do schedule(static) private(i)
  do i = 1, m32, 32
    if (i == 1) nt = omp_get_num_threads()
    a(i) = b(ip(i))
    a(i+1) = b(ip(i+1))
    a(i+2) = b(ip(i+2))
    a(i+3) = b(ip(i+3))
    a(i+4) = b(ip(i+4))
    a(i+5) = b(ip(i+5))
    a(i+6) = b(ip(i+6))
    a(i+7) = b(ip(i+7))
    a(i+8) = b(ip(i+8))
    a(i+9) = b(ip(i+9))
    a(i+10) = b(ip(i+10))
    a(i+11) = b(ip(i+11))
    a(i+12) = b(ip(i+12))
    a(i+13) = b(ip(i+13))
    a(i+14) = b(ip(i+14))
    a(i+15) = b(ip(i+15))
    a(i+16) = b(ip(i+16))
    a(i+17) = b(ip(i+17))
    a(i+18) = b(ip(i+18))
    a(i+19) = b(ip(i+19))
    a(i+20) = b(ip(i+20))
    a(i+21) = b(ip(i+21))
    a(i+22) = b(ip(i+22))
    a(i+23) = b(ip(i+23))
    a(i+24) = b(ip(i+24))
    a(i+25) = b(ip(i+25))
    a(i+26) = b(ip(i+26))
    a(i+27) = b(ip(i+27))
    a(i+28) = b(ip(i+28))
    a(i+29) = b(ip(i+29))
    a(i+30) = b(ip(i+30))
    a(i+31) = b(ip(i+31))
  end do
!$omp end parallel do
  t1 = omp_get_wtime()
  if (len_1d > 1000) write(*,*) "THREADS=", nt, " WALL_NS=", (t1-t0)*1e9, " len=", len_1d
  flush(6)
  do i = m32 + 1, len_1d
    a(i) = b(ip(i))
  end do
end subroutine tsvc_2_vag_fp64
