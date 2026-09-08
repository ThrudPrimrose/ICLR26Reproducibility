subroutine tsvc_2_vag_fp64(a, b, ip, len1d) bind(C, name="tsvc_2_vag_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value :: len1d
  real(c_double), intent(out) :: a(len1d)
  real(c_double), intent(in) :: b(len1d)
  integer(c_int32_t), intent(in) :: ip(len1d)
  integer(c_int64_t) :: i, k
  real(c_double) :: t0, t1
  integer(c_int32_t) :: ts(10)
  interface
    function c_wtime() bind(C, name="omp_get_wtime")
      use iso_c_binding
      real(c_double) c_wtime
    end function
    subroutine c_setnt(n) bind(C, name="omp_set_num_threads")
      use iso_c_binding
      integer(c_int32_t), value :: n
    end subroutine
  end interface
  ts = [1,2,4,8,12,16,24,32,48,96]
  do k = 1, 10
    call c_setnt(ts(k))
    t0 = c_wtime()
    !$omp parallel do default(none) shared(a,b,ip,len1d) private(i)
    do i = 1, len1d
      a(i) = b(ip(i))
    end do
    !$omp end parallel do
    t1 = c_wtime()
    write(6,'(A,I3,A,F9.3)') ' T=', ts(k), ' ms=', (t1-t0)*1000.0d0
  end do
  flush(6)
end subroutine tsvc_2_vag_fp64
