subroutine tsvc_2_s3111_fp64(a, b, len_1d) bind(c, name="tsvc_2_s3111_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value     :: len_1d
  real(c_double), intent(in)    :: a(len_1d)
  real(c_double), intent(out)   :: b(2)

  integer(c_int64_t) :: i
  real(c_double)     :: s, t0, t1
  integer(c_int)     :: nt
  integer, allocatable :: nts(:)

  nts = [16, 20, 24, 28, 32, 48, 64, 96, 128, 192]
  do nt = 1, size(nts)
    call omp_set_num_threads(nts(nt))
    t0 = omp_get_wtime()
    s = 0.0d0
    !$omp parallel reduction(+:s)
    !$omp do schedule(static)
    do i = 1, len_1d
      if (a(i) > 0.0d0) then
        s = s + a(i)
      end if
    end do
    !$omp end do
    !$omp end parallel
    t1 = omp_get_wtime()
    write(*,*) 'SWEEP nt_req=', nts(nt), ' t_ms=', (t1 - t0) * 1.0d3, &
               ' BW_GBps=', real(len_1d) * 8.0d0 / 1.0d9 / max(t1 - t0, 1.0d-9)
  end do
  call omp_set_num_threads(omp_get_max_threads())
  b(1) = s
end subroutine tsvc_2_s3111_fp64
