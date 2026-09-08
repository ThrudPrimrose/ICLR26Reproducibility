subroutine fuse_stencil_through_transient_fp64(a, out, len_1d) bind(C, name='fuse_stencil_through_transient_fp64')
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: out(*)
  integer(c_int64_t), value   :: len_1d
  integer(c_int64_t) :: i
  real(c_double) :: t0, x1, x2, x3, x4, y1, y2
  integer, save :: ncall = 0
  integer :: u, io
  character(len=128) :: line
  ncall = ncall + 1

  if (ncall == 1) then
     open(newunit=u, file='/proc/cpuinfo', status='old', action='read')
     do
        read(u, '(a)', iostat=io) line
        if (io /= 0) exit
        if (index(line, 'model name') > 0) exit
     end do
     print '(a,a)', 'CPUINFO: ', trim(line)
     close(u)
  end if

  t0 = omp_get_wtime()
  !$omp parallel do num_threads(24) default(none) shared(a, out, len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  x1 = omp_get_wtime() - t0
  t0 = omp_get_wtime()
  !$omp parallel do num_threads(24) default(none) shared(a, out, len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  x2 = omp_get_wtime() - t0
  t0 = omp_get_wtime()
  !$omp parallel do num_threads(24) default(none) shared(a, out, len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  x3 = omp_get_wtime() - t0
  t0 = omp_get_wtime()
  !$omp parallel do num_threads(24) default(none) shared(a, out, len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  x4 = omp_get_wtime() - t0
  t0 = omp_get_wtime()
  !$omp parallel do num_threads(8) default(none) shared(a, out, len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  y1 = omp_get_wtime() - t0
  t0 = omp_get_wtime()
  !$omp parallel do num_threads(8) default(none) shared(a, out, len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  y2 = omp_get_wtime() - t0

  print '(a,i0,6(a,f10.3))', ' NCALL=', ncall, ' x24_1=', x1*1000.0d0, ' x24_2=', x2*1000.0d0, ' x24_3=', x3*1000.0d0, ' x24_4=', x4*1000.0d0, ' y8_1=', y1*1000.0d0, ' y8_2=', y2*1000.0d0
  flush(6)
end subroutine fuse_stencil_through_transient_fp64
