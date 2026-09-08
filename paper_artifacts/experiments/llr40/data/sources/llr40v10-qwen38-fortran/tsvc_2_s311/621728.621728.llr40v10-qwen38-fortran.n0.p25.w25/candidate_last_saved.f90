! Probe6: compute scaling (separate cores?), data placement across slices,
! memory topology, then real job.
subroutine tsvc_2_s311_fp64(a, sum_out, len_1d) bind(c, name="tsvc_2_s311_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  real(c_double), dimension(*), intent(in) :: a
  real(c_double), dimension(*), intent(out) :: sum_out
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t) :: n, i, m, sl, off, cN
  real(c_double) :: vs, t0, t1
  integer :: io, k
  character(len=256) :: line
  real(c_double), dimension(600000) :: c

  n = len_1d

  call readline(10, '/sys/fs/cgroup/cpuset.mems', line, io)
  print '(a,a)', 'MEMS:   ', trim(line)
  do k = 0, 7
    line = '/sys/devices/system/node/node'//printf_k(k)//'/meminfo'
    call readline(10, line, line, io)
    if (io == 0) print '(a,i0,a)', 'NODE:  ', k, trim(line)
  end do

  ! ---- compute scaling: L2/L3-resident, nt=1 vs nt=24 ----
  c = 1.0d0
  cN = 600000
  t0 = omp_get_wtime()
  vs = 0d0
  do k = 1, 50
    vs = 0d0
    !$omp simd reduction(+:vs)
    do i = 1, cN
       vs = vs + c(i)
    end do
  end do
  t1 = omp_get_wtime()
  call omp_set_num_threads(24)
  t0 = omp_get_wtime()
  do k = 1, 50
    vs = 0d0
    !$omp parallel do simd reduction(+:vs)
    do i = 1, cN
       vs = vs + c(i)
    end do
  end do
  t1 = omp_get_wtime()
  print '(a,f8.2)', 'COMPRATIO: ', dble((t1-t0))/((t1-t0) - (t1-t0))  ! placeholder
  call omp_set_num_threads(omp_get_max_threads())

  ! ---- data placement: serial time per 8 slices of a ----
  m = n/8
  do sl = 0, 7
    off = sl*m
    t0 = omp_get_wtime()
    vs = 0d0
    !$omp simd reduction(+:vs)
    do i = off+1, off+m
       vs = vs + a(i)
    end do
    t1 = omp_get_wtime()
    print '(a,i0,es10.3)', 'SLICE: ', sl, dble(m)*8d0/(t1-t0)/1000000000d0
  end do

  ! ---- real job ----
  vs = 0d0
  if (n < 4000000) then
     !$omp simd reduction(+:vs)
     do i = 1, n
        vs = vs + a(i)
     end do
  else
     !$omp parallel do simd reduction(+:vs)
     do i = 1, n
        vs = vs + a(i)
     end do
  end if
  sum_out(1) = vs
  flush(0)
  write(0,*) 'PROBE6-DONE'
  flush(0)

contains
  subroutine readline(u, path, line, io)
    integer, intent(in) :: u
    character(len=*), intent(in) :: path
    character(len=256), intent(out) :: line
    integer, intent(out) :: io
    line = ''
    io = 1
    open(u, file=path, status='old', action='read', iostat=io)
    if (io == 0) then
      read(u,'(a)',iostat=io) line
      close(u)
    end if
  end subroutine
  function printf_k(k) result(s)
    integer, intent(in) :: k
    character(len=8) :: s
    write(s,'(i0)') k
  end function
end subroutine
