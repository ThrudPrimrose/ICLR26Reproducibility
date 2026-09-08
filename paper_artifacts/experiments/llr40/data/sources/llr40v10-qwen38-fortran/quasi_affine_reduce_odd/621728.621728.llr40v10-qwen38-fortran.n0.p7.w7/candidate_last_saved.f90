subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(C, name='quasi_affine_reduce_odd_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in)    :: a(*)
  real(c_double), intent(out)   :: out(*)
  integer(c_int64_t), value     :: len_1d
  character(len=512) :: line, s
  integer :: u, io, nt, nt2, tid, nt_req
  integer(c_int64_t) :: k, lo, hi, i, chunk, arr, t0i, t1i
  real(c_double) :: acc, part, t0, t1
  integer(c_int64_t) :: tnts
  real(c_double) :: res(6)
  integer :: j
  integer(c_int64_t) :: nts(6)
  nts = [1_c_int64_t, 2_c_int64_t, 4_c_int64_t, 8_c_int64_t, 16_c_int64_t, 24_c_int64_t]

  call rfile('/sys/fs/cgroup/cpuset.cpus.effective', s)
  print '(A,A)', 'PROBE2 cpuset.effective=', trim(s)
  call rfile('/sys/fs/cgroup/cpu.max', s)
  print '(A,A)', 'PROBE2 cpu.max=', trim(s)
  call rfile('/sys/fs/cgroup/cpuset/cpuset.cpus', s)
  print '(A,A)', 'PROBE2 v1 cpuset.cpus=', trim(s)
  call rfile('/proc/self/cpuset', s)
  print '(A,A)', 'PROBE2 self.cpuset=', trim(s)
  print '(A,I0)', 'PROBE2 num_procs=', omp_get_num_procs()

  k = len_1d / 2
  arr = len_1d
  do j = 1, 6
    nt_req = int(nts(j))
    t0 = omp_get_wtime()
    !$omp parallel default(none) shared(a, k, nt_req, res, j) private(part, lo, hi, i, chunk, nt2, tid) reduction(+:acc) num_threads(nt_req)
    acc = 0.0d0
    nt2 = omp_get_num_threads()
    tid = omp_get_thread_num()
    chunk = (k + nt2 - 1) / nt2
    lo = chunk * tid + 1
    hi = min(k, chunk * (tid + 1))
    if (lo <= hi) then
      part = 0.0d0
      do i = 2 * lo, 2 * hi, 2
        part = part + a(i)
      end do
      acc = acc + part
    end if
    !$omp end parallel
    t1 = omp_get_wtime()
    res(j) = (t1 - t0)
    print '(A,I0,A,F9.4,A,A)', 'PROBE2 threads=', nt_req, ' time=', t1-t0, ' s  GB/s=', (k*8.0d0)/(t1-t0)
  end do
  out(1) = res(1)  ! placeholder (1-thread result)
  flush(6)
contains
  subroutine rfile(fn, out_s)
    character(len=*), intent(in) :: fn
    character(len=512), intent(out) :: out_s
    integer :: uu
    out_s = ''
    open(newunit=uu, file=fn, status='old', action='read', iostat=io)
    if (io == 0) then
      read(uu, '(a)', iostat=io) line
      if (io == 0) out_s = trim(line)
      close(uu)
    end if
  end subroutine rfile
end subroutine quasi_affine_reduce_odd_fp64
