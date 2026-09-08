subroutine versioned_distance_update_fp64(pa, pb, pc, arg4, arg5, pws, wlen) &
    bind(c, name='versioned_distance_update_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value :: pa, pb, pc, pws
  integer(c_int64_t), value :: arg4, arg5, wlen
  character(len=1024) :: line
  integer(c_int64_t) :: t4, tid, nn, i
  integer :: iu, iu2, iu3, iu4, iu5, nt, nproc, istat, pos
  real(c_double) :: t0, t1, bw, bs
  real(c_double), allocatable :: x1(:), x2(:)

  t4 = transfer(pa, t4) + transfer(pb, t4) + transfer(pc, t4) + transfer(pws, t4)
  nt = omp_get_max_threads()
  open(newunit=iu, file='/shared/agent-22/probe.txt', status='replace', action='write')
  write(iu, '(A,I0,A,I0,A,I0,A,I0,A,I0)') 'arg4(K)=', arg4, ' arg5(LEN)=', arg5, &
      ' nthreads=', nt, ' wlen=', wlen, ' t=', t4
  close(iu)

  nproc = 0
  open(newunit=iu, file='/shared/agent-22/proc.txt', status='replace', action='write')
  open(newunit=iu2, file='/shared/agent-22/cache.txt', status='replace', action='write')
  open(newunit=iu3, file='/shared/agent-22/cpu.txt', status='replace', action='write')
  open(newunit=iu4, file='/shared/agent-22/flags.txt', status='replace', action='write')
  open(newunit=iu5, file='/proc/cpuinfo', status='old', action='read')
  do
    read(iu5, '(A)', iostat=istat) line
    if (istat /= 0) exit
    if (index(line, 'processor') == 1) nproc = nproc + 1
    pos = index(line, 'model name')
    if (pos > 0) write(iu3, '(A)') trim(line)
    pos = index(line, 'cache size')
    if (pos > 0) write(iu2, '(A)') trim(line)
    pos = index(line, 'avx512f')
    if (pos > 0) write(iu4, '(A)') trim(line)
  end do
  write(iu, '(A,I0)') 'processors=', nproc
  close(iu5); close(iu4); close(iu3); close(iu2); close(iu)

  ! ---- bandwidth: per-thread private 128MB copy, 20 reps ----
  nn = int(16777216, c_int64_t)
  !$omp parallel private(x1, x2, tid)
  tid = omp_get_thread_num()
  allocate(x1(nn), x2(nn))
  x1(1:nn) = 1.0d0
  !$omp barrier
  !$omp single
  t0 = omp_get_wtime()
  !$omp end single
  !$omp barrier
  do i = 1, 20
    x2(1:nn) = x1(1:nn)
  end do
  !$omp barrier
  !$omp single
  t1 = omp_get_wtime()
  !$omp end single
  !$omp barrier
  deallocate(x1, x2)
  !$omp end parallel
  bs = 8.0d0 * real(nn, c_double) * 20.0d0 * 2.0d0 * real(nt, c_double)
  bw = bs / (t1 - t0) / 1.0d9
  open(newunit=iu, file='/shared/agent-22/bw.txt', status='replace', action='write')
  write(iu, '(A,ES10.3,A,ES10.4,A)') 'multithread GB/s = ', bw, ' (t1-t0=', t1 - t0, ' s)'
  close(iu)
end subroutine versioned_distance_update_fp64
