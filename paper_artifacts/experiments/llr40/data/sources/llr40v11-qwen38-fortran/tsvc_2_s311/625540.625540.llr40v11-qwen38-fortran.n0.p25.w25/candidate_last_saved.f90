subroutine tsvc_2_s311_fp64(a, sum_out, len_) bind(C, name='tsvc_2_s311_fp64')
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_int
  use, intrinsic :: omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_
  real(c_double), intent(in) :: a(len_)
  real(c_double), intent(inout) :: sum_out(len_)
  double precision :: s, t0, t1
  integer(c_int64_t) :: i
  integer :: rep, tid
  double precision, save :: pt(25)
  integer, save :: pc(25)
  interface
    function sched_getcpu() bind(C, name='sched_getcpu') result(rc)
      use, intrinsic :: iso_c_binding, only: c_int
      integer(c_int) :: rc
    end function sched_getcpu
  end interface

  s = 0.0d0
  do i = 1, len_
    s = s + a(i)
  end do
  sum_out(1) = s
  write(*, '(A,I0)') 'LEN=', len_

  !$omp parallel num_threads(24) default(none) shared(a, len_, s, pt, pc) &
  !$omp& private(tid, t0, t1)
    tid = omp_get_thread_num()
    if (tid < 24) then
      !$omp barrier
      do rep = 1, 2
        call chunk_sum(a, len_, 24, tid, s)
      end do
      do rep = 1, 12
        !$omp barrier
        t0 = omp_get_wtime()
        call chunk_sum(a, len_, 24, tid, s)
        t1 = omp_get_wtime()
        if (tid == 0) then
          write(*, '(A,I3,A,F12.3)') 'REP=', rep, ' WALL_MS=', (t1 - t0) * 1000.0d0
        end if
        if (rep == 12) then
          pt(tid + 1) = (t1 - t0) * 1000.0d0
          pc(tid + 1) = sched_getcpu()
        end if
      end do
      !$omp barrier
    end if
  !$omp end parallel

  do tid = 1, 24
    write(*, '(A,I3,A,I3,A,F10.3)') 'TH=', tid - 1, ' CPU=', pc(tid), ' T_MS=', pt(tid)
  end do
  flush(6)
end subroutine tsvc_2_s311_fp64

subroutine chunk_sum(a, len_, nt_, tid_, s_)
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), intent(in) :: len_
  real(c_double), intent(in) :: a(len_)
  integer, intent(in) :: nt_, tid_
  double precision, intent(inout) :: s_
  integer(c_int64_t) :: i, lo, hi, nchunk
  double precision :: s
  nchunk = (len_ + int(nt_, 8) - 1) / int(nt_, 8)
  lo = 1 + int(tid_, 8) * nchunk
  hi = min(len_, 1 + int(tid_ + 1, 8) * nchunk - 1)
  s = 0.0d0
  do i = lo, hi
    s = s + a(i)
  end do
  s_ = s_ + s
end subroutine chunk_sum
